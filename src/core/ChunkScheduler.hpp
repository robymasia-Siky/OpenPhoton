#pragma once

#include "StemCache.hpp"
#include "OnnxEngine.hpp"
#include "STFTProcessor.hpp"
#include "Telemetry.hpp"
#include "DeepFilterBridge.hpp"
#include "ModelManager.hpp"
#include "DebugLog.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace OpenPhoton {

enum class ModelType : uint8_t {
    Music4Stems = 0,
    Music6Stems = 1,
    Voice2Stems = 2,
    VoiceRestore = 3
};

class ChunkScheduler {
public:
    explicit ChunkScheduler(StemCache& cache) : cache_(cache) {}
    ~ChunkScheduler() { stop(); }

    void start(const std::filesystem::path& pluginDir, const std::wstring& customOnnxPath) {
        stop();
        pluginDir_ = pluginDir;
        if (!customOnnxPath.empty()) {
            requestModelLoad(customOnnxPath);
        } else {
            autoSwitchModelForPreset(modelType_.load(std::memory_order_relaxed));
        }
        running_.store(true, std::memory_order_release);
        worker_ = std::thread(&ChunkScheduler::workerLoop, this);
    }

    void loadModelRuntime(const std::wstring& onnxModelPath) {
        requestModelLoad(onnxModelPath);
        cache_.markAllCapturedForReprocess();
        wakeWorker();
    }

    void installSelectedModel() {
        const int id = static_cast<int>(modelType_.load(std::memory_order_relaxed));
        if (id < 0 || id > 2) return;
        modelManager_.startDownload(id, [this](std::wstring downloadedPath) {
            requestModelLoad(downloadedPath);
            cache_.markAllCapturedForReprocess();
            wakeWorker();
        });
    }

    void stop() {
        modelManager_.cancelAndJoin();
        running_.store(false, std::memory_order_release);
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        std::lock_guard<std::mutex> lk(engineMtx_);
        engine_.reset();
    }

    void setModelType(ModelType t) {
        if (modelType_.exchange(t, std::memory_order_acq_rel) != t) {
            autoSwitchModelForPreset(t);
            cache_.markAllCapturedForReprocess();
            wakeWorker();
        }
    }

    void setDenoiseStrength(float s) noexcept {
        const float clamped = std::clamp(s, 0.0f, 1.0f);
        if (std::abs(denoiseStrength_.load(std::memory_order_relaxed) - clamped) > 0.02f) {
            denoiseStrength_.store(clamped, std::memory_order_relaxed);
            if (modelType_.load(std::memory_order_relaxed) == ModelType::VoiceRestore) {
                cache_.markAllCapturedForReprocess();
                wakeWorker();
            }
        }
    }

    void setViewWindow(float startNorm, float zoom) noexcept {
        viewStart_.store(std::clamp(startNorm, 0.0f, 1.0f), std::memory_order_relaxed);
        viewZoom_.store(std::clamp(zoom, 1.0f, 32.0f), std::memory_order_relaxed);
    }

    void notifyPlayhead(int64_t samplePos) noexcept {
        playheadSample_.store(samplePos >= 0 ? samplePos : 0, std::memory_order_relaxed);
        playheadChunk_.store(samplePos >= 0 ? static_cast<size_t>(samplePos) / kChunkSamples : 0,
                             std::memory_order_relaxed);
        hasWork_.store(true, std::memory_order_release);
    }

    void wakeWorker() { cv_.notify_one(); }

    TelemetrySnapshot buildTelemetry(int currentMode, bool internalPlaying) const {
        TelemetrySnapshot snap{};
        int32_t maxActive = 0;
        for (size_t i = 0; i < kMaxChunks; ++i) {
            const auto st = cache_.slot(i).state.load(std::memory_order_acquire);
            snap.chunkStates[i] = static_cast<uint8_t>(st);
            if (st != ChunkState::Empty) maxActive = static_cast<int32_t>(i + 1);
        }
        snap.activeChunks = std::max<int32_t>(1, maxActive);
        snap.playheadChunk = static_cast<int32_t>(playheadChunk_.load(std::memory_order_relaxed));
        const int64_t totalSamples = static_cast<int64_t>(snap.activeChunks) * static_cast<int64_t>(kChunkSamples);
        snap.totalDurationSeconds = static_cast<float>(totalSamples) / 48000.0f;
        snap.playheadNormalized = totalSamples > 0
            ? static_cast<float>(std::clamp(static_cast<double>(playheadSample_.load(std::memory_order_relaxed)) / static_cast<double>(totalSamples), 0.0, 1.0))
            : 0.0f;
        snap.lastInferenceMs = lastMs_.load(std::memory_order_relaxed);
        snap.currentMode = static_cast<uint8_t>(currentMode);
        snap.internalPlaying = internalPlaying ? 1 : 0;

        const auto mt = modelType_.load(std::memory_order_relaxed);
        snap.modelType = static_cast<uint8_t>(mt);
        snap.isDownloading = modelManager_.isDownloading() ? 1 : 0;
        snap.downloadProgress = modelManager_.progress();
        snap.usingDeepFilter = (mt == ModelType::VoiceRestore && dfBridge_.isReady()) ? 1 : 0;
        snap.modelInstalled = (mt == ModelType::VoiceRestore)
            ? (dfBridge_.isReady() ? 1 : 0)
            : (ModelManager::isModelInstalled(static_cast<int>(mt)) ? 1 : 0);

        {
            std::lock_guard<std::mutex> lk(engineMtx_);
            snap.usingOnnx = engine_ ? 1 : 0;
            snap.usingGpu = (engine_ && engine_->activeProvider() == "DirectML") ? 1 : 0;
            if (mt == ModelType::VoiceRestore && dfBridge_.isReady()) {
                std::strncpy(snap.modelName, "DeepFilterNet3 AI (48kHz)", sizeof(snap.modelName) - 1);
            } else {
                std::strncpy(snap.modelName, loadedModelName_.c_str(), sizeof(snap.modelName) - 1);
            }
        }

        const double zoom = static_cast<double>(viewZoom_.load(std::memory_order_relaxed));
        const double span = 1.0 / zoom;
        double vStart = static_cast<double>(viewStart_.load(std::memory_order_relaxed));
        if (vStart + span > 1.0) vStart = std::max(0.0, 1.0 - span);

        const double winStartSample = vStart * static_cast<double>(totalSamples);
        const double winSamples = span * static_cast<double>(totalSamples);
        const double binSamples = std::max(1.0, winSamples / static_cast<double>(kTelemetryBins));

        for (size_t b = 0; b < kTelemetryBins; ++b) {
            const int64_t s0 = static_cast<int64_t>(winStartSample + b * binSamples);
            const int64_t s1 = std::min<int64_t>(totalSamples, static_cast<int64_t>(winStartSample + (b + 1) * binSamples));
            const int64_t stride = std::max<int64_t>(1, (s1 - s0) / 24);

            for (size_t s = 0; s < kNumStems; ++s) {
                float pk = 0.0f;
                for (int64_t p = s0; p < s1; p += stride) {
                    const size_t cIdx = static_cast<size_t>(p) / kChunkSamples;
                    if (cIdx >= kMaxChunks) break;
                    const auto& slot = cache_.slot(cIdx);
                    if (!slot.allocated.load(std::memory_order_acquire)) continue;
                    const auto st = slot.state.load(std::memory_order_acquire);
                    const size_t loc = static_cast<size_t>(p) % kChunkSamples;
                    if (st == ChunkState::Ready) {
                        const size_t base = s * kNumChannels * kChunkSamples;
                        pk = std::max(pk, std::abs(slot.stemAudio[base + loc]));
                        pk = std::max(pk, std::abs(slot.stemAudio[base + kChunkSamples + loc]));
                    } else if (st == ChunkState::Captured || st == ChunkState::Computing) {
                        pk = std::max(pk, std::abs(slot.inputAudio[loc]));
                        pk = std::max(pk, std::abs(slot.inputAudio[kChunkSamples + loc]));
                    }
                }
                snap.peaks[s][b] = static_cast<uint8_t>(std::clamp(pk * 255.0f, 0.0f, 255.0f));
            }
        }
        return snap;
    }

private:
    void requestModelLoad(const std::wstring& path) {
        std::lock_guard<std::mutex> lk(pendingMtx_);
        pendingModelPath_ = path;
        modelLoadPending_.store(true, std::memory_order_release);
    }

    void autoSwitchModelForPreset(ModelType t) {
        const int id = static_cast<int>(t);
        if (id >= 0 && id <= 2 && ModelManager::isModelInstalled(id)) {
            requestModelLoad(ModelManager::getModelPath(id).wstring());
        } else {
            requestModelLoad(L"");
            std::lock_guard<std::mutex> lk(engineMtx_);
            loadedModelName_ = (t == ModelType::VoiceRestore)
                ? (dfBridge_.isReady() ? "DeepFilterNet3 AI" : "Spectral Wiener Denoise")
                : "Spectral DSP (Click Get AI Model)";
        }
    }

    void attachModelOnWorker(const std::wstring& onnxModelPath) {
        std::lock_guard<std::mutex> lk(engineMtx_);
        engine_.reset();
        if (onnxModelPath.empty()) {
            const auto t = modelType_.load(std::memory_order_relaxed);
            loadedModelName_ = (t == ModelType::VoiceRestore)
                ? (dfBridge_.isReady() ? "DeepFilterNet3 AI" : "Spectral Wiener Denoise")
                : "Spectral DSP (Click Get AI Model)";
            return;
        }
        try {
            loadedModelName_ = "Loading ONNX GPU...";
            engine_ = std::make_unique<OnnxEngine>(onnxModelPath, ExecutionBackend::Auto, 0);
            auto u8 = std::filesystem::path(onnxModelPath).filename().u8string();
            loadedModelName_.assign(reinterpret_cast<const char*>(u8.data()),
                                    std::min<size_t>(u8.size(), 60));
        } catch (...) {
            engine_.reset();
            loadedModelName_ = "ONNX Load Failed";
        }
    }

    size_t findBestChunkToProcess() const noexcept {
        const size_t current = playheadChunk_.load(std::memory_order_relaxed);
        size_t bestIdx = kMaxChunks;
        size_t bestDist = kMaxChunks + 1;

        for (size_t i = 0; i < kMaxChunks; ++i) {
            if (cache_.slot(i).state.load(std::memory_order_acquire) == ChunkState::Captured) {
                const size_t dist = (i >= current) ? (i - current) : (current - i + 10);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = i;
                }
            }
        }
        return bestIdx;
    }

    void workerLoop() {
        std::vector<float> outBuf;
        std::vector<int64_t> outShape;
        STFTProcessor stft(2048, 512);

        constexpr size_t kDemucsSamples = 343980;
        constexpr size_t kContext48k = 374400;
        constexpr size_t kPad48k = (kContext48k - kChunkSamples) / 2;
        std::vector<float> win48k(kNumChannels * kContext48k, 0.0f);
        std::vector<float> in44k(kNumChannels * kDemucsSamples, 0.0f);

        while (running_.load(std::memory_order_acquire)) {
            if (modelLoadPending_.exchange(false, std::memory_order_acq_rel)) {
                std::wstring path;
                {
                    std::lock_guard<std::mutex> lk(pendingMtx_);
                    path = pendingModelPath_;
                }
                attachModelOnWorker(path);
            }

            const size_t idx = findBestChunkToProcess();
            if (idx >= kMaxChunks) {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait_for(lk, std::chrono::milliseconds(40), [&] {
                    return !running_.load() || modelLoadPending_.load() || hasWork_.exchange(false);
                });
                continue;
            }

            auto& s = cache_.slot(idx);
            s.updatePeaksFromInput();
            s.state.store(ChunkState::Computing, std::memory_order_release);

            const auto t0 = std::chrono::steady_clock::now();
            const auto mode = modelType_.load(std::memory_order_relaxed);
            bool handled = false;

            if (mode == ModelType::VoiceRestore) {
                if (!dfBridge_.isReady()) dfBridge_.init(pluginDir_);
            }

            if (mode == ModelType::VoiceRestore && dfBridge_.isReady()) {
                std::fill(s.stemAudio.begin(), s.stemAudio.end(), 0.0f);
                dfBridge_.processStereoChunk(
                    s.inputAudio.data(),
                    s.inputAudio.data() + kChunkSamples,
                    s.stemAudio.data(),
                    s.stemAudio.data() + kChunkSamples,
                    kChunkSamples,
                    denoiseStrength_.load(std::memory_order_relaxed)
                );
                const size_t resBase = 5 * kNumChannels * kChunkSamples;
                for (size_t i = 0; i < kNumChannels * kChunkSamples; ++i) {
                    s.stemAudio[resBase + i] = s.inputAudio[i] - s.stemAudio[i];
                }
                const auto t1 = std::chrono::steady_clock::now();
                lastMs_.store(static_cast<float>(std::chrono::duration<double, std::milli>(t1 - t0).count()),
                              std::memory_order_relaxed);
                handled = true;
            }

            if (!handled && mode != ModelType::VoiceRestore) {
                std::lock_guard<std::mutex> lk(engineMtx_);
                if (engine_) {
                    try {
                        std::fill(win48k.begin(), win48k.end(), 0.0f);
                        for (size_t c = 0; c < kNumChannels; ++c) {
                            const size_t dstOff = c * kContext48k;
                            const size_t srcOff = c * kChunkSamples;
                            if (idx > 0 && cache_.slot(idx - 1).state.load(std::memory_order_acquire) != ChunkState::Empty) {
                                std::copy_n(cache_.slot(idx - 1).inputAudio.data() + srcOff + (kChunkSamples - kPad48k),
                                            kPad48k, win48k.data() + dstOff);
                            }
                            std::copy_n(s.inputAudio.data() + srcOff, kChunkSamples, win48k.data() + dstOff + kPad48k);
                            if (idx + 1 < kMaxChunks && cache_.slot(idx + 1).state.load(std::memory_order_acquire) != ChunkState::Empty) {
                                std::copy_n(cache_.slot(idx + 1).inputAudio.data() + srcOff,
                                            kPad48k, win48k.data() + dstOff + kPad48k + kChunkSamples);
                            }
                        }

                        const double ratioTo44 = static_cast<double>(kContext48k - 1) / static_cast<double>(kDemucsSamples - 1);
                        for (size_t c = 0; c < kNumChannels; ++c) {
                            const float* src = win48k.data() + c * kContext48k;
                            float* dst = in44k.data() + c * kDemucsSamples;
                            for (size_t i = 0; i < kDemucsSamples; ++i) {
                                const double p = static_cast<double>(i) * ratioTo44;
                                const size_t i0 = static_cast<size_t>(p);
                                const size_t i1 = std::min<size_t>(i0 + 1, kContext48k - 1);
                                const float frac = static_cast<float>(p - static_cast<double>(i0));
                                dst[i] = src[i0] + (src[i1] - src[i0]) * frac;
                            }
                        }

                        const std::vector<int64_t> inShape = {1, static_cast<int64_t>(kNumChannels),
                                                              static_cast<int64_t>(kDemucsSamples)};
                        auto stats = engine_->runAudioChunk(in44k.data(), inShape, outBuf, outShape);

                        const size_t numOutSources = (outShape.size() >= 2 && outShape[1] > 0)
                            ? static_cast<size_t>(outShape[1])
                            : (outBuf.size() / (kNumChannels * kDemucsSamples));

                        std::fill(s.stemAudio.begin(), s.stemAudio.end(), 0.0f);
                        const double centerStart44 = static_cast<double>(kPad48k) / ratioTo44;
                        const double ratioTo48 = static_cast<double>(kDemucsSamples - 1) / static_cast<double>(kContext48k - 1);

                        auto copyResampledStem = [&](size_t srcStemIdx, size_t dstStemIdx) {
                            const float* srcBase = outBuf.data() + srcStemIdx * kNumChannels * kDemucsSamples;
                            float* dstBase = s.stemAudio.data() + dstStemIdx * kNumChannels * kChunkSamples;
                            for (size_t c = 0; c < kNumChannels; ++c) {
                                const float* src = srcBase + c * kDemucsSamples;
                                float* dst = dstBase + c * kChunkSamples;
                                for (size_t i = 0; i < kChunkSamples; ++i) {
                                    const double p = centerStart44 + static_cast<double>(i) * ratioTo48;
                                    const size_t i0 = std::min<size_t>(static_cast<size_t>(p), kDemucsSamples - 1);
                                    const size_t i1 = std::min<size_t>(i0 + 1, kDemucsSamples - 1);
                                    const float frac = static_cast<float>(p - static_cast<double>(i0));
                                    dst[i] = src[i0] + (src[i1] - src[i0]) * frac;
                                }
                            }
                        };

                        if (numOutSources == 6) {
                            copyResampledStem(3, 0); copyResampledStem(0, 1); copyResampledStem(1, 2);
                            copyResampledStem(4, 3); copyResampledStem(5, 4); copyResampledStem(2, 5);
                            handled = true;
                        } else if (numOutSources == 4) {
                            copyResampledStem(3, 0); copyResampledStem(0, 1); copyResampledStem(1, 2);
                            copyResampledStem(2, 5);
                            handled = true;
                        } else if (numOutSources == 1) {
                            copyResampledStem(0, 0);
                            const size_t othBase = 5 * kNumChannels * kChunkSamples;
                            for (size_t i = 0; i < kNumChannels * kChunkSamples; ++i) {
                                s.stemAudio[othBase + i] = s.inputAudio[i] - s.stemAudio[i];
                            }
                            handled = true;
                        }
                        lastMs_.store(static_cast<float>(stats.inferenceMs), std::memory_order_relaxed);
                    } catch (...) {
                        handled = false;
                    }
                }
            }

            if (!handled) {
                runSpectralDSP(stft, s, mode, denoiseStrength_.load(std::memory_order_relaxed));
                const auto t1 = std::chrono::steady_clock::now();
                lastMs_.store(static_cast<float>(std::chrono::duration<double, std::milli>(t1 - t0).count()),
                              std::memory_order_relaxed);
            }

            s.updatePeaksFromStems();
            s.state.store(ChunkState::Ready, std::memory_order_release);
        }
    }

    static void runSpectralDSP(const STFTProcessor& stft, AudioChunkSlot& s,
                               ModelType mode, float strength)
    {
        std::vector<std::complex<float>> specL, specR;
        size_t framesL = 0, framesR = 0;
        stft.forward(s.inputAudio.data(), kChunkSamples, specL, framesL);
        stft.forward(s.inputAudio.data() + kChunkSamples, kChunkSamples, specR, framesR);
        const size_t frames = std::min(framesL, framesR);
        const size_t bins = stft.numBins();
        if (frames == 0) return;

        std::array<std::vector<std::complex<float>>, kNumStems> outL, outR;
        for (size_t k = 0; k < kNumStems; ++k) {
            outL[k].assign(frames * bins, {0.0f, 0.0f});
            outR[k].assign(frames * bins, {0.0f, 0.0f});
        }

        const float binHz = 48000.0f / static_cast<float>(stft.fftSize());

        if (mode == ModelType::VoiceRestore) {
            std::vector<float> noiseEst(bins, 1e9f);
            for (size_t b = 0; b < bins; ++b) {
                float minMag = 1e9f, avgMag = 0.0f;
                for (size_t f = 0; f < frames; ++f) {
                    const float m = 0.5f * (std::abs(specL[f * bins + b]) + std::abs(specR[f * bins + b]));
                    minMag = std::min(minMag, m);
                    avgMag += m;
                }
                avgMag /= static_cast<float>(frames);
                noiseEst[b] = minMag * 1.4f + avgMag * 0.05f;
            }
            const float alpha = 0.5f + strength * 3.5f;
            const float floorGain = 1.0f - strength * 0.94f;
            for (size_t f = 0; f < frames; ++f) {
                for (size_t b = 0; b < bins; ++b) {
                    const size_t idx = f * bins + b;
                    const float pL = std::norm(specL[idx]), pR = std::norm(specR[idx]);
                    const float nP = noiseEst[b] * noiseEst[b] * alpha;
                    const float gL = std::max(floorGain, (pL - nP) / (pL + 1e-9f));
                    const float gR = std::max(floorGain, (pR - nP) / (pR + 1e-9f));
                    outL[0][idx] = specL[idx] * gL; outR[0][idx] = specR[idx] * gR;
                    outL[5][idx] = specL[idx] * (1.0f - gL); outR[5][idx] = specR[idx] * (1.0f - gR);
                }
            }
        } else {
            std::vector<float> prevEnergy(bins, 0.0f);
            for (size_t f = 0; f < frames; ++f) {
                for (size_t b = 0; b < bins; ++b) {
                    const size_t idx = f * bins + b;
                    const float freq = static_cast<float>(b) * binHz;
                    const auto L = specL[idx], R = specR[idx];
                    const float mMag = std::abs(0.5f * (L + R)), sMag = std::abs(0.5f * (L - R));
                    const float centerRatio = mMag / (mMag + sMag + 1e-7f);
                    const float energy = std::norm(L) + std::norm(R);
                    const float flux = std::max(0.0f, energy - prevEnergy[b]) / (energy + 1e-7f);
                    prevEnergy[b] = energy * 0.6f + prevEnergy[b] * 0.4f;

                    float wVoc = 0.0f, wDrm = 0.0f, wBas = 0.0f, wOth = 0.0f;
                    if (mode == ModelType::Voice2Stems) {
                        const float band = (freq >= 150.0f && freq <= 7500.0f) ? 1.0f : 0.25f;
                        wVoc = std::clamp(centerRatio * centerRatio * 1.4f * band, 0.0f, 1.0f);
                        wOth = 1.0f - wVoc;
                    } else {
                        if (freq < 220.0f) {
                            wDrm = std::clamp(flux * 0.8f, 0.0f, 0.65f);
                            wBas = 1.0f - wDrm;
                        } else if (freq <= 6500.0f) {
                            wDrm = std::clamp(flux * 1.15f, 0.0f, 0.85f);
                            wVoc = (1.0f - wDrm) * std::clamp((centerRatio - 0.35f) * 2.1f, 0.0f, 1.0f);
                            wOth = std::max(0.0f, 1.0f - wDrm - wVoc);
                        } else {
                            wDrm = std::clamp(flux * 1.3f + 0.25f, 0.0f, 0.9f);
                            wOth = 1.0f - wDrm;
                        }
                    }
                    outL[0][idx] = L * wVoc; outR[0][idx] = R * wVoc;
                    outL[1][idx] = L * wDrm; outR[1][idx] = R * wDrm;
                    outL[2][idx] = L * wBas; outR[2][idx] = R * wBas;
                    outL[5][idx] = L * wOth; outR[5][idx] = R * wOth;
                }
            }
        }

        std::vector<float> tmpL, tmpR;
        for (size_t k = 0; k < kNumStems; ++k) {
            stft.inverse(outL[k], frames, kChunkSamples, tmpL);
            stft.inverse(outR[k], frames, kChunkSamples, tmpR);
            const size_t base = k * kNumChannels * kChunkSamples;
            std::copy_n(tmpL.data(), kChunkSamples, s.stemAudio.data() + base);
            std::copy_n(tmpR.data(), kChunkSamples, s.stemAudio.data() + base + kChunkSamples);
        }
    }

    StemCache& cache_;
    std::filesystem::path pluginDir_;
    DeepFilterBridge dfBridge_;
    ModelManager modelManager_;
    mutable std::mutex engineMtx_;
    std::unique_ptr<OnnxEngine> engine_;
    std::string loadedModelName_ = "Spectral DSP";
    std::mutex pendingMtx_;
    std::wstring pendingModelPath_;
    std::atomic<bool> modelLoadPending_{false};
    std::atomic<ModelType> modelType_{ModelType::Music4Stems};
    std::atomic<float> denoiseStrength_{0.7f};
    std::atomic<float> viewStart_{0.0f};
    std::atomic<float> viewZoom_{1.0f};
    std::atomic<float> lastMs_{0.0f};
    std::atomic<bool> running_{false};
    std::atomic<bool> hasWork_{false};
    std::atomic<size_t> playheadChunk_{0};
    std::atomic<int64_t> playheadSample_{0};
    std::thread worker_;
    std::mutex mtx_;
    std::condition_variable cv_;
};

} // namespace OpenPhoton
