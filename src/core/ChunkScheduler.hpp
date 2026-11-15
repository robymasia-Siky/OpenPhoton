#pragma once

#include "StemCache.hpp"
#include "OnnxEngine.hpp"
#include "STFTProcessor.hpp"
#include "Telemetry.hpp"
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
    Voice2Stems = 1,
    VoiceRestore = 2
};

class ChunkScheduler {
public:
    explicit ChunkScheduler(StemCache& cache) : cache_(cache) {}
    ~ChunkScheduler() { stop(); }

    void start(const std::wstring& onnxModelPath) {
        stop();
        attachModel(onnxModelPath);
        running_.store(true, std::memory_order_release);
        worker_ = std::thread(&ChunkScheduler::workerLoop, this);
    }

    void loadModelRuntime(const std::wstring& onnxModelPath) {
        attachModel(onnxModelPath);
        cache_.markAllCapturedForReprocess();
        wakeWorker();
    }

    void stop() {
        running_.store(false, std::memory_order_release);
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        std::lock_guard<std::mutex> lk(engineMtx_);
        engine_.reset();
    }

    void setModelType(ModelType t) {
        if (modelType_.exchange(t, std::memory_order_acq_rel) != t) {
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

    void notifyPlayhead(int64_t samplePos) noexcept {
        playheadSample_.store(samplePos >= 0 ? samplePos : 0, std::memory_order_relaxed);
        playheadChunk_.store(samplePos >= 0 ? static_cast<size_t>(samplePos) / kChunkSamples : 0,
                             std::memory_order_relaxed);
        hasWork_.store(true, std::memory_order_release);
    }

    void wakeWorker() { cv_.notify_one(); }

    TelemetrySnapshot buildTelemetry() const {
        TelemetrySnapshot snap{};
        int32_t maxActive = 0;
        for (size_t i = 0; i < kMaxChunks; ++i) {
            const auto st = cache_.slot(i).state.load(std::memory_order_acquire);
            snap.chunkStates[i] = static_cast<uint8_t>(st);
            if (st != ChunkState::Empty) maxActive = static_cast<int32_t>(i + 1);
        }
        snap.activeChunks = std::max<int32_t>(1, maxActive);
        snap.playheadChunk = static_cast<int32_t>(playheadChunk_.load(std::memory_order_relaxed));
        const double totalSamples = static_cast<double>(snap.activeChunks) * static_cast<double>(kChunkSamples);
        snap.playheadNormalized = totalSamples > 0.0
            ? static_cast<float>(std::clamp(static_cast<double>(playheadSample_.load(std::memory_order_relaxed)) / totalSamples, 0.0, 1.0))
            : 0.0f;
        snap.lastInferenceMs = lastMs_.load(std::memory_order_relaxed);
        snap.modelType = static_cast<uint8_t>(modelType_.load(std::memory_order_relaxed));

        {
            std::lock_guard<std::mutex> lk(engineMtx_);
            snap.usingOnnx = engine_ ? 1 : 0;
            snap.usingGpu = (engine_ && engine_->activeProvider() == "CUDA") ? 1 : 0;
            std::strncpy(snap.modelName, loadedModelName_.c_str(), sizeof(snap.modelName) - 1);
        }

        for (size_t b = 0; b < kTelemetryBins; ++b) {
            const double pos = (static_cast<double>(b) / static_cast<double>(kTelemetryBins)) * static_cast<double>(snap.activeChunks);
            const size_t cIdx = std::min<size_t>(static_cast<size_t>(pos), static_cast<size_t>(snap.activeChunks - 1));
            const size_t sub = std::min<size_t>(static_cast<size_t>((pos - static_cast<double>(cIdx)) * 8.0), 7);
            const auto& slot = cache_.slot(cIdx);
            for (size_t s = 0; s < kNumStems; ++s) {
                snap.peaks[s][b] = slot.stemPeaks[s][sub];
            }
        }
        return snap;
    }

private:
    void attachModel(const std::wstring& onnxModelPath) {
        std::lock_guard<std::mutex> lk(engineMtx_);
        engine_.reset();
        loadedModelName_ = "Spectral DSP (Built-in)";
        if (!onnxModelPath.empty()) {
            try {
                engine_ = std::make_unique<OnnxEngine>(onnxModelPath, ExecutionBackend::Auto, 0);
                auto u8 = std::filesystem::path(onnxModelPath).filename().u8string();
                loadedModelName_.assign(reinterpret_cast<const char*>(u8.data()),
                                        std::min<size_t>(u8.size(), 60));
            } catch (...) {
                engine_.reset();
                loadedModelName_ = "Spectral DSP (ONNX Load Failed)";
            }
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

        while (running_.load(std::memory_order_acquire)) {
            const size_t idx = findBestChunkToProcess();
            if (idx >= kMaxChunks) {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait_for(lk, std::chrono::milliseconds(40), [&] {
                    return !running_.load() || hasWork_.exchange(false);
                });
                continue;
            }

            auto& s = cache_.slot(idx);
            s.updatePeaksFromInput();
            s.state.store(ChunkState::Computing, std::memory_order_release);

            const auto t0 = std::chrono::steady_clock::now();
            bool onnxHandled = false;

            {
                std::lock_guard<std::mutex> lk(engineMtx_);
                if (engine_) {
                    try {
                        const std::vector<int64_t> inShape = {1, static_cast<int64_t>(kNumChannels),
                                                              static_cast<int64_t>(kChunkSamples)};
                        auto stats = engine_->runAudioChunk(s.inputAudio.data(), inShape, outBuf, outShape);
                        if (outBuf.size() >= s.stemAudio.size()) {
                            std::copy_n(outBuf.data(), s.stemAudio.size(), s.stemAudio.data());
                            onnxHandled = true;
                        } else if (outBuf.size() >= kNumChannels * kChunkSamples) {
                            std::fill(s.stemAudio.begin(), s.stemAudio.end(), 0.0f);
                            std::copy_n(outBuf.data(), kNumChannels * kChunkSamples, s.stemAudio.data());
                            const size_t resBase = 3 * kNumChannels * kChunkSamples;
                            for (size_t i = 0; i < kNumChannels * kChunkSamples; ++i) {
                                s.stemAudio[resBase + i] = s.inputAudio[i] - outBuf[i];
                            }
                            onnxHandled = true;
                        }
                        lastMs_.store(static_cast<float>(stats.inferenceMs), std::memory_order_relaxed);
                    } catch (...) {
                        onnxHandled = false;
                    }
                }
            }

            if (!onnxHandled) {
                runSpectralDSP(stft, s, modelType_.load(std::memory_order_relaxed),
                               denoiseStrength_.load(std::memory_order_relaxed));
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
                float minMag = 1e9f;
                float avgMag = 0.0f;
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
                    const float pL = std::norm(specL[idx]);
                    const float pR = std::norm(specR[idx]);
                    const float nP = noiseEst[b] * noiseEst[b] * alpha;
                    const float gL = std::max(floorGain, (pL - nP) / (pL + 1e-9f));
                    const float gR = std::max(floorGain, (pR - nP) / (pR + 1e-9f));

                    outL[0][idx] = specL[idx] * gL;
                    outR[0][idx] = specR[idx] * gR;
                    outL[3][idx] = specL[idx] * (1.0f - gL);
                    outR[3][idx] = specR[idx] * (1.0f - gR);
                }
            }
        } else {
            std::vector<float> prevEnergy(bins, 0.0f);
            for (size_t f = 0; f < frames; ++f) {
                for (size_t b = 0; b < bins; ++b) {
                    const size_t idx = f * bins + b;
                    const float freq = static_cast<float>(b) * binHz;
                    const auto L = specL[idx];
                    const auto R = specR[idx];
                    const auto Mid = 0.5f * (L + R);
                    const auto Side = 0.5f * (L - R);

                    const float mMag = std::abs(Mid);
                    const float sMag = std::abs(Side);
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
                    outL[3][idx] = L * wOth; outR[3][idx] = R * wOth;
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
    mutable std::mutex engineMtx_;
    std::unique_ptr<OnnxEngine> engine_;
    std::string loadedModelName_ = "Spectral DSP (Built-in)";
    std::atomic<ModelType> modelType_{ModelType::Music4Stems};
    std::atomic<float> denoiseStrength_{0.7f};
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
