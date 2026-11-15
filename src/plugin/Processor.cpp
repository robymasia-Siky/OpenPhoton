#include "Processor.hpp"
#include "PluginIds.hpp"
#include "core/WavIO.hpp"
#include "core/DebugLog.hpp"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "pluginterfaces/vst/ivstprocesscontext.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

namespace OpenPhoton {

using namespace Steinberg;
using namespace Steinberg::Vst;

namespace {
std::filesystem::path getPluginDir() {
#ifdef _WIN32
    HMODULE hMod = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&getPluginDir),
                           &hMod)) {
        wchar_t pathBuf[MAX_PATH]{};
        GetModuleFileNameW(hMod, pathBuf, MAX_PATH);
        return std::filesystem::path(pathBuf).parent_path();
    }
#endif
    return {};
}

void preloadOnnxRuntimeFromPluginDir() {
#ifdef _WIN32
    static bool loaded = false;
    if (loaded) return;
    auto dir = getPluginDir();
    if (!dir.empty()) {
        LoadLibraryW((dir / L"DirectML.dll").c_str());
        LoadLibraryW((dir / L"onnxruntime.dll").c_str());
        loaded = true;
    }
#endif
}

std::string utf16ToUtf8(const TChar* str) {
    if (!str) return {};
#ifdef _WIN32
    const wchar_t* wstr = reinterpret_cast<const wchar_t*>(str);
    const int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 1) return {};
    std::string out(static_cast<size_t>(len - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, out.data(), len, nullptr, nullptr);
    return out;
#else
    return {};
#endif
}

std::wstring utf8ToWide(const std::string& utf8) {
    if (utf8.empty()) return {};
#ifdef _WIN32
    const int len = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    if (len <= 1) return {};
    std::wstring out(static_cast<size_t>(len - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, out.data(), len);
    return out;
#else
    return std::wstring(utf8.begin(), utf8.end());
#endif
}
} // namespace

Processor::Processor() {
    setControllerClass(kControllerUID);
}

tresult PLUGIN_API Processor::initialize(FUnknown* context) {
    tresult res = AudioEffect::initialize(context);
    if (res != kResultOk) return res;

    addAudioInput(STR16("Stereo In"), SpeakerArr::kStereo);
    addAudioOutput(STR16("Stereo Out"), SpeakerArr::kStereo);
    return kResultOk;
}

tresult PLUGIN_API Processor::terminate() {
    scheduler_.stop();
    return AudioEffect::terminate();
}

tresult PLUGIN_API Processor::setBusArrangements(
    SpeakerArrangement* inputs, int32 numIns,
    SpeakerArrangement* outputs, int32 numOuts)
{
    if (numIns == 1 && numOuts == 1 &&
        inputs[0] == SpeakerArr::kStereo && outputs[0] == SpeakerArr::kStereo) {
        return AudioEffect::setBusArrangements(inputs, numIns, outputs, numOuts);
    }
    return kResultFalse;
}

tresult PLUGIN_API Processor::setupProcessing(ProcessSetup& newSetup) {
    sampleRate_ = newSetup.sampleRate > 0.0 ? newSetup.sampleRate : 48000.0;
    return AudioEffect::setupProcessing(newSetup);
}

tresult PLUGIN_API Processor::setActive(TBool state) {
    if (state) {
        cache_.preallocateInitialChunks(8);
        preloadOnnxRuntimeFromPluginDir();
        std::wstring modelPath;
        if (const char* env = std::getenv("OPENPHOTON_MODEL")) {
            modelPath = utf8ToWide(env);
        }
        scheduler_.start(getPluginDir(), modelPath);
    } else {
        internalPlaying_.store(false, std::memory_order_release);
        scheduler_.stop();
    }
    return AudioEffect::setActive(state);
}

int64_t Processor::findTotalLoadedSamples() const noexcept {
    int64_t maxChunks = 1;
    for (size_t i = 0; i < kMaxChunks; ++i) {
        if (cache_.slot(i).state.load(std::memory_order_acquire) != ChunkState::Empty) {
            maxChunks = static_cast<int64_t>(i + 1);
        }
    }
    return maxChunks * static_cast<int64_t>(kChunkSamples);
}

tresult PLUGIN_API Processor::process(ProcessData& data) {
    if (data.inputParameterChanges) {
        const int32 numParams = data.inputParameterChanges->getParameterCount();
        for (int32 i = 0; i < numParams; ++i) {
            auto* q = data.inputParameterChanges->getParameterData(i);
            if (!q) continue;
            ParamValue val = 0.0;
            int32 sampleOffset = 0;
            const int32 count = q->getPointCount();
            if (count <= 0 || q->getPoint(count - 1, sampleOffset, val) != kResultOk) continue;

            const ParamID id = q->getParameterId();
            if (id == kParamMode) mode_.store(static_cast<int>(std::round(val * 2.0)), std::memory_order_relaxed);
            else if (id == kParamDryWet) mix_.dryWet = static_cast<float>(val);
            else if (id == kParamModelType) scheduler_.setModelType(static_cast<ModelType>(std::round(val * 3.0)));
            else if (id == kParamDenoiseStrength) scheduler_.setDenoiseStrength(static_cast<float>(val));
            else if (id >= kParamVocalsGain && id <= kParamOtherGain)
                mix_.gains[id - kParamVocalsGain] = static_cast<float>(val * 2.0);
            else if (id >= kParamVocalsMute && id <= kParamOtherMute)
                mix_.mutes[id - kParamVocalsMute] = (val >= 0.5);
            else if (id >= kParamVocalsSolo && id <= kParamOtherSolo)
                mix_.solos[id - kParamVocalsSolo] = (val >= 0.5);
            else if (id == kParamClearCache && val >= 0.5) {
                internalPlaying_.store(false, std::memory_order_release);
                internalCursor_.store(0, std::memory_order_release);
                cache_.clearAll();
            }
        }
    }

    if (data.numInputs == 0 || data.numOutputs == 0 || data.numSamples <= 0)
        return kResultOk;
    if (!data.inputs[0].channelBuffers32 || !data.outputs[0].channelBuffers32)
        return kResultOk;
    if (data.inputs[0].numChannels < 1 || data.outputs[0].numChannels < 1)
        return kResultOk;

    float* inL  = data.inputs[0].channelBuffers32[0];
    float* inR  = (data.inputs[0].numChannels > 1 && data.inputs[0].channelBuffers32[1])
                    ? data.inputs[0].channelBuffers32[1] : inL;
    float* outL = data.outputs[0].channelBuffers32[0];
    float* outR = (data.outputs[0].numChannels > 1 && data.outputs[0].channelBuffers32[1])
                    ? data.outputs[0].channelBuffers32[1] : nullptr;
    if (!inL || !outL) return kResultOk;

    const int mode = mode_.load(std::memory_order_relaxed);

    if (internalPlaying_.load(std::memory_order_acquire)) {
        const int64_t totalLen = findTotalLoadedSamples();
        int64_t cur = internalCursor_.load(std::memory_order_relaxed);
        if (cur >= totalLen) cur = 0;

        scheduler_.notifyPlayhead(cur);
        scheduler_.wakeWorker();

        for (int32 s = 0; s < data.numSamples; ++s) {
            int64_t p = cur + s;
            if (p >= totalLen) p %= totalLen;
            float mL = 0.0f, mR = 0.0f;
            if (cache_.readMixedSample(p, mix_, mL, mR)) {
                outL[s] = mL;
                if (outR) outR[s] = mR;
            } else {
                const size_t cIdx = static_cast<size_t>(p) / kChunkSamples;
                const size_t loc = static_cast<size_t>(p) % kChunkSamples;
                if (cIdx < kMaxChunks && cache_.slot(cIdx).allocated.load(std::memory_order_acquire) &&
                    cache_.slot(cIdx).state.load(std::memory_order_acquire) != ChunkState::Empty) {
                    outL[s] = cache_.slot(cIdx).inputAudio[loc];
                    if (outR) outR[s] = cache_.slot(cIdx).inputAudio[kChunkSamples + loc];
                } else {
                    outL[s] = 0.0f;
                    if (outR) outR[s] = 0.0f;
                }
            }
        }
        internalCursor_.store((cur + data.numSamples) % std::max<int64_t>(1, totalLen), std::memory_order_release);
        return kResultOk;
    }

    int64_t pos = fallbackCursor_;
    bool playing = true;
    if (data.processContext) {
        if (data.processContext->projectTimeSamples >= 0) {
            pos = data.processContext->projectTimeSamples;
        }
        playing = (data.processContext->state & ProcessContext::kPlaying) != 0;
    }

    if (mode == 1 && playing) {
        for (int32 s = 0; s < data.numSamples; ++s) {
            cache_.writeCaptureSample(pos + s, inL[s], inR[s]);
            outL[s] = inL[s];
            if (outR) outR[s] = inR[s];
        }
        scheduler_.notifyPlayhead(pos);
        scheduler_.wakeWorker();
    } else if (mode == 2) {
        scheduler_.notifyPlayhead(pos);
        for (int32 s = 0; s < data.numSamples; ++s) {
            float mL = 0.0f, mR = 0.0f;
            if (cache_.readMixedSample(pos + s, mix_, mL, mR)) {
                outL[s] = mL;
                if (outR) outR[s] = mR;
            } else {
                outL[s] = inL[s];
                if (outR) outR[s] = inR[s];
            }
        }
    } else {
        if (outL != inL) std::copy_n(inL, data.numSamples, outL);
        if (outR && outR != inR) std::copy_n(inR, data.numSamples, outR);
    }

    fallbackCursor_ = pos + data.numSamples;
    return kResultOk;
}

tresult PLUGIN_API Processor::notify(IMessage* message) {
    if (!message) return kInvalidArgument;
    const char* id = message->getMessageID();
    if (!id) return AudioEffect::notify(message);

    if (std::strcmp(id, "PollTelemetry") == 0) {
        double vStart = 0.0, vZoom = 1.0;
        if (message->getAttributes()->getFloat("vStart", vStart) == kResultOk &&
            message->getAttributes()->getFloat("vZoom", vZoom) == kResultOk) {
            scheduler_.setViewWindow(static_cast<float>(vStart), static_cast<float>(vZoom));
        }
        if (auto reply = Steinberg::owned(allocateMessage())) {
            reply->setMessageID("TelemetryData");
            auto snap = scheduler_.buildTelemetry(
                mode_.load(std::memory_order_relaxed),
                internalPlaying_.load(std::memory_order_relaxed)
            );
            reply->getAttributes()->setBinary("data", &snap, sizeof(snap));
            sendMessage(reply);
        }
        return kResultOk;
    }

    if (std::strcmp(id, "InternalPlayToggle") == 0) {
        const bool next = !internalPlaying_.load(std::memory_order_acquire);
        if (next) mode_.store(2, std::memory_order_relaxed);
        internalPlaying_.store(next, std::memory_order_release);
        return kResultOk;
    }

    if (std::strcmp(id, "InternalStop") == 0) {
        internalPlaying_.store(false, std::memory_order_release);
        internalCursor_.store(0, std::memory_order_release);
        scheduler_.notifyPlayhead(0);
        return kResultOk;
    }

    if (std::strcmp(id, "SeekNormalized") == 0) {
        double norm = 0.0;
        if (message->getAttributes()->getFloat("norm", norm) == kResultOk) {
            const int64_t totalLen = findTotalLoadedSamples();
            const int64_t target = static_cast<int64_t>(std::clamp(norm, 0.0, 0.999) * static_cast<double>(totalLen));
            internalCursor_.store(target, std::memory_order_release);
            scheduler_.notifyPlayhead(target);
            scheduler_.wakeWorker();
        }
        return kResultOk;
    }

    if (std::strcmp(id, "InstallModel") == 0) {
        preloadOnnxRuntimeFromPluginDir();
        scheduler_.installSelectedModel();
        return kResultOk;
    }

    if (std::strcmp(id, "LoadWav") == 0) {
        TChar buf[1024]{};
        if (message->getAttributes()->getString("path", buf, sizeof(buf)) == kResultOk) {
            auto r = loadWavToCache(utf16ToUtf8(buf), cache_, 48000.0);
            if (r.ok) {
                mode_.store(2, std::memory_order_relaxed);
                internalCursor_.store(0, std::memory_order_release);
                internalPlaying_.store(true, std::memory_order_release);
                scheduler_.notifyPlayhead(0);
                scheduler_.wakeWorker();
            }
        }
        return kResultOk;
    }

    if (std::strcmp(id, "LoadModel") == 0) {
        TChar buf[1024]{};
        if (message->getAttributes()->getString("path", buf, sizeof(buf)) == kResultOk) {
            preloadOnnxRuntimeFromPluginDir();
            scheduler_.loadModelRuntime(utf8ToWide(utf16ToUtf8(buf)));
        }
        return kResultOk;
    }

    if (std::strcmp(id, "ExportStems") == 0) {
        TChar buf[1024]{};
        if (message->getAttributes()->getString("path", buf, sizeof(buf)) == kResultOk) {
            exportStemsToFolder(utf16ToUtf8(buf), cache_, 48000.0);
        }
        return kResultOk;
    }

    return AudioEffect::notify(message);
}

} // namespace OpenPhoton
