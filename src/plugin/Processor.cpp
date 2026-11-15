#include "Processor.hpp"
#include "PluginIds.hpp"
#include "core/WavIO.hpp"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "pluginterfaces/vst/ivstprocesscontext.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#endif

namespace OpenPhoton {

using namespace Steinberg;
using namespace Steinberg::Vst;

namespace {
void preloadOnnxRuntimeFromPluginDir() {
#ifdef _WIN32
    static bool loaded = false;
    if (loaded) return;
    HMODULE hMod = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&preloadOnnxRuntimeFromPluginDir),
                           &hMod)) {
        wchar_t pathBuf[MAX_PATH]{};
        GetModuleFileNameW(hMod, pathBuf, MAX_PATH);
        auto dir = std::filesystem::path(pathBuf).parent_path();
        LoadLibraryW((dir / L"onnxruntime_providers_shared.dll").c_str());
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
        preloadOnnxRuntimeFromPluginDir();
        std::wstring modelPath;
        if (const char* env = std::getenv("OPENPHOTON_MODEL")) {
            modelPath = utf8ToWide(env);
        }
        scheduler_.start(modelPath);
    } else {
        scheduler_.stop();
    }
    return AudioEffect::setActive(state);
}

tresult PLUGIN_API Processor::process(ProcessData& data) {
    if (data.inputParameterChanges) {
        const int32 numParams = data.inputParameterChanges->getParameterCount();
        for (int32 i = 0; i < numParams; ++i) {
            auto* q = data.inputParameterChanges->getParameterData(i);
            if (!q) continue;
            ParamValue val = 0.0;
            int32 sampleOffset = 0;
            if (q->getPoint(q->getPointCount() - 1, sampleOffset, val) != kResultOk) continue;

            const ParamID id = q->getParameterId();
            if (id == kParamMode) mode_ = static_cast<int>(std::round(val * 2.0));
            else if (id == kParamDryWet) mix_.dryWet = static_cast<float>(val);
            else if (id == kParamModelType) scheduler_.setModelType(static_cast<ModelType>(std::round(val * 2.0)));
            else if (id == kParamDenoiseStrength) scheduler_.setDenoiseStrength(static_cast<float>(val));
            else if (id >= kParamVocalsGain && id <= kParamOtherGain)
                mix_.gains[id - kParamVocalsGain] = static_cast<float>(val * 2.0);
            else if (id >= kParamVocalsMute && id <= kParamOtherMute)
                mix_.mutes[id - kParamVocalsMute] = (val >= 0.5);
            else if (id >= kParamVocalsSolo && id <= kParamOtherSolo)
                mix_.solos[id - kParamVocalsSolo] = (val >= 0.5);
            else if (id == kParamClearCache && val >= 0.5)
                cache_.clearAll();
        }
    }

    if (data.numInputs == 0 || data.numOutputs == 0 || data.numSamples <= 0)
        return kResultOk;

    float* inL  = data.inputs[0].channelBuffers32[0];
    float* inR  = data.inputs[0].channelBuffers32[1];
    float* outL = data.outputs[0].channelBuffers32[0];
    float* outR = data.outputs[0].channelBuffers32[1];

    int64_t pos = fallbackCursor_;
    bool playing = true;
    if (data.processContext) {
        if (data.processContext->projectTimeSamples >= 0) {
            pos = data.processContext->projectTimeSamples;
        }
        playing = (data.processContext->state & ProcessContext::kPlaying) != 0;
    }

    if (mode_ == 1 && playing) {
        for (int32 s = 0; s < data.numSamples; ++s) {
            cache_.writeCaptureSample(pos + s, inL[s], inR[s]);
            outL[s] = inL[s];
            outR[s] = inR[s];
        }
        scheduler_.notifyPlayhead(pos);
        scheduler_.wakeWorker();
    } else if (mode_ == 2) {
        scheduler_.notifyPlayhead(pos);
        for (int32 s = 0; s < data.numSamples; ++s) {
            float mL = 0.0f, mR = 0.0f;
            if (cache_.readMixedSample(pos + s, mix_, mL, mR)) {
                outL[s] = mL;
                outR[s] = mR;
            } else {
                outL[s] = inL[s];
                outR[s] = inR[s];
            }
        }
    } else {
        if (outL != inL) std::copy_n(inL, data.numSamples, outL);
        if (outR != inR) std::copy_n(inR, data.numSamples, outR);
    }

    fallbackCursor_ = pos + data.numSamples;
    return kResultOk;
}

tresult PLUGIN_API Processor::notify(IMessage* message) {
    if (!message) return kInvalidArgument;
    const char* id = message->getMessageID();
    if (!id) return AudioEffect::notify(message);

    if (std::strcmp(id, "PollTelemetry") == 0) {
        if (auto reply = Steinberg::owned(allocateMessage())) {
            reply->setMessageID("TelemetryData");
            auto snap = scheduler_.buildTelemetry();
            reply->getAttributes()->setBinary("data", &snap, sizeof(snap));
            sendMessage(reply);
        }
        return kResultOk;
    }

    if (std::strcmp(id, "LoadWav") == 0) {
        TChar buf[1024]{};
        if (message->getAttributes()->getString("path", buf, sizeof(buf)) == kResultOk) {
            auto r = loadWavToCache(utf16ToUtf8(buf), cache_, 48000.0);
            if (r.ok) {
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
