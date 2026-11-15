#include "Controller.hpp"
#include "PhotonEditor.hpp"
#include "PluginIds.hpp"
#include "core/DebugLog.hpp"
#include "public.sdk/source/vst/vstparameters.h"
#include "pluginterfaces/base/ftypes.h"
#include <cstring>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace OpenPhoton {

using namespace Steinberg;
using namespace Steinberg::Vst;

tresult PLUGIN_API Controller::initialize(FUnknown* context) {
    logTrace("Controller::initialize start");
    tresult res = EditController::initialize(context);
    if (res != kResultOk) return res;

    auto* modeParam = new StringListParameter(STR16("Mode"), kParamMode);
    modeParam->appendString(STR16("Bypass"));
    modeParam->appendString(STR16("Capture & Demix"));
    modeParam->appendString(STR16("Stem Playback"));
    parameters.addParameter(modeParam);

    parameters.addParameter(STR16("Dry / Wet"), STR16("%"), 0, 1.0, ParameterInfo::kCanAutomate, kParamDryWet);

    auto* modelParam = new StringListParameter(STR16("Model"), kParamModelType);
    modelParam->appendString(STR16("Music 4-Stem (HTDemucs)"));
    modelParam->appendString(STR16("Music 6-Stem (+Guitar/Piano)"));
    modelParam->appendString(STR16("Voice 2-Stem (FT)"));
    modelParam->appendString(STR16("Voice Restore (DFN3)"));
    parameters.addParameter(modelParam);

    parameters.addParameter(STR16("Denoise Strength"), STR16("%"), 0, 0.7, ParameterInfo::kCanAutomate, kParamDenoiseStrength);

    parameters.addParameter(STR16("Vocals Gain"), STR16("%"), 0, 0.5, ParameterInfo::kCanAutomate, kParamVocalsGain);
    parameters.addParameter(STR16("Drums Gain"),  STR16("%"), 0, 0.5, ParameterInfo::kCanAutomate, kParamDrumsGain);
    parameters.addParameter(STR16("Bass Gain"),   STR16("%"), 0, 0.5, ParameterInfo::kCanAutomate, kParamBassGain);
    parameters.addParameter(STR16("Guitar Gain"), STR16("%"), 0, 0.5, ParameterInfo::kCanAutomate, kParamGuitarGain);
    parameters.addParameter(STR16("Piano Gain"),  STR16("%"), 0, 0.5, ParameterInfo::kCanAutomate, kParamPianoGain);
    parameters.addParameter(STR16("Other Gain"),  STR16("%"), 0, 0.5, ParameterInfo::kCanAutomate, kParamOtherGain);

    parameters.addParameter(STR16("Vocals Mute"), nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamVocalsMute);
    parameters.addParameter(STR16("Drums Mute"),  nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamDrumsMute);
    parameters.addParameter(STR16("Bass Mute"),   nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamBassMute);
    parameters.addParameter(STR16("Guitar Mute"), nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamGuitarMute);
    parameters.addParameter(STR16("Piano Mute"),  nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamPianoMute);
    parameters.addParameter(STR16("Other Mute"),  nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamOtherMute);

    parameters.addParameter(STR16("Vocals Solo"), nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamVocalsSolo);
    parameters.addParameter(STR16("Drums Solo"),  nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamDrumsSolo);
    parameters.addParameter(STR16("Bass Solo"),   nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamBassSolo);
    parameters.addParameter(STR16("Guitar Solo"), nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamGuitarSolo);
    parameters.addParameter(STR16("Piano Solo"),  nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamPianoSolo);
    parameters.addParameter(STR16("Other Solo"),  nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamOtherSolo);

    parameters.addParameter(STR16("Clear Cache"), nullptr, 1, 0.0, 0, kParamClearCache);
    logTrace("Controller::initialize end");
    return kResultOk;
}

IPlugView* PLUGIN_API Controller::createView(FIDString name) {
    logTrace("Controller::createView");
    if (FIDStringsEqual(name, ViewType::kEditor)) {
        activeEditor_ = new PhotonEditor(this);
        return activeEditor_;
    }
    return nullptr;
}

tresult PLUGIN_API Controller::setParamNormalized(ParamID tag, ParamValue value) {
    tresult res = EditController::setParamNormalized(tag, value);
    if (activeEditor_) {
        activeEditor_->syncParameterToUI(tag, value);
    }
    return res;
}

void Controller::editorDestroyed(EditorView* editor) {
    if (static_cast<EditorView*>(activeEditor_) == editor) {
        activeEditor_ = nullptr;
    }
}

void Controller::sendCommandMessage(const char* msgId, const std::string& utf8Path) {
    if (auto msg = Steinberg::owned(allocateMessage())) {
        msg->setMessageID(msgId);
        if (!utf8Path.empty()) {
#ifdef _WIN32
            const int len = MultiByteToWideChar(CP_UTF8, 0, utf8Path.c_str(), -1, nullptr, 0);
            if (len > 0) {
                std::vector<wchar_t> wbuf(static_cast<size_t>(len));
                MultiByteToWideChar(CP_UTF8, 0, utf8Path.c_str(), -1, wbuf.data(), len);
                msg->getAttributes()->setString("path", reinterpret_cast<const TChar*>(wbuf.data()));
            }
#endif
        }
        sendMessage(msg);
    }
}

tresult PLUGIN_API Controller::notify(IMessage* message) {
    if (!message) return kInvalidArgument;
    const char* id = message->getMessageID();
    if (id && std::strcmp(id, "TelemetryData") == 0) {
        const void* ptr = nullptr;
        uint32 size = 0;
        if (message->getAttributes()->getBinary("data", ptr, size) == kResultOk &&
            ptr && size == sizeof(TelemetrySnapshot)) {
            std::lock_guard<std::mutex> lk(telemetryMtx_);
            std::memcpy(&latestTelemetry_, ptr, sizeof(TelemetrySnapshot));
        }
        return kResultOk;
    }
    return EditController::notify(message);
}

TelemetrySnapshot Controller::getLatestTelemetry() const {
    std::lock_guard<std::mutex> lk(telemetryMtx_);
    return latestTelemetry_;
}

} // namespace OpenPhoton


