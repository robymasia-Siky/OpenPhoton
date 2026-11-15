#include "Controller.hpp"
#include "PhotonEditor.hpp"
#include "PluginIds.hpp"
#include "public.sdk/source/vst/vstparameters.h"
#include "pluginterfaces/base/ftypes.h"

namespace OpenPhoton {

using namespace Steinberg;
using namespace Steinberg::Vst;

tresult PLUGIN_API Controller::initialize(FUnknown* context) {
    tresult res = EditController::initialize(context);
    if (res != kResultOk) return res;

    auto* modeParam = new StringListParameter(STR16("Mode"), kParamMode);
    modeParam->appendString(STR16("Bypass"));
    modeParam->appendString(STR16("Capture & Demix"));
    modeParam->appendString(STR16("Stem Playback"));
    parameters.addParameter(modeParam);

    parameters.addParameter(STR16("Dry / Wet"), STR16("%"), 0, 1.0, ParameterInfo::kCanAutomate, kParamDryWet);

    parameters.addParameter(STR16("Vocals Gain"), STR16("%"), 0, 0.5, ParameterInfo::kCanAutomate, kParamVocalsGain);
    parameters.addParameter(STR16("Drums Gain"),  STR16("%"), 0, 0.5, ParameterInfo::kCanAutomate, kParamDrumsGain);
    parameters.addParameter(STR16("Bass Gain"),   STR16("%"), 0, 0.5, ParameterInfo::kCanAutomate, kParamBassGain);
    parameters.addParameter(STR16("Other Gain"),  STR16("%"), 0, 0.5, ParameterInfo::kCanAutomate, kParamOtherGain);

    parameters.addParameter(STR16("Vocals Mute"), nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamVocalsMute);
    parameters.addParameter(STR16("Drums Mute"),  nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamDrumsMute);
    parameters.addParameter(STR16("Bass Mute"),   nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamBassMute);
    parameters.addParameter(STR16("Other Mute"),  nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamOtherMute);

    parameters.addParameter(STR16("Vocals Solo"), nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamVocalsSolo);
    parameters.addParameter(STR16("Drums Solo"),  nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamDrumsSolo);
    parameters.addParameter(STR16("Bass Solo"),   nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamBassSolo);
    parameters.addParameter(STR16("Other Solo"),  nullptr, 1, 0.0, ParameterInfo::kCanAutomate, kParamOtherSolo);

    parameters.addParameter(STR16("Clear Cache"), nullptr, 1, 0.0, 0, kParamClearCache);
    return kResultOk;
}

IPlugView* PLUGIN_API Controller::createView(FIDString name) {
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

} // namespace OpenPhoton
