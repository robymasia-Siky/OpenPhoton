#pragma once
#include "public.sdk/source/vst/vsteditcontroller.h"
#include "core/Telemetry.hpp"
#include <mutex>
#include <string>

namespace OpenPhoton {

class PhotonEditor;

class Controller : public Steinberg::Vst::EditController {
public:
    static Steinberg::FUnknown* createInstance(void*) {
        return static_cast<Steinberg::Vst::IEditController*>(new Controller());
    }
    Steinberg::tresult PLUGIN_API initialize(Steinberg::FUnknown* context) override;
    Steinberg::IPlugView* PLUGIN_API createView(Steinberg::FIDString name) override;
    Steinberg::tresult PLUGIN_API setParamNormalized(Steinberg::Vst::ParamID tag,
                                                     Steinberg::Vst::ParamValue value) override;
    Steinberg::tresult PLUGIN_API notify(Steinberg::Vst::IMessage* message) override;
    void editorDestroyed(Steinberg::Vst::EditorView* editor) override;

    void sendCommandMessage(const char* msgId, const std::string& utf8Path = {});
    TelemetrySnapshot getLatestTelemetry() const;

private:
    PhotonEditor* activeEditor_ = nullptr;
    mutable std::mutex telemetryMtx_;
    TelemetrySnapshot latestTelemetry_{};
};

} // namespace OpenPhoton
