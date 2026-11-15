#pragma once

#include "public.sdk/source/vst/vstguieditor.h"
#include "vstgui/vstgui.h"
#include "PluginIds.hpp"
#include <unordered_map>

namespace OpenPhoton {

class Controller;

class PhotonEditor final : public Steinberg::Vst::VSTGUIEditor,
                           public VSTGUI::IControlListener {
public:
    explicit PhotonEditor(Controller* controller);
    ~PhotonEditor() override = default;

    bool PLUGIN_API open(void* parent, const VSTGUI::PlatformType& platformType) override;
    void PLUGIN_API close() override;
    void valueChanged(VSTGUI::CControl* pControl) override;

    void syncParameterToUI(Steinberg::Vst::ParamID tag, Steinberg::Vst::ParamValue value);
    void toggleSingleStemSolo(int stemIndex);

    DELEGATE_REFCOUNT(VSTGUIEditor)

private:
    void openFileAction(int actionTag);

    Controller* photonController_ = nullptr;
    VSTGUI::CView* backdropView_ = nullptr;
    VSTGUI::CView* waveformView_ = nullptr;
    VSTGUI::SharedPointer<VSTGUI::CVSTGUITimer> timer_;
    std::unordered_map<Steinberg::Vst::ParamID, VSTGUI::CControl*> controls_;
};

} // namespace OpenPhoton
