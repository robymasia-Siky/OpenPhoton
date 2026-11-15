#include "PhotonEditor.hpp"
#include "Controller.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace OpenPhoton {

using namespace VSTGUI;

namespace {

inline CColor colBg()      { return CColor(14, 17, 24, 255); }
inline CColor colPanel()   { return CColor(22, 27, 38, 255); }
inline CColor colCard()    { return CColor(30, 36, 50, 255); }
inline CColor colBorder()  { return CColor(48, 57, 78, 255); }
inline CColor colText()    { return CColor(235, 240, 248, 255); }
inline CColor colMuted()   { return CColor(140, 152, 175, 255); }

inline CColor stemColor(int idx) {
    switch (idx) {
        case 0: return CColor(76, 201, 240, 255);   // Vocals - Cyan
        case 1: return CColor(247, 37, 133, 255);   // Drums  - Magenta
        case 2: return CColor(247, 127, 0, 255);    // Bass   - Amber
        case 3: return CColor(181, 131, 255, 255);  // Guitar - Violet
        case 4: return CColor(255, 209, 102, 255);  // Piano  - Gold
        default: return CColor(128, 237, 153, 255); // Other  - Mint
    }
}

enum ActionTags : int32_t {
    kActionLoadWav = 9001,
    kActionLoadModel = 9002,
    kActionExportStems = 9003,
    kActionInstallModel = 9004
};

class PhotonBackdropView : public CView {
public:
    explicit PhotonBackdropView(const CRect& size) : CView(size) {}

    void setTelemetry(const TelemetrySnapshot& snap) {
        snap_ = snap;
        invalid();
    }

    void draw(CDrawContext* ctx) override {
        const CRect r = getViewSize();
        ctx->setFillColor(colBg());
        ctx->drawRect(r, kDrawFilled);

        CRect header(r.left, r.top, r.right, r.top + 64);
        ctx->setFillColor(colPanel());
        ctx->drawRect(header, kDrawFilled);
        ctx->setFrameColor(colBorder());
        ctx->setLineWidth(1.0);
        ctx->drawLine(CPoint(r.left, r.top + 64), CPoint(r.right, r.top + 64));

        ctx->setFont(kNormalFont, 18, kBoldFace);
        ctx->setFontColor(CColor(76, 201, 240, 255));
        ctx->drawString("OPEN", CRect(18, 12, 84, 40), kLeftText, true);
        ctx->setFontColor(colText());
        ctx->drawString("PHOTON", CRect(74, 12, 168, 40), kLeftText, true);

        char statusBuf[160]{};
        if (snap_.isDownloading) {
            std::snprintf(statusBuf, sizeof(statusBuf), "DOWNLOADING AI MODEL: %.0f%% ...", snap_.downloadProgress * 100.0f);
        } else {
            const char* engineTag = snap_.usingDeepFilter
                ? "DEEPFILTERNET 3 AI"
                : (snap_.usingOnnx ? (snap_.usingGpu ? "ONNX CUDA GPU" : "ONNX CPU") : "SPECTRAL DSP");
            std::snprintf(statusBuf, sizeof(statusBuf), "%s  |  %s  (%.1f ms)",
                          engineTag,
                          snap_.modelName[0] ? snap_.modelName : "Built-in",
                          snap_.lastInferenceMs);
        }
        ctx->setFont(kNormalFont, 10, kNormalFace);
        ctx->setFontColor((snap_.usingGpu || snap_.usingDeepFilter) ? CColor(128, 237, 153, 255) : colMuted());
        ctx->drawString(statusBuf, CRect(18, 40, 420, 58), kLeftText, true);

        if (snap_.isDownloading) {
            CRect pBg(18, 59, 410, 63);
            ctx->setFillColor(CColor(40, 48, 66, 255));
            ctx->drawRect(pBg, kDrawFilled);
            CRect pFill(18, 59, 18 + 392.0 * std::clamp(snap_.downloadProgress, 0.0f, 1.0f), 63);
            ctx->setFillColor(CColor(76, 201, 240, 255));
            ctx->drawRect(pFill, kDrawFilled);
        }

        CRect subBar(r.left, r.top + 64, r.right, r.top + 106);
        ctx->setFillColor(CColor(18, 22, 31, 255));
        ctx->drawRect(subBar, kDrawFilled);
        ctx->setFrameColor(colBorder());
        ctx->drawLine(CPoint(r.left, r.top + 106), CPoint(r.right, r.top + 106));

        ctx->setFont(kNormalFont, 10, kBoldFace);
        ctx->setFontColor(colMuted());
        ctx->drawString("MODEL:", CRect(18, 74, 68, 96), kLeftText, true);

        CRect waveBox(18, 118, r.right - 18, 344);
        ctx->setFillColor(CColor(10, 13, 19, 255));
        ctx->drawRect(waveBox, kDrawFilled);
        ctx->setFrameColor(colBorder());
        ctx->drawRect(waveBox, kDrawStroked);

        const int activeChunks = std::max<int>(1, snap_.activeChunks);
        const CCoord chunkBarH = 8.0;
        const CCoord barW = waveBox.getWidth() - 2.0;
        for (int c = 0; c < activeChunks; ++c) {
            const CCoord x0 = waveBox.left + 1.0 + (barW * c) / activeChunks;
            const CCoord x1 = waveBox.left + 1.0 + (barW * (c + 1)) / activeChunks;
            CColor stCol(28, 33, 46, 255);
            if (snap_.chunkStates[c] == 1) stCol = CColor(247, 127, 0, 220);
            else if (snap_.chunkStates[c] == 2) stCol = CColor(247, 37, 133, 255);
            else if (snap_.chunkStates[c] == 3) stCol = CColor(76, 201, 240, 220);
            ctx->setFillColor(stCol);
            ctx->drawRect(CRect(x0, waveBox.top + 1.0, std::max(x0 + 1.0, x1 - 1.0), waveBox.top + 1.0 + chunkBarH), kDrawFilled);
        }

        ctx->setFrameColor(CColor(28, 34, 48, 255));
        for (int i = 1; i < 8; ++i) {
            const CCoord x = waveBox.left + (waveBox.getWidth() * i) / 8.0;
            ctx->drawLine(CPoint(x, waveBox.top + 10), CPoint(x, waveBox.bottom - 1));
        }

        const char* labelsMusic[6] = { "VOCALS", "DRUMS", "BASS", "GUITAR", "PIANO", "OTHER" };
        const char* labelsVoice[6] = { "CLEAN",  "---",   "---",  "---",    "---",   "NOISE" };
        const bool isVoiceRestore = (snap_.modelType == 3);
        const CCoord topWave = waveBox.top + 12.0;
        const CCoord laneH = (waveBox.bottom - topWave - 4.0) / 6.0;

        const CCoord startX = waveBox.left + 74.0;
        const CCoord waveW = waveBox.right - 14.0 - startX;

        for (int s = 0; s < 6; ++s) {
            const CCoord midY = topWave + laneH * (s + 0.5);
            const CColor c = stemColor(s);

            ctx->setFont(kNormalFont, 9, kBoldFace);
            ctx->setFontColor(c);
            ctx->drawString(isVoiceRestore ? labelsVoice[s] : labelsMusic[s],
                            CRect(waveBox.left + 8, midY - 7, waveBox.left + 70, midY + 7), kLeftText, true);

            ctx->setFrameColor(CColor(c.red, c.green, c.blue, 210));
            ctx->setLineWidth(1.4);
            for (size_t b = 0; b < kTelemetryBins; ++b) {
                const CCoord x = startX + (waveW * b) / static_cast<double>(kTelemetryBins);
                const double norm = static_cast<double>(snap_.peaks[s][b]) / 255.0;
                const double amp = std::max(1.0, norm * (laneH * 0.44));
                ctx->drawLine(CPoint(x, midY - amp), CPoint(x, midY + amp));
            }
        }

        const CCoord playX = startX + waveW * std::clamp(snap_.playheadNormalized, 0.0f, 1.0f);
        ctx->setFrameColor(CColor(255, 255, 255, 220));
        ctx->setLineWidth(1.5);
        ctx->drawLine(CPoint(playX, waveBox.top + 10), CPoint(playX, waveBox.bottom - 1));

        ctx->setFont(kNormalFont, 10, kNormalFace);
        ctx->setFontColor(colMuted());
        ctx->drawString("Seleziona modello e clicca 'Get AI Model' per scaricare HTDemucs (MIT) | Voice Restore usa DeepFilterNet3 AI 48kHz integrato",
                        CRect(18, r.bottom - 22, r.right - 18, r.bottom - 6), kLeftText, true);
        setDirty(false);
    }

    CLASS_METHODS_NOCOPY(PhotonBackdropView, CView)

private:
    TelemetrySnapshot snap_{};
};

class PhotonSlider : public CControl {
public:
    PhotonSlider(const CRect& size, IControlListener* listener, int32_t tag,
                 CColor accent, std::string title, bool isPercent200 = true)
        : CControl(size, listener, tag), accent_(accent), title_(std::move(title)), isPercent200_(isPercent200) {}

    void draw(CDrawContext* ctx) override {
        const CRect r = getViewSize();
        ctx->setFillColor(colCard());
        ctx->drawRect(r, kDrawFilled);
        ctx->setFrameColor(colBorder());
        ctx->setLineWidth(1.0);
        ctx->drawRect(r, kDrawStroked);

        const CRect track(r.left + r.getWidth() * 0.42, r.top + 26,
                          r.left + r.getWidth() * 0.58, r.bottom - 30);
        ctx->setFillColor(CColor(12, 15, 22, 255));
        ctx->drawRect(track, kDrawFilled);

        const float val = std::clamp(getValue(), 0.0f, 1.0f);
        const CCoord fillTop = track.bottom - track.getHeight() * val;
        ctx->setFillColor(accent_);
        ctx->drawRect(CRect(track.left, fillTop, track.right, track.bottom), kDrawFilled);

        ctx->setFillColor(colText());
        ctx->drawRect(CRect(track.left - 6, fillTop - 4, track.right + 6, fillTop + 4), kDrawFilled);

        ctx->setFont(kNormalFont, 10, kBoldFace);
        ctx->setFontColor(accent_);
        ctx->drawString(title_.c_str(), CRect(r.left, r.top + 6, r.right, r.top + 20), kCenterText, true);

        char buf[32]{};
        const int pct = static_cast<int>(std::round(val * (isPercent200_ ? 200.0f : 100.0f)));
        std::snprintf(buf, sizeof(buf), "%d%%", pct);
        ctx->setFont(kNormalFont, 10, kNormalFace);
        ctx->setFontColor(colText());
        ctx->drawString(buf, CRect(r.left, r.bottom - 22, r.right, r.bottom - 6), kCenterText, true);

        setDirty(false);
    }

    CMouseEventResult onMouseDown(CPoint& where, const CButtonState& buttons) override {
        if (!(buttons & kLButton)) return kMouseEventNotHandled;
        dragging_ = true;
        beginEdit();
        updateFromPoint(where);
        return kMouseEventHandled;
    }

    CMouseEventResult onMouseMoved(CPoint& where, const CButtonState& buttons) override {
        if (!dragging_) return kMouseEventNotHandled;
        updateFromPoint(where);
        return kMouseEventHandled;
    }

    CMouseEventResult onMouseUp(CPoint& where, const CButtonState& buttons) override {
        if (!dragging_) return kMouseEventNotHandled;
        updateFromPoint(where);
        dragging_ = false;
        endEdit();
        return kMouseEventHandled;
    }

    CLASS_METHODS_NOCOPY(PhotonSlider, CControl)

private:
    void updateFromPoint(const CPoint& where) {
        const CRect r = getViewSize();
        const CCoord top = r.top + 26;
        const CCoord bottom = r.bottom - 30;
        float v = static_cast<float>((bottom - where.y) / (bottom - top));
        v = std::clamp(v, 0.0f, 1.0f);
        if (v != getValue()) {
            setValue(v);
            valueChanged();
            invalid();
        }
    }

    CColor accent_;
    std::string title_;
    bool isPercent200_;
    bool dragging_ = false;
};

class PhotonButton : public CControl {
public:
    PhotonButton(const CRect& size, IControlListener* listener, int32_t tag,
                 std::string label, CColor activeColor, float targetValue = 1.0f,
                 bool isToggle = true, bool isMomentary = false)
        : CControl(size, listener, tag), label_(std::move(label)),
          activeColor_(activeColor), targetValue_(targetValue),
          isToggle_(isToggle), isMomentary_(isMomentary) {}

    void setSelected(bool sel) {
        selected_ = sel;
        invalid();
    }

    void draw(CDrawContext* ctx) override {
        const CRect r = getViewSize();
        const bool active = isMomentary_ ? pressed_ : (isToggle_ ? (getValue() >= 0.5f) : selected_);

        ctx->setFillColor(active ? activeColor_ : CColor(28, 34, 48, 255));
        ctx->drawRect(r, kDrawFilled);
        ctx->setFrameColor(active ? activeColor_ : colBorder());
        ctx->setLineWidth(1.0);
        ctx->drawRect(r, kDrawStroked);

        ctx->setFont(kNormalFont, 9, kBoldFace);
        ctx->setFontColor(active ? CColor(15, 18, 25, 255) : colText());
        ctx->drawString(label_.c_str(), r, kCenterText, true);
        setDirty(false);
    }

    CMouseEventResult onMouseDown(CPoint& where, const CButtonState& buttons) override {
        if (!(buttons & kLButton)) return kMouseEventNotHandled;
        if (isMomentary_) {
            pressed_ = true;
            setValue(1.0f);
            valueChanged();
            invalid();
            return kMouseEventHandled;
        }
        beginEdit();
        setValue(isToggle_ ? (getValue() >= 0.5f ? 0.0f : 1.0f) : targetValue_);
        valueChanged();
        endEdit();
        invalid();
        return kMouseEventHandled;
    }

    CMouseEventResult onMouseUp(CPoint& where, const CButtonState& buttons) override {
        if (isMomentary_ && pressed_) {
            pressed_ = false;
            setValue(0.0f);
            invalid();
            return kMouseEventHandled;
        }
        return kMouseEventNotHandled;
    }

    CLASS_METHODS_NOCOPY(PhotonButton, CControl)

private:
    std::string label_;
    CColor activeColor_;
    float targetValue_;
    bool isToggle_;
    bool isMomentary_;
    bool selected_ = false;
    bool pressed_ = false;
};

} // namespace

PhotonEditor::PhotonEditor(Controller* controller)
    : VSTGUIEditor(controller), photonController_(controller)
{
    Steinberg::ViewRect vr(0, 0, 1020, 610);
    setRect(vr);
}

bool PLUGIN_API PhotonEditor::open(void* parent, const PlatformType& platformType) {
    if (frame) {
        frame->forget();
        frame = nullptr;
    }

    const auto r = getRect();
    CRect size(r.left, r.top, r.right, r.bottom);
    frame = new CFrame(size, this);
    if (!frame || !frame->open(parent, platformType)) {
        return false;
    }

    controls_.clear();
    auto* backdrop = new PhotonBackdropView(size);
    backdropView_ = backdrop;
    frame->addView(backdrop);

    // Header Row 1: Mode Switcher + Clear Cache
    const char* modeLabels[3] = { "Bypass", "Capture & Demix", "Stem Playback" };
    const float modeVals[3]   = { 0.0f, 0.5f, 1.0f };
    const CColor modeCols[3]  = { CColor(160, 170, 190, 255), CColor(247, 37, 133, 255), CColor(76, 201, 240, 255) };
    for (int m = 0; m < 3; ++m) {
        const CCoord x = 440 + m * 146;
        auto* btn = new PhotonButton(CRect(x, 16, x + 140, 48), this, kParamMode + 1000 + m,
                                     modeLabels[m], modeCols[m], modeVals[m], false);
        const float curMode = static_cast<float>(controller->getParamNormalized(kParamMode));
        btn->setSelected(std::abs(curMode - modeVals[m]) < 0.2f);
        frame->addView(btn);
        controls_[kParamMode + 1000 + m] = btn;
    }

    auto* clearBtn = new PhotonButton(CRect(890, 16, 1002, 48), this, kParamClearCache,
                                      "Clear Cache", CColor(247, 127, 0, 255), 1.0f, false, true);
    frame->addView(clearBtn);
    controls_[kParamClearCache] = clearBtn;

    // Header Row 2: 4 Modelli AI + Get AI Model + WAV/ONNX/Export
    const char* modelLabels[4] = { "Music 4-Stem", "Music 6-Stem", "Voice 2-Stem", "Voice Restore" };
    const float modelVals[4]   = { 0.0f, 1.0f / 3.0f, 2.0f / 3.0f, 1.0f };
    for (int k = 0; k < 4; ++k) {
        const CCoord x = 70 + k * 108;
        auto* btn = new PhotonButton(CRect(x, 72, x + 102, 98), this, kParamModelType + 2000 + k,
                                     modelLabels[k], CColor(128, 237, 153, 255), modelVals[k], false);
        const float curMod = static_cast<float>(controller->getParamNormalized(kParamModelType));
        btn->setSelected(std::abs(curMod - modelVals[k]) < 0.15f);
        frame->addView(btn);
        controls_[kParamModelType + 2000 + k] = btn;
    }

    frame->addView(new PhotonButton(CRect(520, 72, 636, 98), this, kActionInstallModel,
                                    "Get AI Model", CColor(255, 209, 102, 255), 1.0f, false, true));
    frame->addView(new PhotonButton(CRect(644, 72, 756, 98), this, kActionLoadWav,
                                    "Load WAV...", CColor(76, 201, 240, 255), 1.0f, false, true));
    frame->addView(new PhotonButton(CRect(764, 72, 876, 98), this, kActionLoadModel,
                                    "Load .onnx...", CColor(181, 131, 255, 255), 1.0f, false, true));
    frame->addView(new PhotonButton(CRect(884, 72, 1002, 98), this, kActionExportStems,
                                    "Export WAVs...", CColor(128, 237, 153, 255), 1.0f, false, true));

    // Bottom Mixer: 6 Stems + Strength + Dry/Wet (8 colonne da 116px)
    const char* stemTitles[6] = { "VOCALS", "DRUMS", "BASS", "GUITAR", "PIANO", "OTHER" };
    for (int i = 0; i < 6; ++i) {
        const CCoord x = 18 + i * 124;
        const CColor c = stemColor(i);

        auto* slider = new PhotonSlider(CRect(x, 358, x + 114, 534), this, kParamVocalsGain + i, c, stemTitles[i], true);
        slider->setValue(static_cast<float>(controller->getParamNormalized(kParamVocalsGain + i)));
        frame->addView(slider);
        controls_[kParamVocalsGain + i] = slider;

        auto* muteBtn = new PhotonButton(CRect(x, 542, x + 54, 572), this, kParamVocalsMute + i,
                                         "M", CColor(239, 71, 111, 255), 1.0f, true);
        muteBtn->setValue(static_cast<float>(controller->getParamNormalized(kParamVocalsMute + i)));
        frame->addView(muteBtn);
        controls_[kParamVocalsMute + i] = muteBtn;

        auto* soloBtn = new PhotonButton(CRect(x + 60, 542, x + 114, 572), this, kParamVocalsSolo + i,
                                         "S", CColor(255, 209, 102, 255), 1.0f, true);
        soloBtn->setValue(static_cast<float>(controller->getParamNormalized(kParamVocalsSolo + i)));
        frame->addView(soloBtn);
        controls_[kParamVocalsSolo + i] = soloBtn;
    }

    auto* strength = new PhotonSlider(CRect(768, 358, 880, 572), this, kParamDenoiseStrength,
                                      CColor(181, 131, 255, 255), "STRENGTH", false);
    strength->setValue(static_cast<float>(controller->getParamNormalized(kParamDenoiseStrength)));
    frame->addView(strength);
    controls_[kParamDenoiseStrength] = strength;

    auto* dryWet = new PhotonSlider(CRect(890, 358, 1002, 572), this, kParamDryWet,
                                    CColor(120, 160, 255, 255), "DRY / WET", false);
    dryWet->setValue(static_cast<float>(controller->getParamNormalized(kParamDryWet)));
    frame->addView(dryWet);
    controls_[kParamDryWet] = dryWet;

    timer_ = makeOwned<CVSTGUITimer>([this](CVSTGUITimer*) {
        if (photonController_ && backdropView_) {
            photonController_->sendCommandMessage("PollTelemetry");
            auto snap = photonController_->getLatestTelemetry();
            if (auto* bv = dynamic_cast<PhotonBackdropView*>(backdropView_)) {
                bv->setTelemetry(snap);
            }
        }
    }, 66);
    timer_->start();

    return true;
}

void PLUGIN_API PhotonEditor::close() {
    if (timer_) {
        timer_->stop();
        timer_ = nullptr;
    }
    backdropView_ = nullptr;
    controls_.clear();
    if (frame) {
        auto* oldFrame = frame;
        frame = nullptr;
        oldFrame->forget();
    }
}

void PhotonEditor::openFileAction(int actionTag) {
    if (!frame || !photonController_) return;
    const auto style = (actionTag == kActionExportStems)
        ? CNewFileSelector::kSelectDirectory
        : CNewFileSelector::kSelectFile;

    if (auto sel = VSTGUI::owned(CNewFileSelector::create(frame, style))) {
        if (actionTag == kActionLoadWav) {
            sel->setTitle("Seleziona file audio WAV");
            sel->addFileExtension(CFileExtension("WAV Audio", "wav"));
        } else if (actionTag == kActionLoadModel) {
            sel->setTitle("Seleziona modello neurale ONNX");
            sel->addFileExtension(CFileExtension("ONNX Model", "onnx"));
        } else {
            sel->setTitle("Scegli cartella per esportare i 6 Stem WAV");
        }

        sel->run([this, actionTag](CNewFileSelector* s) {
            if (!s || s->getNumSelectedFiles() == 0 || !photonController_) return;
            const char* p = s->getSelectedFile(0);
            if (!p) return;
            if (actionTag == kActionLoadWav) {
                photonController_->sendCommandMessage("LoadWav", p);
            } else if (actionTag == kActionLoadModel) {
                photonController_->sendCommandMessage("LoadModel", p);
            } else {
                photonController_->sendCommandMessage("ExportStems", p);
            }
        });
    }
}

void PhotonEditor::valueChanged(CControl* pControl) {
    if (!pControl || !controller) return;
    const int32_t tag = pControl->getTag();

    if (tag == kActionInstallModel) {
        if (pControl->getValue() >= 0.5f && photonController_) {
            photonController_->sendCommandMessage("InstallModel");
        }
        return;
    }

    if (tag == kActionLoadWav || tag == kActionLoadModel || tag == kActionExportStems) {
        if (pControl->getValue() >= 0.5f) {
            openFileAction(tag);
        }
        return;
    }

    Steinberg::Vst::ParamID paramId = static_cast<Steinberg::Vst::ParamID>(tag);
    if (tag >= kParamMode + 1000 && tag <= kParamMode + 1002) {
        paramId = kParamMode;
    } else if (tag >= kParamModelType + 2000 && tag <= kParamModelType + 2003) {
        paramId = kParamModelType;
    }

    const auto val = static_cast<Steinberg::Vst::ParamValue>(pControl->getValue());
    controller->beginEdit(paramId);
    controller->setParamNormalized(paramId, val);
    controller->performEdit(paramId, val);
    controller->endEdit(paramId);

    if (paramId == kParamMode || paramId == kParamModelType) {
        syncParameterToUI(paramId, val);
    }
}

void PhotonEditor::syncParameterToUI(Steinberg::Vst::ParamID tag, Steinberg::Vst::ParamValue value) {
    if (!frame) return;
    if (tag == kParamMode) {
        const float vals[3] = { 0.0f, 0.5f, 1.0f };
        for (int m = 0; m < 3; ++m) {
            if (auto it = controls_.find(kParamMode + 1000 + m); it != controls_.end()) {
                if (auto* btn = dynamic_cast<PhotonButton*>(it->second)) {
                    btn->setSelected(std::abs(static_cast<float>(value) - vals[m]) < 0.2f);
                }
            }
        }
        return;
    }
    if (tag == kParamModelType) {
        const float vals[4] = { 0.0f, 1.0f / 3.0f, 2.0f / 3.0f, 1.0f };
        for (int m = 0; m < 4; ++m) {
            if (auto it = controls_.find(kParamModelType + 2000 + m); it != controls_.end()) {
                if (auto* btn = dynamic_cast<PhotonButton*>(it->second)) {
                    btn->setSelected(std::abs(static_cast<float>(value) - vals[m]) < 0.15f);
                }
            }
        }
        return;
    }
    if (auto it = controls_.find(tag); it != controls_.end()) {
        it->second->setValue(static_cast<float>(value));
        it->second->invalid();
    }
}

} // namespace OpenPhoton
