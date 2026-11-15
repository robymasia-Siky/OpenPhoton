#include "PhotonEditor.hpp"
#include "Controller.hpp"
#include "core/DebugLog.hpp"
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
    kActionInstallModel = 9004,
    kActionPlayPause = 9005,
    kActionStop = 9006,
    kActionZoomOut = 9007,
    kActionZoomIn = 9008,
    kActionZoomReset = 9009
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
        ctx->drawString("OPEN", CRect(18, 10, 84, 38), kLeftText, true);
        ctx->setFontColor(colText());
        ctx->drawString("PHOTON", CRect(74, 10, 168, 38), kLeftText, true);

        char statusBuf[160]{};
        if (snap_.isDownloading) {
            std::snprintf(statusBuf, sizeof(statusBuf), "DOWNLOADING AI MODEL: %.0f%% ...", snap_.downloadProgress * 100.0f);
        } else {
            const char* engineTag = snap_.usingDeepFilter
                ? "DEEPFILTERNET 3 AI"
                : (snap_.usingOnnx ? (snap_.usingGpu ? "DIRECTML GPU" : "ONNX CPU") : "SPECTRAL DSP");
            std::snprintf(statusBuf, sizeof(statusBuf), "%s | %s (%.0f ms)",
                          engineTag,
                          snap_.modelName[0] ? snap_.modelName : "Built-in",
                          snap_.lastInferenceMs);
        }
        ctx->setFont(kNormalFont, 10, kNormalFace);
        ctx->setFontColor((snap_.usingGpu || snap_.usingDeepFilter) ? CColor(128, 237, 153, 255) : colMuted());
        ctx->drawString(statusBuf, CRect(18, 40, 340, 58), kLeftText, true);

        if (snap_.isDownloading) {
            ctx->setFillColor(CColor(40, 48, 66, 255));
            ctx->drawRect(CRect(18, 59, 340, 63), kDrawFilled);
            ctx->setFillColor(CColor(76, 201, 240, 255));
            ctx->drawRect(CRect(18, 59, 18 + 322.0 * std::clamp(snap_.downloadProgress, 0.0f, 1.0f), 63), kDrawFilled);
        }

        CRect subBar(r.left, r.top + 64, r.right, r.top + 106);
        ctx->setFillColor(CColor(18, 22, 31, 255));
        ctx->drawRect(subBar, kDrawFilled);
        ctx->setFrameColor(colBorder());
        ctx->drawLine(CPoint(r.left, r.top + 106), CPoint(r.right, r.top + 106));

        ctx->setFont(kNormalFont, 10, kBoldFace);
        ctx->setFontColor(colMuted());
        ctx->drawString("MODEL:", CRect(18, 74, 68, 96), kLeftText, true);

        ctx->setFont(kNormalFont, 10, kNormalFace);
        ctx->setFontColor(colMuted());
        ctx->drawString("Rotella=Scroll tracce | Ctrl+Rotella=Zoom | Shift+Rotella=Orizzontale | Click nome traccia=Solo | Click onda=Play da qui",
                        CRect(18, r.bottom - 20, r.right - 18, r.bottom - 4), kLeftText, true);
        setDirty(false);
    }

    CLASS_METHODS_NOCOPY(PhotonBackdropView, CView)

private:
    TelemetrySnapshot snap_{};
};

class PhotonWaveformView : public CView {
public:
    PhotonWaveformView(const CRect& size, Controller* ctrl, PhotonEditor* editor)
        : CView(size), ctrl_(ctrl), editor_(editor)
    {
        setMouseEnabled(true);
    }

    void setTelemetry(const TelemetrySnapshot& snap) {
        snap_ = snap;
        invalid();
    }

    float viewStart() const noexcept { return viewStart_; }
    float viewZoom() const noexcept { return viewZoom_; }

    void adjustZoom(float factor, float anchorNorm = 0.5f) {
        const float oldSpan = 1.0f / viewZoom_;
        const float newZoom = std::clamp(viewZoom_ * factor, 1.0f, 32.0f);
        const float newSpan = 1.0f / newZoom;
        const float worldAnchor = viewStart_ + anchorNorm * oldSpan;
        viewZoom_ = newZoom;
        viewStart_ = std::clamp(worldAnchor - anchorNorm * newSpan, 0.0f, std::max(0.0f, 1.0f - newSpan));
        invalid();
    }

    void resetZoom() {
        viewZoom_ = 1.0f;
        viewStart_ = 0.0f;
        invalid();
    }

    void draw(CDrawContext* ctx) override {
        const CRect r = getViewSize();
        const CCoord vScrollW = 14.0;
        const CCoord hScrollH = 14.0;
        const CCoord labelW = 92.0;

        const CRect plotArea(r.left, r.top, r.right - vScrollW, r.bottom - hScrollH);
        const CRect waveArea(plotArea.left + labelW, plotArea.top + 12.0, plotArea.right - 2.0, plotArea.bottom - 1.0);

        ctx->setFillColor(CColor(10, 13, 19, 255));
        ctx->drawRect(r, kDrawFilled);
        ctx->setFrameColor(colBorder());
        ctx->setLineWidth(1.0);
        ctx->drawRect(r, kDrawStroked);

        const int activeChunks = std::max<int>(1, snap_.activeChunks);
        const float span = 1.0f / viewZoom_;
        const CCoord barW = plotArea.getWidth() - 2.0;
        for (int c = 0; c < activeChunks; ++c) {
            const float c0 = static_cast<float>(c) / static_cast<float>(activeChunks);
            const float c1 = static_cast<float>(c + 1) / static_cast<float>(activeChunks);
            if (c1 < viewStart_ || c0 > viewStart_ + span) continue;
            const CCoord x0 = plotArea.left + 1.0 + barW * std::clamp((c0 - viewStart_) / span, 0.0f, 1.0f);
            const CCoord x1 = plotArea.left + 1.0 + barW * std::clamp((c1 - viewStart_) / span, 0.0f, 1.0f);
            CColor stCol(28, 33, 46, 255);
            if (snap_.chunkStates[c] == 1) stCol = CColor(247, 127, 0, 220);
            else if (snap_.chunkStates[c] == 2) stCol = CColor(247, 37, 133, 255);
            else if (snap_.chunkStates[c] == 3) stCol = CColor(76, 201, 240, 220);
            ctx->setFillColor(stCol);
            ctx->drawRect(CRect(x0, plotArea.top + 1.0, std::max(x0 + 1.0, x1 - 1.0), plotArea.top + 9.0), kDrawFilled);
        }

        const CCoord laneH = 72.0;
        const CCoord totalLanesH = laneH * 6.0;
        const CCoord visibleH = waveArea.getHeight();
        const CCoord maxScrollY = std::max(0.0, totalLanesH - visibleH);
        scrollY_ = std::clamp(scrollY_, 0.0, maxScrollY);

        // Clipping geometrico sicuro senza setClipRect Direct2D

        const char* labelsMusic[6] = { "VOCALS", "DRUMS", "BASS", "GUITAR", "PIANO", "OTHER" };
        const char* labelsVoice[6] = { "CLEAN",  "---",   "---",  "---",    "---",   "NOISE" };
        const bool isVoiceRestore = (snap_.modelType == 3);

        for (int s = 0; s < 6; ++s) {
            const CCoord y0 = waveArea.top + s * laneH - scrollY_;
            const CCoord y1 = y0 + laneH;
            if (y0 < waveArea.top - 1.0 || y1 > waveArea.bottom + 1.0) continue;

            const CColor c = stemColor(s);
            ctx->setFillColor(CColor(20, 25, 36, 255));
            ctx->drawRect(CRect(plotArea.left + 1.0, y0 + 1.0, plotArea.left + labelW - 2.0, y1 - 1.0), kDrawFilled);
            ctx->setFillColor(c);
            ctx->drawRect(CRect(plotArea.left + 1.0, y0 + 1.0, plotArea.left + 5.0, y1 - 1.0), kDrawFilled);

            ctx->setFont(kNormalFont, 10, kBoldFace);
            ctx->setFontColor(c);
            ctx->drawString(isVoiceRestore ? labelsVoice[s] : labelsMusic[s],
                            CRect(plotArea.left + 10.0, y0 + 14.0, plotArea.left + labelW - 4.0, y0 + 34.0), kLeftText, true);

            ctx->setFont(kNormalFont, 8, kNormalFace);
            ctx->setFontColor(colMuted());
            ctx->drawString("[Click = Solo]",
                            CRect(plotArea.left + 10.0, y0 + 36.0, plotArea.left + labelW - 4.0, y0 + 52.0), kLeftText, true);

            ctx->setFrameColor(CColor(26, 32, 44, 255));
            ctx->setLineWidth(1.0);
            ctx->drawLine(CPoint(waveArea.left, y1), CPoint(waveArea.right, y1));

            const CCoord midY = (y0 + y1) * 0.5;
            const CCoord wW = waveArea.getWidth();
            ctx->setFrameColor(CColor(c.red, c.green, c.blue, 220));
            ctx->setLineWidth(1.5);
            for (size_t b = 0; b < kTelemetryBins; ++b) {
                const CCoord x = waveArea.left + (wW * b) / static_cast<double>(kTelemetryBins);
                const double norm = static_cast<double>(snap_.peaks[s][b]) / 255.0;
                const double amp = std::max(1.0, norm * (laneH * 0.44));
                ctx->drawLine(CPoint(x, midY - amp), CPoint(x, midY + amp));
            }
        }

        if (snap_.playheadNormalized >= viewStart_ && snap_.playheadNormalized <= viewStart_ + span) {
            const CCoord px = waveArea.left + waveArea.getWidth() * ((snap_.playheadNormalized - viewStart_) / span);
            ctx->setFrameColor(CColor(255, 255, 255, 235));
            ctx->setLineWidth(1.5);
            ctx->drawLine(CPoint(px, waveArea.top), CPoint(px, waveArea.bottom));
        }



        const CRect vTrack(r.right - vScrollW, r.top, r.right, r.bottom - hScrollH);
        ctx->setFillColor(CColor(16, 20, 28, 255));
        ctx->drawRect(vTrack, kDrawFilled);
        if (totalLanesH > 0.0) {
            const CCoord thumbH = std::max(24.0, vTrack.getHeight() * std::min(1.0, visibleH / totalLanesH));
            const CCoord thumbY = vTrack.top + (maxScrollY > 0.0 ? (scrollY_ / maxScrollY) * (vTrack.getHeight() - thumbH) : 0.0);
            ctx->setFillColor(CColor(76, 201, 240, 180));
            ctx->drawRect(CRect(vTrack.left + 2.0, thumbY + 2.0, vTrack.right - 2.0, thumbY + thumbH - 2.0), kDrawFilled);
        }

        const CRect hTrack(r.left, r.bottom - hScrollH, r.right - vScrollW, r.bottom);
        ctx->setFillColor(CColor(16, 20, 28, 255));
        ctx->drawRect(hTrack, kDrawFilled);
        const CCoord thumbW = std::max(28.0, hTrack.getWidth() * span);
        const CCoord maxStart = std::max(0.0001f, 1.0f - span);
        const CCoord thumbX = hTrack.left + (span < 0.999f ? (viewStart_ / maxStart) * (hTrack.getWidth() - thumbW) : 0.0);
        ctx->setFillColor(CColor(140, 152, 175, 180));
        ctx->drawRect(CRect(thumbX + 2.0, hTrack.top + 2.0, thumbX + thumbW - 2.0, hTrack.bottom - 2.0), kDrawFilled);

        char infoBuf[64]{};
        std::snprintf(infoBuf, sizeof(infoBuf), "Zoom %.1fx | %.1fs", viewZoom_, snap_.totalDurationSeconds);
        ctx->setFont(kNormalFont, 9, kBoldFace);
        ctx->setFontColor(colMuted());
        ctx->drawString(infoBuf, CRect(plotArea.right - 130, plotArea.top + 12, plotArea.right - 8, plotArea.top + 28), kRightText, true);

        setDirty(false);
    }

    CMouseEventResult onMouseDown(CPoint& where, const CButtonState& buttons) override {
        if (!(buttons & kLButton)) return kMouseEventNotHandled;
        const CRect r = getViewSize();
        const CCoord vScrollW = 14.0;
        const CCoord hScrollH = 14.0;
        const CCoord labelW = 92.0;

        if (where.x >= r.right - vScrollW && where.y < r.bottom - hScrollH) {
            dragMode_ = DragMode::VScroll;
            updateVScrollFromY(where.y);
            return kMouseEventHandled;
        }
        if (where.y >= r.bottom - hScrollH && where.x < r.right - vScrollW) {
            dragMode_ = DragMode::HScroll;
            updateHScrollFromX(where.x);
            return kMouseEventHandled;
        }
        if (where.x < r.left + labelW && where.y >= r.top + 12.0 && where.y < r.bottom - hScrollH) {
            const int stemIdx = static_cast<int>((where.y - (r.top + 12.0) + scrollY_) / 72.0);
            if (stemIdx >= 0 && stemIdx < 6 && editor_) {
                editor_->toggleSingleStemSolo(stemIdx);
            }
            return kMouseEventHandled;
        }
        if (where.x >= r.left + labelW && where.x < r.right - vScrollW && where.y < r.bottom - hScrollH) {
            dragMode_ = DragMode::Seek;
            seekFromX(where.x);
            return kMouseEventHandled;
        }
        return kMouseEventNotHandled;
    }

    CMouseEventResult onMouseMoved(CPoint& where, const CButtonState& buttons) override {
        if (dragMode_ == DragMode::VScroll) { updateVScrollFromY(where.y); return kMouseEventHandled; }
        if (dragMode_ == DragMode::HScroll) { updateHScrollFromX(where.x); return kMouseEventHandled; }
        if (dragMode_ == DragMode::Seek)    { seekFromX(where.x); return kMouseEventHandled; }
        return kMouseEventNotHandled;
    }

    CMouseEventResult onMouseUp(CPoint& where, const CButtonState& buttons) override {
        if (dragMode_ != DragMode::None) {
            dragMode_ = DragMode::None;
            return kMouseEventHandled;
        }
        return kMouseEventNotHandled;
    }

    void onMouseWheelEvent(MouseWheelEvent& event) override {
        if (event.modifiers.has(ModifierKey::Control)) {
            const CRect r = getViewSize();
            const CCoord waveLeft = r.left + 92.0;
            const CCoord waveW = std::max(1.0, (r.right - 16.0) - waveLeft);
            const float anchor = std::clamp(static_cast<float>((event.mousePosition.x - waveLeft) / waveW), 0.0f, 1.0f);
            adjustZoom(event.deltaY > 0.0 ? 1.35f : 0.75f, anchor);
        } else if (event.modifiers.has(ModifierKey::Shift) || std::abs(event.deltaX) > std::abs(event.deltaY)) {
            const double d = (std::abs(event.deltaX) > std::abs(event.deltaY)) ? event.deltaX : event.deltaY;
            const float span = 1.0f / viewZoom_;
            viewStart_ = std::clamp(viewStart_ - static_cast<float>(d) * span * 0.15f, 0.0f, std::max(0.0f, 1.0f - span));
            invalid();
        } else {
            const CCoord visibleH = getViewSize().getHeight() - 26.0;
            const CCoord maxScrollY = std::max(0.0, 72.0 * 6.0 - visibleH);
            scrollY_ = std::clamp(scrollY_ - event.deltaY * 32.0, 0.0, maxScrollY);
            invalid();
        }
        event.consumed = true;
    }

    CLASS_METHODS_NOCOPY(PhotonWaveformView, CView)

private:
    enum class DragMode { None, VScroll, HScroll, Seek };

    void updateVScrollFromY(CCoord y) {
        const CRect r = getViewSize();
        const CCoord visibleH = r.getHeight() - 26.0;
        const CCoord maxScrollY = std::max(0.0, 72.0 * 6.0 - visibleH);
        const CCoord trackH = std::max(1.0, r.getHeight() - 14.0);
        const double norm = std::clamp((y - r.top) / trackH, 0.0, 1.0);
        scrollY_ = norm * maxScrollY;
        invalid();
    }

    void updateHScrollFromX(CCoord x) {
        const CRect r = getViewSize();
        const CCoord trackW = std::max(1.0, r.getWidth() - 14.0);
        const double norm = std::clamp((x - r.left) / trackW, 0.0, 1.0);
        const float span = 1.0f / viewZoom_;
        viewStart_ = static_cast<float>(norm) * std::max(0.0f, 1.0f - span);
        invalid();
    }

    void seekFromX(CCoord x) {
        const CRect r = getViewSize();
        const CCoord waveLeft = r.left + 92.0;
        const CCoord waveRight = r.right - 16.0;
        const double localNorm = std::clamp((x - waveLeft) / std::max(1.0, waveRight - waveLeft), 0.0, 1.0);
        const double worldNorm = static_cast<double>(viewStart_) + localNorm * (1.0 / static_cast<double>(viewZoom_));
        if (ctrl_) ctrl_->sendSeekNormalized(std::clamp(worldNorm, 0.0, 1.0));
    }

    Controller* ctrl_ = nullptr;
    PhotonEditor* editor_ = nullptr;
    TelemetrySnapshot snap_{};
    double scrollY_ = 0.0;
    float viewStart_ = 0.0f;
    float viewZoom_ = 1.0f;
    DragMode dragMode_ = DragMode::None;
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
        if (selected_ != sel) {
            selected_ = sel;
            invalid();
        }
    }

    void setLabel(const std::string& lbl) {
        if (label_ != lbl) {
            label_ = lbl;
            invalid();
        }
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
    Steinberg::ViewRect vr(0, 0, 1060, 620);
    setRect(vr);
}

bool PLUGIN_API PhotonEditor::open(void* parent, const PlatformType& platformType) {
    logTrace("PhotonEditor::open v5 start");
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

    // Header Row 1: Player Interno + Mode Switcher + Clear Cache
    auto* playBtn = new PhotonButton(CRect(348, 16, 434, 48), this, kActionPlayPause,
                                     "PLAY / PAUSE", CColor(128, 237, 153, 255), 1.0f, false, false);
    frame->addView(playBtn);
    controls_[kActionPlayPause] = playBtn;

    frame->addView(new PhotonButton(CRect(440, 16, 496, 48), this, kActionStop,
                                    "STOP", CColor(239, 71, 111, 255), 1.0f, false, true));

    const char* modeLabels[3] = { "Bypass", "Capture & Demix", "Stem Playback" };
    const float modeVals[3]   = { 0.0f, 0.5f, 1.0f };
    const CColor modeCols[3]  = { CColor(160, 170, 190, 255), CColor(247, 37, 133, 255), CColor(76, 201, 240, 255) };
    for (int m = 0; m < 3; ++m) {
        const CCoord x = 512 + m * 134;
        auto* btn = new PhotonButton(CRect(x, 16, x + 128, 48), this, kParamMode + 1000 + m,
                                     modeLabels[m], modeCols[m], modeVals[m], false);
        const float curMode = static_cast<float>(controller->getParamNormalized(kParamMode));
        btn->setSelected(std::abs(curMode - modeVals[m]) < 0.2f);
        frame->addView(btn);
        controls_[kParamMode + 1000 + m] = btn;
    }

    auto* clearBtn = new PhotonButton(CRect(924, 16, 1042, 48), this, kParamClearCache,
                                      "Clear Cache", CColor(247, 127, 0, 255), 1.0f, false, true);
    frame->addView(clearBtn);
    controls_[kParamClearCache] = clearBtn;

    // Header Row 2: 4 Modelli + Azioni + Zoom - / 1:1 / +
    const char* modelLabels[4] = { "Music 4-Stem", "Music 6-Stem", "Voice 2-Stem", "Voice Restore" };
    const float modelVals[4]   = { 0.0f, 1.0f / 3.0f, 2.0f / 3.0f, 1.0f };
    for (int k = 0; k < 4; ++k) {
        const CCoord x = 70 + k * 104;
        auto* btn = new PhotonButton(CRect(x, 72, x + 98, 98), this, kParamModelType + 2000 + k,
                                     modelLabels[k], CColor(128, 237, 153, 255), modelVals[k], false);
        const float curMod = static_cast<float>(controller->getParamNormalized(kParamModelType));
        btn->setSelected(std::abs(curMod - modelVals[k]) < 0.15f);
        frame->addView(btn);
        controls_[kParamModelType + 2000 + k] = btn;
    }

    frame->addView(new PhotonButton(CRect(494, 72, 590, 98), this, kActionInstallModel,
                                    "Get AI Model", CColor(255, 209, 102, 255), 1.0f, false, true));
    frame->addView(new PhotonButton(CRect(596, 72, 688, 98), this, kActionLoadWav,
                                    "Load WAV...", CColor(76, 201, 240, 255), 1.0f, false, true));
    frame->addView(new PhotonButton(CRect(694, 72, 788, 98), this, kActionLoadModel,
                                    "Load .onnx...", CColor(181, 131, 255, 255), 1.0f, false, true));
    frame->addView(new PhotonButton(CRect(794, 72, 896, 98), this, kActionExportStems,
                                    "Export WAVs...", CColor(128, 237, 153, 255), 1.0f, false, true));

    frame->addView(new PhotonButton(CRect(912, 72, 948, 98), this, kActionZoomOut,
                                    "-", CColor(140, 152, 175, 255), 1.0f, false, true));
    frame->addView(new PhotonButton(CRect(952, 72, 1002, 98), this, kActionZoomReset,
                                    "1:1", CColor(140, 152, 175, 255), 1.0f, false, true));
    frame->addView(new PhotonButton(CRect(1006, 72, 1042, 98), this, kActionZoomIn,
                                    "+", CColor(140, 152, 175, 255), 1.0f, false, true));

    // Viewport Waveform Scrollabile & Zoomabile
    auto* wf = new PhotonWaveformView(CRect(18, 118, 1042, 348), photonController_, this);
    waveformView_ = wf;
    frame->addView(wf);

    // Bottom Mixer: 6 Stems + Strength + Dry/Wet
    const char* stemTitles[6] = { "VOCALS", "DRUMS", "BASS", "GUITAR", "PIANO", "OTHER" };
    for (int i = 0; i < 6; ++i) {
        const CCoord x = 18 + i * 128;
        const CColor c = stemColor(i);

        auto* slider = new PhotonSlider(CRect(x, 364, x + 118, 540), this, kParamVocalsGain + i, c, stemTitles[i], true);
        slider->setValue(static_cast<float>(controller->getParamNormalized(kParamVocalsGain + i)));
        frame->addView(slider);
        controls_[kParamVocalsGain + i] = slider;

        auto* muteBtn = new PhotonButton(CRect(x, 548, x + 56, 580), this, kParamVocalsMute + i,
                                         "M", CColor(239, 71, 111, 255), 1.0f, true);
        muteBtn->setValue(static_cast<float>(controller->getParamNormalized(kParamVocalsMute + i)));
        frame->addView(muteBtn);
        controls_[kParamVocalsMute + i] = muteBtn;

        auto* soloBtn = new PhotonButton(CRect(x + 62, 548, x + 118, 580), this, kParamVocalsSolo + i,
                                         "S", CColor(255, 209, 102, 255), 1.0f, true);
        soloBtn->setValue(static_cast<float>(controller->getParamNormalized(kParamVocalsSolo + i)));
        frame->addView(soloBtn);
        controls_[kParamVocalsSolo + i] = soloBtn;
    }

    auto* strength = new PhotonSlider(CRect(794, 364, 912, 580), this, kParamDenoiseStrength,
                                      CColor(181, 131, 255, 255), "STRENGTH", false);
    strength->setValue(static_cast<float>(controller->getParamNormalized(kParamDenoiseStrength)));
    frame->addView(strength);
    controls_[kParamDenoiseStrength] = strength;

    auto* dryWet = new PhotonSlider(CRect(924, 364, 1042, 580), this, kParamDryWet,
                                    CColor(120, 160, 255, 255), "DRY / WET", false);
    dryWet->setValue(static_cast<float>(controller->getParamNormalized(kParamDryWet)));
    frame->addView(dryWet);
    controls_[kParamDryWet] = dryWet;

    timer_ = makeOwned<CVSTGUITimer>([this](CVSTGUITimer*) {
        if (!photonController_) return;
        float vStart = 0.0f, vZoom = 1.0f;
        if (auto* wfView = dynamic_cast<PhotonWaveformView*>(waveformView_)) {
            vStart = wfView->viewStart();
            vZoom = wfView->viewZoom();
        }
        photonController_->sendPollTelemetry(vStart, vZoom);
        auto snap = photonController_->getLatestTelemetry();

        if (auto* bv = dynamic_cast<PhotonBackdropView*>(backdropView_)) bv->setTelemetry(snap);
        if (auto* wfView = dynamic_cast<PhotonWaveformView*>(waveformView_)) wfView->setTelemetry(snap);

        if (auto it = controls_.find(kActionPlayPause); it != controls_.end()) {
            if (auto* pbtn = dynamic_cast<PhotonButton*>(it->second)) {
                pbtn->setSelected(snap.internalPlaying != 0);
                pbtn->setLabel(snap.internalPlaying ? "PAUSE ||" : "PLAY >");
            }
        }
        syncParameterToUI(kParamMode, static_cast<double>(snap.currentMode) * 0.5);
    }, 66);
    timer_->start();
    logTrace("PhotonEditor::open v5 OK");

    return true;
}

void PLUGIN_API PhotonEditor::close() {
    if (timer_) {
        timer_->stop();
        timer_ = nullptr;
    }
    backdropView_ = nullptr;
    waveformView_ = nullptr;
    controls_.clear();
    if (frame) {
        auto* oldFrame = frame;
        frame = nullptr;
        oldFrame->forget();
    }
}

void PhotonEditor::toggleSingleStemSolo(int stemIndex) {
    if (!controller || stemIndex < 0 || stemIndex >= 6) return;
    const bool alreadySolo = (controller->getParamNormalized(kParamVocalsSolo + stemIndex) >= 0.5);

    for (int i = 0; i < 6; ++i) {
        const auto pid = static_cast<Steinberg::Vst::ParamID>(kParamVocalsSolo + i);
        const double nextVal = (!alreadySolo && i == stemIndex) ? 1.0 : 0.0;
        controller->beginEdit(pid);
        controller->setParamNormalized(pid, nextVal);
        controller->performEdit(pid, nextVal);
        controller->endEdit(pid);
        syncParameterToUI(pid, nextVal);
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

    if (tag == kActionPlayPause && photonController_) {
        photonController_->sendCommandMessage("InternalPlayToggle");
        return;
    }
    if (tag == kActionStop && photonController_) {
        if (pControl->getValue() >= 0.5f) photonController_->sendCommandMessage("InternalStop");
        return;
    }
    if (tag == kActionZoomIn || tag == kActionZoomOut || tag == kActionZoomReset) {
        if (pControl->getValue() >= 0.5f) {
            if (auto* wf = dynamic_cast<PhotonWaveformView*>(waveformView_)) {
                if (tag == kActionZoomIn) wf->adjustZoom(1.5f);
                else if (tag == kActionZoomOut) wf->adjustZoom(0.66f);
                else wf->resetZoom();
            }
        }
        return;
    }
    if (tag == kActionInstallModel) {
        if (pControl->getValue() >= 0.5f && photonController_) {
            photonController_->sendCommandMessage("InstallModel");
        }
        return;
    }
    if (tag == kActionLoadWav || tag == kActionLoadModel || tag == kActionExportStems) {
        if (pControl->getValue() >= 0.5f) openFileAction(tag);
        return;
    }

    Steinberg::Vst::ParamID paramId = static_cast<Steinberg::Vst::ParamID>(tag);
    if (tag >= kParamMode + 1000 && tag <= kParamMode + 1002) paramId = kParamMode;
    else if (tag >= kParamModelType + 2000 && tag <= kParamModelType + 2003) paramId = kParamModelType;

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

