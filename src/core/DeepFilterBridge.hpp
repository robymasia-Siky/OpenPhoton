#pragma once

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace OpenPhoton {

class DeepFilterBridge {
public:
    ~DeepFilterBridge() { unload(); }

    bool init(const std::filesystem::path& pluginDir) {
        if (initialized_) return true;
#ifdef _WIN32
        const auto dllPath = pluginDir / L"df.dll";
        const auto tarPath = pluginDir / L"DeepFilterNet3_onnx.tar.gz";
        if (!std::filesystem::exists(dllPath) || !std::filesystem::exists(tarPath)) {
            return false;
        }
        hDll_ = LoadLibraryW(dllPath.c_str());
        if (!hDll_) return false;

        fnCreate_ = reinterpret_cast<FnCreate>(GetProcAddress(hDll_, "df_create"));
        fnFrameLen_ = reinterpret_cast<FnFrameLen>(GetProcAddress(hDll_, "df_get_frame_length"));
        fnAttenLim_ = reinterpret_cast<FnAttenLim>(GetProcAddress(hDll_, "df_set_atten_lim"));
        fnPostBeta_ = reinterpret_cast<FnPostBeta>(GetProcAddress(hDll_, "df_set_post_filter_beta"));
        fnProcess_ = reinterpret_cast<FnProcess>(GetProcAddress(hDll_, "df_process_frame"));
        fnFree_ = reinterpret_cast<FnFree>(GetProcAddress(hDll_, "df_free"));

        if (!fnCreate_ || !fnFrameLen_ || !fnProcess_ || !fnFree_) {
            unload();
            return false;
        }

        const auto u8 = tarPath.u8string();
        modelTarUtf8_.assign(reinterpret_cast<const char*>(u8.data()), u8.size());
        stateL_ = fnCreate_(modelTarUtf8_.c_str(), 100.0f, "error");
        stateR_ = fnCreate_(modelTarUtf8_.c_str(), 100.0f, "error");
        if (!stateL_ || !stateR_) {
            unload();
            return false;
        }
        frameLen_ = fnFrameLen_(stateL_);
        initialized_ = (frameLen_ > 0);
        return initialized_;
#else
        (void)pluginDir;
        return false;
#endif
    }

    void unload() {
        if (fnFree_) {
            if (stateL_) fnFree_(stateL_);
            if (stateR_) fnFree_(stateR_);
        }
        stateL_ = nullptr;
        stateR_ = nullptr;
#ifdef _WIN32
        if (hDll_) {
            FreeLibrary(hDll_);
            hDll_ = nullptr;
        }
#endif
        initialized_ = false;
    }

    bool isReady() const noexcept { return initialized_; }

    void processStereoChunk(const float* inL, const float* inR,
                            float* outL, float* outR,
                            size_t numSamples, float strength)
    {
        if (!initialized_ || frameLen_ == 0) return;

        const float attenDb = 6.0f + std::clamp(strength, 0.0f, 1.0f) * 94.0f;
        const float beta = (strength > 0.75f) ? (strength - 0.75f) * 0.08f : 0.0f;
        if (fnAttenLim_) {
            fnAttenLim_(stateL_, attenDb);
            fnAttenLim_(stateR_, attenDb);
        }
        if (fnPostBeta_) {
            fnPostBeta_(stateL_, beta);
            fnPostBeta_(stateR_, beta);
        }

        // Ricrea stati puliti per ogni chunk indipendente + compensa 2 frame (20ms) di lookahead
        if (fnFree_ && fnCreate_) {
            fnFree_(stateL_);
            fnFree_(stateR_);
            stateL_ = fnCreate_(modelTarUtf8_.c_str(), attenDb, "error");
            stateR_ = fnCreate_(modelTarUtf8_.c_str(), attenDb, "error");
            if (fnPostBeta_) {
                fnPostBeta_(stateL_, beta);
                fnPostBeta_(stateR_, beta);
            }
        }

        const size_t delaySamples = frameLen_ * 2;
        std::vector<float> tmpIn(frameLen_, 0.0f);
        std::vector<float> tmpOut(frameLen_, 0.0f);

        const size_t totalFrames = (numSamples + delaySamples + frameLen_ - 1) / frameLen_;
        std::vector<float> fullOutL(totalFrames * frameLen_, 0.0f);
        std::vector<float> fullOutR(totalFrames * frameLen_, 0.0f);

        for (size_t f = 0; f < totalFrames; ++f) {
            const size_t srcOff = f * frameLen_;
            for (size_t i = 0; i < frameLen_; ++i) {
                tmpIn[i] = (srcOff + i < numSamples) ? inL[srcOff + i] : 0.0f;
            }
            fnProcess_(stateL_, tmpIn.data(), tmpOut.data());
            std::copy_n(tmpOut.data(), frameLen_, fullOutL.data() + srcOff);

            for (size_t i = 0; i < frameLen_; ++i) {
                tmpIn[i] = (srcOff + i < numSamples) ? inR[srcOff + i] : 0.0f;
            }
            fnProcess_(stateR_, tmpIn.data(), tmpOut.data());
            std::copy_n(tmpOut.data(), frameLen_, fullOutR.data() + srcOff);
        }

        for (size_t i = 0; i < numSamples; ++i) {
            outL[i] = fullOutL[i + delaySamples];
            outR[i] = fullOutR[i + delaySamples];
        }
    }

private:
    using FnCreate = void* (*)(const char*, float, const char*);
    using FnFrameLen = size_t (*)(void*);
    using FnAttenLim = void (*)(void*, float);
    using FnPostBeta = void (*)(void*, float);
    using FnProcess = float (*)(void*, float*, float*);
    using FnFree = void (*)(void*);

#ifdef _WIN32
    HMODULE hDll_ = nullptr;
#endif
    FnCreate fnCreate_ = nullptr;
    FnFrameLen fnFrameLen_ = nullptr;
    FnAttenLim fnAttenLim_ = nullptr;
    FnPostBeta fnPostBeta_ = nullptr;
    FnProcess fnProcess_ = nullptr;
    FnFree fnFree_ = nullptr;
    void* stateL_ = nullptr;
    void* stateR_ = nullptr;
    size_t frameLen_ = 0;
    bool initialized_ = false;
    std::string modelTarUtf8_;
};

} // namespace OpenPhoton
