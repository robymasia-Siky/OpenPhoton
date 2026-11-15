#pragma once

#include "DebugLog.hpp"
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

namespace OpenPhoton {

class DeepFilterBridge {
public:
    ~DeepFilterBridge() { unload(); }

    bool init(const std::filesystem::path& pluginDir) {
        if (initialized_) return true;
        if (initAttempted_) return false;
        initAttempted_ = true;

#ifdef _WIN32
        const auto dllPath = pluginDir / L"df.dll";
        const auto tarPath = pluginDir / L"DeepFilterNet3_onnx.tar.gz";
        if (!std::filesystem::exists(dllPath) || !std::filesystem::exists(tarPath)) {
            logTrace("DeepFilterBridge: df.dll or model tar.gz not found.");
            return false;
        }

        hDll_ = LoadLibraryW(dllPath.c_str());
        if (!hDll_) {
            logTrace("DeepFilterBridge: LoadLibraryW(df.dll) failed.");
            return false;
        }

        fnCreate_ = reinterpret_cast<FnCreate>(GetProcAddress(hDll_, "df_create"));
        fnFrameLen_ = reinterpret_cast<FnFrameLen>(GetProcAddress(hDll_, "df_get_frame_length"));
        fnAttenLim_ = reinterpret_cast<FnAttenLim>(GetProcAddress(hDll_, "df_set_atten_lim"));
        fnPostBeta_ = reinterpret_cast<FnPostBeta>(GetProcAddress(hDll_, "df_set_post_filter_beta"));
        fnProcess_ = reinterpret_cast<FnProcess>(GetProcAddress(hDll_, "df_process_frame"));
        fnFree_ = reinterpret_cast<FnFree>(GetProcAddress(hDll_, "df_free"));

        if (!fnCreate_ || !fnFrameLen_ || !fnProcess_ || !fnFree_) {
            logTrace("DeepFilterBridge: missing exported symbols.");
            unload();
            return false;
        }

        const auto u8 = tarPath.u8string();
        modelTarUtf8_.assign(reinterpret_cast<const char*>(u8.data()), u8.size());

        state_ = sehCreate(fnCreate_, modelTarUtf8_.c_str(), 100.0f);
        if (!state_) {
            logTrace("DeepFilterBridge: df_create failed or trapped safely.");
            unload();
            return false;
        }
        frameLen_ = sehFrameLen(fnFrameLen_, state_);
        initialized_ = (frameLen_ > 0);
        logTrace(initialized_ ? "DeepFilterBridge: initialized OK." : "DeepFilterBridge: invalid frameLen.");
        return initialized_;
#else
        (void)pluginDir;
        return false;
#endif
    }

    void unload() {
        if (fnFree_ && state_) {
            sehFree(fnFree_, state_);
        }
        state_ = nullptr;
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
        if (!initialized_ || frameLen_ == 0 || !state_) return;

        const float attenDb = 6.0f + std::clamp(strength, 0.0f, 1.0f) * 94.0f;
        const float beta = (strength > 0.75f) ? (strength - 0.75f) * 0.08f : 0.0f;
        if (fnAttenLim_) fnAttenLim_(state_, attenDb);
        if (fnPostBeta_) fnPostBeta_(state_, beta);

        const size_t delaySamples = frameLen_ * 2;
        std::vector<float> tmpIn(frameLen_, 0.0f);
        std::vector<float> tmpOut(frameLen_, 0.0f);

        const size_t totalFrames = (numSamples + delaySamples + frameLen_ - 1) / frameLen_;
        std::vector<float> monoOut(totalFrames * frameLen_, 0.0f);

        for (int z = 0; z < 4; ++z) {
            sehProcess(fnProcess_, state_, tmpIn.data(), tmpOut.data());
        }

        for (size_t f = 0; f < totalFrames; ++f) {
            const size_t srcOff = f * frameLen_;
            for (size_t i = 0; i < frameLen_; ++i) {
                tmpIn[i] = (srcOff + i < numSamples) ? 0.5f * (inL[srcOff + i] + inR[srcOff + i]) : 0.0f;
            }
            sehProcess(fnProcess_, state_, tmpIn.data(), tmpOut.data());
            std::copy_n(tmpOut.data(), frameLen_, monoOut.data() + srcOff);
        }

        for (size_t i = 0; i < numSamples; ++i) {
            const float side = 0.5f * (inL[i] - inR[i]) * (1.0f - strength);
            const float mid = monoOut[i + delaySamples];
            outL[i] = mid + side;
            outR[i] = mid - side;
        }
    }

private:
    using FnCreate = void* (*)(const char*, float, const char*);
    using FnFrameLen = size_t (*)(void*);
    using FnAttenLim = void (*)(void*, float);
    using FnPostBeta = void (*)(void*, float);
    using FnProcess = float (*)(void*, float*, float*);
    using FnFree = void (*)(void*);

    static void* sehCreate(FnCreate fn, const char* path, float atten);
    static size_t sehFrameLen(FnFrameLen fn, void* st);
    static float sehProcess(FnProcess fn, void* st, float* in, float* out);
    static void sehFree(FnFree fn, void* st);

#ifdef _WIN32
    HMODULE hDll_ = nullptr;
#endif
    FnCreate fnCreate_ = nullptr;
    FnFrameLen fnFrameLen_ = nullptr;
    FnAttenLim fnAttenLim_ = nullptr;
    FnPostBeta fnPostBeta_ = nullptr;
    FnProcess fnProcess_ = nullptr;
    FnFree fnFree_ = nullptr;
    void* state_ = nullptr;
    size_t frameLen_ = 0;
    bool initialized_ = false;
    bool initAttempted_ = false;
    std::string modelTarUtf8_;
};

} // namespace OpenPhoton
