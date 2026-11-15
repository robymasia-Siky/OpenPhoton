#pragma once
#include "StemCache.hpp"
#include <cstdint>

namespace OpenPhoton {

inline constexpr size_t kTelemetryBins = 240;

#pragma pack(push, 1)
struct TelemetrySnapshot {
    int32_t activeChunks = 1;
    int32_t playheadChunk = 0;
    float playheadNormalized = 0.0f;
    float totalDurationSeconds = 4.0f;
    float lastInferenceMs = 0.0f;
    float downloadProgress = 0.0f;
    uint8_t currentMode = 0;
    uint8_t internalPlaying = 0;
    uint8_t isDownloading = 0;
    uint8_t modelInstalled = 0;
    uint8_t usingGpu = 0;
    uint8_t usingOnnx = 0;
    uint8_t usingDeepFilter = 0;
    uint8_t modelType = 0;
    char modelName[64]{};
    uint8_t chunkStates[kMaxChunks]{};
    uint8_t peaks[kNumStems][kTelemetryBins]{};
};
#pragma pack(pop)

} // namespace OpenPhoton
