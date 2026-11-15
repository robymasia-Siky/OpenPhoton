#pragma once
#include "StemCache.hpp"
#include <string>

namespace OpenPhoton {

struct WavLoadResult {
    bool ok = false;
    double durationSeconds = 0.0;
    std::string error;
};

WavLoadResult loadWavToCache(const std::string& utf8Path, StemCache& cache, double targetSampleRate = 48000.0);
bool exportStemsToFolder(const std::string& folderUtf8, const StemCache& cache, double sampleRate = 48000.0);

} // namespace OpenPhoton
