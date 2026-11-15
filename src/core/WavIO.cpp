#include "WavIO.hpp"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

namespace OpenPhoton {

namespace {
std::filesystem::path pathFromUtf8(const std::string& utf8) {
    return std::filesystem::path(reinterpret_cast<const char8_t*>(utf8.c_str()));
}
}

WavLoadResult loadWavToCache(const std::string& utf8Path, StemCache& cache, double targetSampleRate) {
    WavLoadResult res;
    drwav wav{};
#ifdef _WIN32
    auto wpath = pathFromUtf8(utf8Path).wstring();
    if (!drwav_init_file_w(&wav, wpath.c_str(), nullptr)) {
#else
    if (!drwav_init_file(&wav, utf8Path.c_str(), nullptr)) {
#endif
        res.error = "Cannot open WAV file.";
        return res;
    }

    const uint64_t totalFrames = wav.totalPCMFrameCount;
    const uint32_t channels = wav.channels;
    const uint32_t srcRate = wav.sampleRate;
    if (totalFrames == 0 || channels == 0 || srcRate == 0) {
        drwav_uninit(&wav);
        res.error = "Invalid WAV file.";
        return res;
    }

    const uint64_t maxFramesToRead = std::min<uint64_t>(
        totalFrames,
        static_cast<uint64_t>(srcRate) * 240ULL
    );

    std::vector<float> interleaved(static_cast<size_t>(maxFramesToRead) * channels);
    const uint64_t readFrames = drwav_read_pcm_frames_f32(&wav, maxFramesToRead, interleaved.data());
    drwav_uninit(&wav);

    if (readFrames == 0) {
        res.error = "No frames read.";
        return res;
    }

    cache.clearAll();
    const double ratio = static_cast<double>(srcRate) / targetSampleRate;
    const size_t outTotalFrames = std::min<size_t>(
        kMaxChunks * kChunkSamples,
        static_cast<size_t>(static_cast<double>(readFrames) / ratio)
    );

    for (size_t i = 0; i < outTotalFrames; ++i) {
        const double srcPos = static_cast<double>(i) * ratio;
        const size_t idx0 = std::min<size_t>(static_cast<size_t>(srcPos), static_cast<size_t>(readFrames - 1));
        const size_t idx1 = std::min<size_t>(idx0 + 1, static_cast<size_t>(readFrames - 1));
        const float frac = static_cast<float>(srcPos - static_cast<double>(idx0));

        const float l0 = interleaved[idx0 * channels];
        const float l1 = interleaved[idx1 * channels];
        const float left = l0 + (l1 - l0) * frac;

        const uint32_t rCh = (channels > 1) ? 1 : 0;
        const float r0 = interleaved[idx0 * channels + rCh];
        const float r1 = interleaved[idx1 * channels + rCh];
        const float right = r0 + (r1 - r0) * frac;

        cache.writeCaptureSample(static_cast<int64_t>(i), left, right, true);
    }

    res.ok = true;
    res.durationSeconds = static_cast<double>(outTotalFrames) / targetSampleRate;
    return res;
}

bool exportStemsToFolder(const std::string& folderUtf8, const StemCache& cache, double sampleRate) {
    size_t readyChunks = 0;
    for (size_t i = 0; i < kMaxChunks; ++i) {
        if (cache.slot(i).state.load(std::memory_order_acquire) == ChunkState::Ready) {
            readyChunks = i + 1;
        }
    }
    if (readyChunks == 0) return false;

    const std::filesystem::path dir = pathFromUtf8(folderUtf8);
    const char* fileNames[kNumStems] = {
        "OpenPhoton_Vocals.wav",
        "OpenPhoton_Drums.wav",
        "OpenPhoton_Bass.wav",
        "OpenPhoton_Guitar.wav",
        "OpenPhoton_Piano.wav",
        "OpenPhoton_Other.wav"
    };

    drwav_data_format format{};
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 2;
    format.sampleRate = static_cast<drwav_uint32>(sampleRate);
    format.bitsPerSample = 32;

    std::vector<float> interleaved(kChunkSamples * 2);

    for (size_t s = 0; s < kNumStems; ++s) {
        drwav wav{};
        const auto outPath = dir / fileNames[s];
#ifdef _WIN32
        if (!drwav_init_file_write_w(&wav, outPath.wstring().c_str(), &format, nullptr)) return false;
#else
        if (!drwav_init_file_write(&wav, outPath.string().c_str(), &format, nullptr)) return false;
#endif
        const size_t base = s * kNumChannels * kChunkSamples;
        for (size_t c = 0; c < readyChunks; ++c) {
            const auto& slot = cache.slot(c);
            if (slot.state.load(std::memory_order_acquire) != ChunkState::Ready) {
                std::fill(interleaved.begin(), interleaved.end(), 0.0f);
            } else {
                for (size_t i = 0; i < kChunkSamples; ++i) {
                    interleaved[i * 2]     = slot.stemAudio[base + i];
                    interleaved[i * 2 + 1] = slot.stemAudio[base + kChunkSamples + i];
                }
            }
            drwav_write_pcm_frames(&wav, kChunkSamples, interleaved.data());
        }
        drwav_uninit(&wav);
    }
    return true;
}

} // namespace OpenPhoton

