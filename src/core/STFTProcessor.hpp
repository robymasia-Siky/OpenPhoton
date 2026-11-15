#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

namespace OpenPhoton {

class STFTProcessor {
public:
    STFTProcessor(size_t fftSize = 2048, size_t hopSize = 512)
        : fftSize_(fftSize), hopSize_(hopSize), window_(fftSize, 0.0f), bitRev_(fftSize, 0)
    {
        for (size_t i = 0; i < fftSize_; ++i) {
            window_[i] = 0.5f * (1.0f - std::cos(2.0f * std::numbers::pi_v<float> *
                         static_cast<float>(i) / static_cast<float>(fftSize_)));
        }
        size_t log2n = 0;
        while ((size_t(1) << log2n) < fftSize_) ++log2n;
        for (size_t i = 0; i < fftSize_; ++i) {
            size_t r = 0, x = i;
            for (size_t b = 0; b < log2n; ++b) {
                r = (r << 1) | (x & 1);
                x >>= 1;
            }
            bitRev_[i] = r;
        }
        twiddleFwd_.resize(fftSize_ / 2);
        twiddleInv_.resize(fftSize_ / 2);
        for (size_t k = 0; k < fftSize_ / 2; ++k) {
            const float ang = -2.0f * std::numbers::pi_v<float> * static_cast<float>(k) / static_cast<float>(fftSize_);
            twiddleFwd_[k] = { std::cos(ang), std::sin(ang) };
            twiddleInv_[k] = { std::cos(-ang), std::sin(-ang) };
        }
    }

    size_t fftSize() const noexcept { return fftSize_; }
    size_t hopSize() const noexcept { return hopSize_; }
    size_t numBins() const noexcept { return fftSize_ / 2 + 1; }

    void forward(const float* input, size_t numSamples,
                 std::vector<std::complex<float>>& outSpec, size_t& outFrames) const
    {
        if (numSamples < fftSize_) { outFrames = 0; outSpec.clear(); return; }
        outFrames = (numSamples - fftSize_) / hopSize_ + 1;
        const size_t bins = numBins();
        outSpec.resize(outFrames * bins);

        std::vector<std::complex<float>> buf(fftSize_);
        for (size_t f = 0; f < outFrames; ++f) {
            const size_t offset = f * hopSize_;
            for (size_t i = 0; i < fftSize_; ++i) {
                buf[bitRev_[i]] = { input[offset + i] * window_[i], 0.0f };
            }
            fftPrecomputed(buf, false);
            std::copy_n(buf.data(), bins, outSpec.data() + f * bins);
        }
    }

    void inverse(const std::vector<std::complex<float>>& spec, size_t numFrames,
                 size_t targetSamples, std::vector<float>& outAudio) const
    {
        outAudio.assign(targetSamples, 0.0f);
        std::vector<float> winNorm(targetSamples, 0.0f);
        std::vector<std::complex<float>> tmp(fftSize_), buf(fftSize_);
        const size_t bins = numBins();

        for (size_t f = 0; f < numFrames; ++f) {
            const auto* src = spec.data() + f * bins;
            for (size_t b = 0; b < bins; ++b) tmp[b] = src[b];
            for (size_t b = 1; b < bins - 1; ++b) tmp[fftSize_ - b] = std::conj(tmp[b]);
            for (size_t i = 0; i < fftSize_; ++i) buf[bitRev_[i]] = tmp[i];

            fftPrecomputed(buf, true);

            const size_t offset = f * hopSize_;
            const float invN = 1.0f / static_cast<float>(fftSize_);
            for (size_t i = 0; i < fftSize_ && (offset + i) < targetSamples; ++i) {
                outAudio[offset + i] += buf[i].real() * invN * window_[i];
                winNorm[offset + i]  += window_[i] * window_[i];
            }
        }
        for (size_t i = 0; i < targetSamples; ++i) {
            if (winNorm[i] > 1e-6f) outAudio[i] /= winNorm[i];
        }
    }

private:
    void fftPrecomputed(std::vector<std::complex<float>>& a, bool invert) const {
        const auto& tw = invert ? twiddleInv_ : twiddleFwd_;
        for (size_t len = 2; len <= fftSize_; len <<= 1) {
            const size_t half = len >> 1;
            const size_t step = fftSize_ / len;
            for (size_t i = 0; i < fftSize_; i += len) {
                for (size_t j = 0; j < half; ++j) {
                    const auto u = a[i + j];
                    const auto v = a[i + j + half] * tw[j * step];
                    a[i + j] = u + v;
                    a[i + j + half] = u - v;
                }
            }
        }
    }

    size_t fftSize_;
    size_t hopSize_;
    std::vector<float> window_;
    std::vector<size_t> bitRev_;
    std::vector<std::complex<float>> twiddleFwd_;
    std::vector<std::complex<float>> twiddleInv_;
};

} // namespace OpenPhoton
