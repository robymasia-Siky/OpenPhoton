#pragma once

#include <memory>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

namespace OpenPhoton {

enum class ExecutionBackend {
    Auto,
    CUDA,
    CPU
};

struct InferenceStats {
    double inferenceMs = 0.0;
    std::string activeProvider;
};

class OnnxEngine {
public:
    explicit OnnxEngine(const std::wstring& modelPath,
                        ExecutionBackend backend = ExecutionBackend::Auto,
                        int gpuDeviceId = 0);
    ~OnnxEngine() = default;

    OnnxEngine(const OnnxEngine&) = delete;
    OnnxEngine& operator=(const OnnxEngine&) = delete;

    InferenceStats runAudioChunk(const float* inputData,
                                 const std::vector<int64_t>& inputShape,
                                 std::vector<float>& outputData,
                                 std::vector<int64_t>& outputShape);

    const std::string& activeProvider() const noexcept { return activeProvider_; }

private:
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memoryInfo_;

    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> inputNamePtrs_;
    std::vector<const char*> outputNamePtrs_;
    std::string activeProvider_ = "CPU";
};

} // namespace OpenPhoton
