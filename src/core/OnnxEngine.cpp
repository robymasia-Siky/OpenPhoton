#include "OnnxEngine.hpp"
#include "DebugLog.hpp"
#include <chrono>
#include <stdexcept>
#ifdef _WIN32
#include <dml_provider_factory.h>
#endif

namespace OpenPhoton {

OnnxEngine::OnnxEngine(const std::wstring& modelPath,
                       ExecutionBackend backend,
                       int gpuDeviceId)
    : env_(ORT_LOGGING_LEVEL_WARNING, "OpenPhotonORT"),
      memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool dmlAttached = false;
#ifdef _WIN32
    if (backend == ExecutionBackend::Auto || backend == ExecutionBackend::DirectML) {
        try {
            sessionOptions_.DisableMemPattern();
            sessionOptions_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions_, gpuDeviceId));
            dmlAttached = true;
            activeProvider_ = "DirectML";
            logTrace("OnnxEngine: DirectML GPU attached.");
        } catch (const Ort::Exception& e) {
            logTrace(std::string("OnnxEngine: DirectML fallback CPU: ") + e.what());
            if (backend == ExecutionBackend::DirectML) throw;
            activeProvider_ = "CPU";
        }
    }
#endif

    if (!dmlAttached) {
        sessionOptions_.SetIntraOpNumThreads(0);
    }

    session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t nInputs = session_->GetInputCount();
    const size_t nOutputs = session_->GetOutputCount();

    inputNames_.reserve(nInputs);
    inputNamePtrs_.reserve(nInputs);
    for (size_t i = 0; i < nInputs; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator);
        inputNames_.emplace_back(name.get());
    }
    for (const auto& s : inputNames_) inputNamePtrs_.push_back(s.c_str());

    outputNames_.reserve(nOutputs);
    outputNamePtrs_.reserve(nOutputs);
    for (size_t i = 0; i < nOutputs; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        outputNames_.emplace_back(name.get());
    }
    for (const auto& s : outputNames_) outputNamePtrs_.push_back(s.c_str());
}

InferenceStats OnnxEngine::runAudioChunk(const float* inputData,
                                         const std::vector<int64_t>& inputShape,
                                         std::vector<float>& outputData,
                                         std::vector<int64_t>& outputShape)
{
    int64_t numInputElements = 1;
    for (auto dim : inputShape) numInputElements *= dim;

    auto inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo_,
        const_cast<float*>(inputData),
        static_cast<size_t>(numInputElements),
        inputShape.data(),
        inputShape.size()
    );

    const auto t0 = std::chrono::steady_clock::now();
    auto outputTensors = session_->Run(
        Ort::RunOptions{nullptr},
        inputNamePtrs_.data(),
        &inputTensor,
        1,
        outputNamePtrs_.data(),
        outputNamePtrs_.size()
    );
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (outputTensors.empty() || !outputTensors[0].IsTensor()) {
        throw std::runtime_error("OpenPhoton: invalid ONNX output tensor.");
    }

    auto typeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    outputShape = typeInfo.GetShape();

    const size_t totalOut = typeInfo.GetElementCount();
    const float* rawOut = outputTensors[0].GetTensorData<float>();
    outputData.assign(rawOut, rawOut + totalOut);

    return InferenceStats{ ms, activeProvider_ };
}

} // namespace OpenPhoton
