#include "XFeatModel.h"

#include <onnxruntime_cxx_api.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::array<const char*, 3> outputNames{"features", "keypoint_logits", "reliability"};
constexpr std::array<int, 3> outputChannels{64, 65, 1};

void checkTensor(const Ort::TypeInfo& type, int channels) {
    const auto tensor = type.GetTensorTypeAndShapeInfo();
    const auto shape = tensor.GetShape();
    if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape.size() != 4 || shape[0] != 1 || shape[1] != channels) {
        throw std::invalid_argument("XFeat model must use float32 NCHW tensors with batch size 1");
    }
}
}

struct XFeatModel::Impl {
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "Schlam-XFeat"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};

    Impl(const std::filesystem::path& modelPath, int inferenceThreads) {
        if (inferenceThreads < 1) throw std::invalid_argument("XFeat inferenceThreads must be positive");
        if (!std::filesystem::is_regular_file(modelPath)) {
            throw std::invalid_argument("XFeat model file does not exist: " + modelPath.string());
        }
        options.SetIntraOpNumThreads(inferenceThreads);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(environment, modelPath.c_str(), options);
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 3) {
            throw std::invalid_argument("XFeat backbone must have one input and three outputs");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        const auto modelKind = session.GetModelMetadata().LookupCustomMetadataMapAllocated("schlam.model", allocator);
        if (!modelKind || std::string(modelKind.get()) != "xfeat-backbone") {
            throw std::invalid_argument("Expected the XFeat backbone produced by recipes/export-xfeat.py");
        }
        const auto inputName = session.GetInputNameAllocated(0, allocator);
        if (std::string(inputName.get()) != "image") {
            throw std::invalid_argument("XFeat backbone input must be named 'image'");
        }
        checkTensor(session.GetInputTypeInfo(0), 1);
        for (std::size_t i = 0; i < outputNames.size(); ++i) {
            const auto outputName = session.GetOutputNameAllocated(i, allocator);
            if (std::string(outputName.get()) != outputNames[i]) {
                throw std::invalid_argument("Unexpected XFeat backbone output names or order");
            }
            checkTensor(session.GetOutputTypeInfo(i), outputChannels[i]);
        }
    }
};

XFeatModel::XFeatModel(const std::filesystem::path& modelPath, int inferenceThreads) {
    try {
        impl_ = std::make_unique<Impl>(modelPath, inferenceThreads);
    } catch (const Ort::Exception& error) {
        throw std::runtime_error("Cannot load XFeat model '" + modelPath.string() + "': " + error.what());
    }
}

XFeatModel::~XFeatModel() = default;

XFeatModel::Output XFeatModel::run(const cv::Mat& image) const {
    if (image.empty() || image.dims != 2 || image.type() != CV_8UC1 ||
        image.rows < 32 || image.cols < 32 || image.rows % 32 || image.cols % 32) {
        throw std::invalid_argument("XFeat input must be CV_8UC1 with H/W >= 32 and divisible by 32");
    }
    cv::Mat pixels;
    image.convertTo(pixels, CV_32FC1, 1.0 / 255.0);
    const std::array<std::int64_t, 4> inputShape{1, 1, image.rows, image.cols};
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory, pixels.ptr<float>(), pixels.total(),
        inputShape.data(), inputShape.size());
    const char* inputName = "image";
    auto tensors = impl_->session.Run(Ort::RunOptions{nullptr}, &inputName, &input, 1,
        outputNames.data(), outputNames.size());
    std::array<cv::Mat, 3> outputs;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const std::array<int, 4> shape{1, outputChannels[i], image.rows / 8, image.cols / 8};
        const auto actualShape = tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        if (actualShape != std::vector<std::int64_t>(shape.begin(), shape.end())) {
            throw std::runtime_error("XFeat backbone returned an unexpected tensor shape");
        }
        outputs[i] = cv::Mat(static_cast<int>(shape.size()), shape.data(), CV_32FC1);
        std::memcpy(outputs[i].ptr<float>(), tensors[i].GetTensorData<float>(), outputs[i].total() * sizeof(float));
    }
    return {outputs[0], outputs[1], outputs[2]};
}
