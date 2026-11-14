#include <gtest/gtest.h>

#include "schlam/feature_detection/xfeat/XFeatModel.h"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <opencv2/core.hpp>

namespace {
const std::filesystem::path assets{SCHLAM_XFEAT_ASSET_DIR};

cv::Mat referenceImage() {
    cv::Mat image(32, 32, CV_8UC1);
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
            image.at<std::uint8_t>(y, x) = (13 * y + 7 * x) % 256;
    return image;
}

TEST(XFeatModelTest, LoadsPretrainedModelAndMatchesUpstreamPyTorchTensors) {
    XFeatModel model(assets / "xfeat.onnx");
    const auto output = model.run(referenceImage());
    std::ifstream reference(assets / "reference-32x32.txt");
    ASSERT_TRUE(reference.is_open());
    for (const auto& tensor : {output.features, output.keypointLogits, output.reliability}) {
        ASSERT_EQ(tensor.type(), CV_32FC1);
        for (std::size_t i = 0; i < tensor.total(); ++i) {
            float expected;
            ASSERT_TRUE(static_cast<bool>(reference >> expected));
            EXPECT_NEAR(tensor.ptr<float>()[i], expected, 2e-4 + 2e-4 * std::abs(expected)) << "element " << i;
        }
    }
    float extra;
    EXPECT_FALSE(static_cast<bool>(reference >> extra));
}

TEST(XFeatModelTest, ReusesSessionForDifferentResolutionsAndReturnsOwnedTensors) {
    XFeatModel model(assets / "xfeat.onnx");
    const auto first = model.run(referenceImage());
    const auto retained = first.features.clone();
    const auto second = model.run(cv::Mat(64, 96, CV_8UC1, cv::Scalar(127)));
    for (const auto& [tensor, channels] : {
             std::pair(second.features, 64), std::pair(second.keypointLogits, 65), std::pair(second.reliability, 1)}) {
        ASSERT_EQ(tensor.dims, 4);
        EXPECT_EQ(tensor.size[0], 1);
        EXPECT_EQ(tensor.size[1], channels);
        EXPECT_EQ(tensor.size[2], 8);
        EXPECT_EQ(tensor.size[3], 12);
        for (std::size_t i = 0; i < tensor.total(); ++i) EXPECT_TRUE(std::isfinite(tensor.ptr<float>()[i]));
    }
    EXPECT_DOUBLE_EQ(cv::norm(first.features, retained, cv::NORM_INF), 0.0);
    for (std::size_t i = 0; i < second.reliability.total(); ++i) {
        EXPECT_GE(second.reliability.ptr<float>()[i], 0.0f);
        EXPECT_LE(second.reliability.ptr<float>()[i], 1.0f);
    }
}

TEST(XFeatModelTest, AcceptsNonContiguousGrayscaleImage) {
    XFeatModel model(assets / "xfeat.onnx");
    cv::Mat backing(64, 96, CV_8UC1, cv::Scalar(0));
    auto region = backing(cv::Rect(3, 7, 32, 32));
    referenceImage().copyTo(region);
    ASSERT_FALSE(region.isContinuous());
    const auto actual = model.run(region);
    const auto expected = model.run(referenceImage());
    EXPECT_DOUBLE_EQ(cv::norm(actual.features, expected.features, cv::NORM_INF), 0.0);
}

TEST(XFeatModelTest, RejectsMissingModelAndInvalidThreadCount) {
    EXPECT_THROW(XFeatModel(assets / "missing.onnx"), std::invalid_argument);
    EXPECT_THROW(XFeatModel(assets / "xfeat.onnx", 0), std::invalid_argument);
}

TEST(XFeatModelTest, RejectsInvalidImagesBeforeInference) {
    XFeatModel model(assets / "xfeat.onnx");
    for (const auto& image : {
             cv::Mat(), cv::Mat(16, 32, CV_8UC1), cv::Mat(33, 32, CV_8UC1),
             cv::Mat(32, 33, CV_8UC1), cv::Mat(32, 32, CV_8UC3), cv::Mat(32, 32, CV_32FC1)}) {
        EXPECT_THROW(static_cast<void>(model.run(image)), std::invalid_argument);
    }
}
}
