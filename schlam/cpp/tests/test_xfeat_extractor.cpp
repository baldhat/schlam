#include <gtest/gtest.h>

#include "schlam/Frame.h"
#include "schlam/feature_detection/FeatureExtractorFactory.h"
#include "schlam/feature_detection/orb/ORBFeatureDetector.h"
#include "schlam/feature_detection/orb/OrbDescriptor.h"
#include "schlam/feature_detection/xfeat/XFeatDescriptor.h"
#include "schlam/feature_detection/xfeat/XFeatFeatureExtractor.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {
cv::Mat texture(int height, int width) {
    cv::Mat image(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            image.at<std::uint8_t>(y, x) = (13*y + 7*x + (3*x*y) % 251 + ((x/12 + y/10) % 2)*80) % 256;
    return image;
}

Frame frameFor(cv::Mat image) {
    return Frame(std::make_shared<ImageData>(utils::TTimestamp{}, image, Eigen::Matrix3d::Identity(), "camera"));
}
}

TEST(XFeatDescriptorTest, NormalizesAndComputesCosineDistance) {
    XFeatDescriptor::Values x{}, y{}, negative{};
    x[0] = 3; y[1] = 4; negative[0] = -5;
    const auto a = std::make_shared<XFeatDescriptor>(x);
    EXPECT_NEAR(a->values()[0], 1.0, 1e-7);
    EXPECT_NEAR(a->getDistance(std::make_shared<XFeatDescriptor>(x)), 0.0, 1e-7);
    EXPECT_NEAR(a->getDistance(std::make_shared<XFeatDescriptor>(y)), 1.0, 1e-7);
    EXPECT_NEAR(a->getDistance(std::make_shared<XFeatDescriptor>(negative)), 2.0, 1e-7);
    const auto orb = std::make_shared<OrbDescriptor>(TDescriptor{});
    EXPECT_THROW(a->getDistance(orb), std::invalid_argument);
    EXPECT_THROW(orb->getDistance(a), std::invalid_argument);
    EXPECT_THROW(a->getDistance(nullptr), std::invalid_argument);
}

TEST(XFeatDescriptorTest, RejectsZeroAndNonfiniteDescriptors) {
    XFeatDescriptor::Values values{};
    EXPECT_THROW({ XFeatDescriptor descriptor(values); }, std::invalid_argument);
    values[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW({ XFeatDescriptor descriptor(values); }, std::invalid_argument);
    values[0] = std::numeric_limits<float>::infinity();
    EXPECT_THROW({ XFeatDescriptor descriptor(values); }, std::invalid_argument);
}

TEST(XFeatExtractorTest, MatchesPinnedUpstreamSparseExtractionIncludingImageResizing) {
    schlam::config::Features settings;
    settings.type = schlam::config::FeatureType::XFEAT;
    settings.count = 40;
    settings.xfeat.inferenceThreads = 1;
    settings.xfeat.detectionThreshold = 0.01;
    auto extractor = createFeatureExtractor(settings);
    std::ifstream reference(std::filesystem::path(SCHLAM_XFEAT_ASSET_DIR) / "reference-sparse.txt");
    ASSERT_TRUE(reference.is_open());
    int cases;
    ASSERT_TRUE(static_cast<bool>(reference >> cases));
    for (int i = 0; i < cases; ++i) {
        int height, width, count;
        ASSERT_TRUE(static_cast<bool>(reference >> height >> width >> count));
        SCOPED_TRACE(std::to_string(height) + "x" + std::to_string(width));
        auto frame = frameFor(texture(height, width));
        extractor->extractFeatures(frame);
        auto features = frame.mKeypointTree->getFeatures();
        ASSERT_EQ(features.size(), count);
        ASSERT_GT(count, 0);
        std::sort(features.begin(), features.end(), [](const auto& a, const auto& b) { return a->getScore() > b->getScore(); });
        for (const auto& feature : features) {
            double x, y, score;
            ASSERT_TRUE(static_cast<bool>(reference >> x >> y >> score));
            EXPECT_NEAR(feature->getImgX(), x, 3e-5);
            EXPECT_NEAR(feature->getImgY(), y, 3e-5);
            EXPECT_NEAR(feature->getScore(), score, 2e-4);
            const auto descriptor = std::dynamic_pointer_cast<XFeatDescriptor>(feature->getDescriptor());
            ASSERT_TRUE(descriptor);
            double squaredNorm = 0;
            for (const float value : descriptor->values()) {
                float expected;
                ASSERT_TRUE(static_cast<bool>(reference >> expected));
                EXPECT_NEAR(value, expected, 1e-3);
                squaredNorm += static_cast<double>(value) * value;
            }
            EXPECT_NEAR(squaredNorm, 1.0, 1e-6);
            EXPECT_EQ(feature->getLevel(), 0);
            EXPECT_TRUE(feature->getIndex().has_value());
        }
    }
    float extra;
    EXPECT_FALSE(static_cast<bool>(reference >> extra));
}

TEST(XFeatExtractorTest, HonorsBudgetAndPreservesSpatialIndicesAndDescriptorOwnership) {
    schlam::config::Features settings;
    settings.type = schlam::config::FeatureType::XFEAT;
    settings.count = 7;
    settings.xfeat.inferenceThreads = 1;
    settings.xfeat.detectionThreshold = 0.01;
    auto extractor = createFeatureExtractor(settings);
    auto frame = frameFor(texture(97, 131));
    extractor->extractFeatures(frame);
    const auto features = frame.mKeypointTree->getFeatures();
    ASSERT_EQ(features.size(), 7);
    const auto descriptor = std::dynamic_pointer_cast<XFeatDescriptor>(features.front()->getDescriptor());
    ASSERT_TRUE(descriptor);
    const auto retained = descriptor->values();
    for (const auto& feature : features) {
        const auto index = feature->getIndex();
        ASSERT_TRUE(index);
        EXPECT_EQ(features[*index], feature);
        EXPECT_GE(feature->getImgX(), 0); EXPECT_LT(feature->getImgX(), 131);
        EXPECT_GE(feature->getImgY(), 0); EXPECT_LT(feature->getImgY(), 97);
        const auto queried = frame.mKeypointTree->getFeatures(feature->getImgX() - 0.01, feature->getImgX() + 0.01,
            feature->getImgY() - 0.01, feature->getImgY() + 0.01);
        EXPECT_NE(std::find(queried.begin(), queried.end(), feature), queried.end());
        EXPECT_EQ(feature->getIndex(), index);
    }
    auto next = frameFor(texture(128, 160));
    extractor->extractFeatures(next);
    EXPECT_EQ(descriptor->values(), retained);
}

TEST(XFeatExtractorTest, HandlesNoDetectionsAndRejectsInvalidImages) {
    schlam::config::Features settings;
    settings.type = schlam::config::FeatureType::XFEAT;
    settings.xfeat.inferenceThreads = 1;
    settings.xfeat.detectionThreshold = 1.0;
    auto extractor = createFeatureExtractor(settings);
    auto frame = frameFor(texture(96, 128));
    extractor->extractFeatures(frame);
    EXPECT_TRUE(frame.mKeypointTree->getFeatures().empty());
    for (const auto& pixels : {cv::Mat(), cv::Mat(32, 32, CV_8UC3)}) {
        auto invalid = frameFor(pixels);
        EXPECT_THROW(extractor->extractFeatures(invalid), std::invalid_argument);
    }
    settings.count = 0;
    settings.xfeat.detectionThreshold = 0.01;
    auto disabled = createFeatureExtractor(settings);
    disabled->extractFeatures(frame);
    EXPECT_TRUE(frame.mKeypointTree->getFeatures().empty());
}

TEST(XFeatExtractorTest, PaddingKeepsPixelCoordinatesAndSupportsSmallAndNonDivisibleImages) {
    schlam::config::Features settings;
    settings.type = schlam::config::FeatureType::XFEAT;
    settings.xfeat.inferenceThreads = 1;
    settings.xfeat.imageSizing = schlam::config::XFeatImageSizing::PAD;
    settings.xfeat.detectionThreshold = 0.01;
    auto extractor = createFeatureExtractor(settings);
    for (const auto size : {cv::Size(131, 97), cv::Size(13, 17), cv::Size(1, 1)}) {
        auto frame = frameFor(texture(size.height, size.width));
        EXPECT_NO_THROW(extractor->extractFeatures(frame));
        ASSERT_TRUE(frame.mKeypointTree);
        for (const auto& feature : frame.mKeypointTree->getFeatures()) {
            EXPECT_GE(feature->getImgX(), 0); EXPECT_LT(feature->getImgX(), size.width);
            EXPECT_GE(feature->getImgY(), 0); EXPECT_LT(feature->getImgY(), size.height);
            EXPECT_DOUBLE_EQ(feature->getImgX(), std::round(feature->getImgX()));
            EXPECT_DOUBLE_EQ(feature->getImgY(), std::round(feature->getImgY()));
        }
    }
}

TEST(FeatureExtractorFactoryTest, DefaultsToOrbWithoutLoadingTheXFeatModel) {
    schlam::config::Features settings;
    settings.xfeat.modelPath = "/missing/xfeat.onnx";
    EXPECT_TRUE(std::dynamic_pointer_cast<ORBFeatureDetector>(createFeatureExtractor(settings)));
    settings.type = schlam::config::FeatureType::XFEAT;
    EXPECT_THROW(createFeatureExtractor(settings), std::invalid_argument);
}
