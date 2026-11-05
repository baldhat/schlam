#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "data/DatasetInterface.hpp"
#include "plotting/DummyPlotter.h"
#include "random.hpp"
#include "schlam/MotionModel.h"
#include "schlam/ORBFeatureDetector.h"
#include "schlam/Optimizer.h"
#include "schlam/SchlamSystem.h"
#include "tft/transformer.hpp"

namespace {
class MemoryDataset : public DatasetInterface {
public:
    std::vector<cv::Mat> images;
    std::size_t index = 0;
    Eigen::Vector3d origin{3, 4, 5};
    Eigen::Matrix3d orientation = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d intrinsics = (Eigen::Matrix3d() <<
        200, 0, 128, 0, 200, 128, 0, 0, 1).finished();

    static utils::TTimestamp timestamp(std::size_t index) {
        return utils::TTimestamp(std::chrono::seconds(100) + std::chrono::milliseconds(50 * index));
    }
    double getImageAspectRatio() override { return 1.0; }
    bool hasIMU() override { return false; }
    bool hasGT() override { return true; }
    bool empty() override { return index == images.size(); }
    Eigen::Matrix3d getCameraIntrinsics() override { return intrinsics; }
    float percentageRead() override { return images.empty() ? 100 : 100.0 * index / images.size(); }
    std::shared_ptr<ImageData> getNextImageData() override {
        const auto current = index++;
        return std::make_shared<ImageData>(timestamp(current), images[current], intrinsics, "gt/cam0");
    }
    std::shared_ptr<IMUData> getNextIMUData() override { return nullptr; }
    std::shared_ptr<GTData> getNextGTData() override {
        return std::make_shared<GTData>(timestamp(index - 1),
            origin + orientation * Eigen::Vector3d(0.1 * (index - 1), 0, 0),
            orientation, Eigen::Vector3d::Zero(), "world");
    }
};

class RecordingPlotter : public DummyPlotter {
public:
    double processingSeconds = 0.0;
    std::size_t frames = 0;
    const Map* map = nullptr;
    void setMap(const Map* value) override { map = value; }
    void recordFrameTime(std::chrono::steady_clock::duration elapsed) override {
        processingSeconds += std::chrono::duration<double>(elapsed).count();
        ++frames;
    }
};

class SchlamSystemResultTest : public ::testing::Test {
protected:
    std::shared_ptr<MemoryDataset> dataset = std::make_shared<MemoryDataset>();
    std::shared_ptr<RecordingPlotter> plotter = std::make_shared<RecordingPlotter>();
    std::shared_ptr<tft::Transformer> transformer = std::make_shared<tft::Transformer>();
    schlam::config::Config settings;
    MotionModel motion;
    MapSnapshot finalMap;

    void addTranslatedTextureFrames(int count) {
        cv::Mat texture(256, 256, CV_8UC1);
        cv::RNG random(1234);
        random.fill(texture, cv::RNG::UNIFORM, 0, 256);
        for (int frame = 0; frame < count; ++frame) {
            cv::Mat image;
            cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, -10 * frame, 0, 1, 0);
            cv::warpAffine(texture, image, shift, texture.size());
            dataset->images.push_back(image);
        }
    }

    SchlamResult run() {
        transformer->registerTransform(tft::RigidTransform3D::identity("gt/cam0", "gt/body"));
        utils::randomEngine().seed(settings.reconstruction.randomSeed);
        SchlamSystem system(dataset, transformer, plotter,
            std::make_shared<ORBFeatureDetector>(settings.features, plotter),
            std::make_shared<Optimizer>(settings.optimizer), motion, settings);
        auto result = system.run();
        if (plotter->map) finalMap = plotter->map->getSnapshot();
        return result;
    }
};

TEST_F(SchlamSystemResultTest, EmptyDatasetReturnsEmptyResultAndZeroFrameRate) {
    const auto result = run();
    EXPECT_TRUE(result.mapPoints.empty());
    EXPECT_TRUE(result.predictedPath.empty());
    EXPECT_DOUBLE_EQ(result.averageFrameRate, 0.0);
}

TEST_F(SchlamSystemResultTest, UninitializedFramesHaveTimingButNoInventedPath) {
    dataset->images.assign(2, cv::Mat(256, 256, CV_8UC1, cv::Scalar(0)));
    const auto result = run();
    EXPECT_TRUE(result.mapPoints.empty());
    EXPECT_TRUE(result.predictedPath.empty());
    ASSERT_EQ(plotter->frames, 2);
    ASSERT_GT(plotter->processingSeconds, 0.0);
    EXPECT_NEAR(result.averageFrameRate, plotter->frames / plotter->processingSeconds, 1e-6);
}

TEST_F(SchlamSystemResultTest, InitializedRunReturnsWorldMapAndTimestampedPath) {
    addTranslatedTextureFrames(3);
    dataset->orientation = Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    settings.mapping.keyframeMaximumInliers = 10000;
    settings.mapping.keyframeMinimumFrameGapWhenBusy = 0;
    const auto result = run();
    ASSERT_FALSE(result.mapPoints.empty());
    ASSERT_EQ(finalMap.keyFrames.size(), 3);
    ASSERT_EQ(result.mapPoints.size(), finalMap.points.size());
    for (std::size_t point = 0; point < result.mapPoints.size(); ++point) {
        EXPECT_TRUE(result.mapPoints[point].isApprox(
            dataset->orientation * finalMap.points[point] + dataset->origin));
    }
    ASSERT_EQ(result.predictedPath.size(), 3);
    EXPECT_TRUE(result.predictedPath.front().position.isApprox(dataset->origin));
    for (std::size_t frame = 0; frame < result.predictedPath.size(); ++frame) {
        EXPECT_EQ(result.predictedPath[frame].timestamp, MemoryDataset::timestamp(frame));
        EXPECT_TRUE(result.predictedPath[frame].rotation.allFinite());
        EXPECT_TRUE(result.predictedPath[frame].rotation.isApprox(dataset->orientation, 0.01));
        EXPECT_TRUE(result.predictedPath[frame].position.isApprox(
            dataset->origin + dataset->orientation * Eigen::Vector3d(0.1 * frame, 0, 0), 0.01));
    }
    for (const auto& point : result.mapPoints) {
        EXPECT_TRUE(point.allFinite());
        EXPECT_GT(point.z(), dataset->origin.z());
    }
    EXPECT_NEAR(result.averageFrameRate, plotter->frames / plotter->processingSeconds, 1e-6);
}

TEST_F(SchlamSystemResultTest, LostTrackingReturnsLastMapAndPathWithoutReadingAnotherFrame) {
    addTranslatedTextureFrames(2);
    dataset->images.emplace_back(256, 256, CV_8UC1, cv::Scalar(0));
    dataset->images.emplace_back(256, 256, CV_8UC1, cv::Scalar(0));
    const auto result = run();
    ASSERT_EQ(dataset->index, 3);
    ASSERT_EQ(plotter->frames, 3);
    EXPECT_FALSE(result.mapPoints.empty());
    ASSERT_EQ(result.predictedPath.size(), 2);
    EXPECT_EQ(result.predictedPath.back().timestamp, MemoryDataset::timestamp(1));
    EXPECT_NEAR(result.averageFrameRate, plotter->frames / plotter->processingSeconds, 1e-6);
}
}  // namespace
