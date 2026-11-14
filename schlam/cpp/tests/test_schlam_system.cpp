#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <map>
#include <functional>

#include "data/DatasetInterface.hpp"
#include "plotting/DummyPlotter.h"
#include "random.hpp"
#include "schlam/MotionModel.h"
#include "../src/schlam/feature_detection/orb/ORBFeatureDetector.h"
#include "../src/schlam/optimization/Optimizer.h"
#include "schlam/SchlamSystem.h"
#include "schlam/SchlamValidator.h"
#include "tft/transformer.hpp"

namespace {
class MemoryDataset : public DatasetInterface {
public:
    std::vector<cv::Mat> images;
    std::vector<utils::TTimestamp> imageTimestamps;
    std::size_t index = 0;
    Eigen::Vector3d origin{3, 4, 5};
    Eigen::Matrix3d orientation = Eigen::Matrix3d::Identity();
    std::chrono::nanoseconds gtTimestampOffset{0};
    std::vector<utils::TTimestamp> groundTruthTimestamps;
    std::size_t groundTruthIndex{0};
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
        return std::make_shared<ImageData>(imageTimestamps.empty() ? timestamp(current) : imageTimestamps.at(current),
            images[current], intrinsics, "gt/cam0");
    }
    std::shared_ptr<IMUData> getNextIMUData() override { return nullptr; }
    std::shared_ptr<GTData> getNextGTData() override {
        const auto imageTimestamp = imageTimestamps.empty() ? timestamp(index - 1) : imageTimestamps.at(index - 1);
        const auto gtTimestamp = groundTruthTimestamps.empty() ? imageTimestamp + gtTimestampOffset :
            groundTruthTimestamps.at(groundTruthIndex++);
        return std::make_shared<GTData>(gtTimestamp,
            origin + orientation * Eigen::Vector3d(0.1 * (index - 1), 0, 0),
            orientation, Eigen::Vector3d::Zero(), "world");
    }
};

class RecordingPlotter : public DummyPlotter {
public:
    double processingSeconds = 0.0;
    std::size_t frames = 0;
    const Map* map = nullptr;
    std::map<utils::TTimestamp, Eigen::Vector3d> initialPredictions;
    std::map<utils::TTimestamp, std::vector<std::shared_ptr<tft::RigidTransform3D>>> frustumPoses;
    std::function<void(const std::shared_ptr<ImageData>&)> beforeFrame;
    std::vector<std::size_t> matchingCounts;
    void updateFrustum(const std::shared_ptr<ImageData> image,
                       const std::shared_ptr<tft::RigidTransform3D> pose = nullptr) override {
        if (pose) {
            initialPredictions.try_emplace(image->mTimestamp, pose->mTranslation);
            frustumPoses[image->mTimestamp].push_back(pose);
        } else if (beforeFrame) beforeFrame(image);
    }
    void plotMatches(const cv::Mat&, const cv::Mat&,
                     const std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>& matches) override {
        matchingCounts.push_back(matches[0].size());
    }
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
    Eigen::Vector3d cameraOffset = Eigen::Vector3d::Zero();
    Eigen::Matrix3d cameraRotation = Eigen::Matrix3d::Identity();

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
        transformer->registerTransform(std::make_shared<tft::RigidTransform3D>(
            "gt/cam0", "gt/body", cameraRotation, cameraOffset));
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
    EXPECT_TRUE(result.groundTruthPath.empty());
    EXPECT_EQ(result.processedFrames, 0);
    EXPECT_DOUBLE_EQ(result.averageFrameRate, 0.0);
}

TEST_F(SchlamSystemResultTest, UninitializedFramesHaveTimingButNoInventedPath) {
    dataset->images.assign(2, cv::Mat(256, 256, CV_8UC1, cv::Scalar(0)));
    const auto result = run();
    EXPECT_TRUE(result.mapPoints.empty());
    EXPECT_TRUE(result.predictedPath.empty());
    EXPECT_EQ(result.groundTruthPath.size(), 2);
    EXPECT_EQ(result.processedFrames, 2);
    EXPECT_EQ(result.finalTrackingState, TrackingState::INITIALIZING);
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
    ASSERT_EQ(result.groundTruthPath.size(), 3);
    EXPECT_EQ(result.processedFrames, 3);
    EXPECT_EQ(result.finalTrackingState, TrackingState::TRACKING);
    EXPECT_TRUE(result.predictedPath.front().position.isApprox(dataset->origin));
    for (std::size_t frame = 0; frame < result.predictedPath.size(); ++frame) {
        EXPECT_EQ(result.predictedPath[frame].timestamp, MemoryDataset::timestamp(frame));
        EXPECT_EQ(result.groundTruthPath[frame].timestamp, MemoryDataset::timestamp(frame));
        EXPECT_TRUE(result.groundTruthPath[frame].rotation.isApprox(dataset->orientation));
        EXPECT_TRUE(result.groundTruthPath[frame].position.isApprox(
            dataset->origin + dataset->orientation * Eigen::Vector3d(0.1 * frame, 0, 0)));
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
    SchlamValidator::Options options;
    options.alignment = SchlamValidator::Alignment::NONE;
    const auto validation = SchlamValidator(options).validate(result);
    ASSERT_EQ(validation.status, SchlamValidator::Status::SUCCESS);
    EXPECT_EQ(validation.matchedPoses, 3);
    EXPECT_LT(validation.absoluteTranslation->rmse, 0.01);
}

TEST_F(SchlamSystemResultTest, LostTrackingReturnsLastMapAndPathWithoutReadingAnotherFrame) {
    settings.tracking.recoveryTimeoutSeconds = 0.0;
    addTranslatedTextureFrames(2);
    dataset->images.emplace_back(256, 256, CV_8UC1, cv::Scalar(0));
    dataset->images.emplace_back(256, 256, CV_8UC1, cv::Scalar(0));
    const auto result = run();
    ASSERT_EQ(dataset->index, 3);
    ASSERT_EQ(plotter->frames, 3);
    EXPECT_FALSE(result.mapPoints.empty());
    ASSERT_EQ(result.predictedPath.size(), 2);
    EXPECT_EQ(result.groundTruthPath.size(), 3);
    EXPECT_EQ(result.processedFrames, 3);
    EXPECT_EQ(result.finalTrackingState, TrackingState::LOST);
    EXPECT_EQ(result.predictedPath.back().timestamp, MemoryDataset::timestamp(1));
    EXPECT_NEAR(result.averageFrameRate, plotter->frames / plotter->processingSeconds, 1e-6);
}

TEST_F(SchlamSystemResultTest, RejectedFramesEndInRecoveringAndDoNotEnterMotionHistory) {
    addTranslatedTextureFrames(2);
    dataset->images.assign({dataset->images[0], dataset->images[1], cv::Mat(256, 256, CV_8UC1, cv::Scalar(0))});
    const auto result = run();
    EXPECT_EQ(result.finalTrackingState, TrackingState::RECOVERING);
    ASSERT_EQ(result.predictedPath.size(), 2);
    EXPECT_EQ(result.processedFrames, 3);
    const auto& first = result.predictedPath[0].position;
    const auto& last = result.predictedPath[1].position;
    EXPECT_TRUE(motion.estimateLinearAt(MemoryDataset::timestamp(20))->mTranslation.isApprox(last + 2 * (last - first), 1e-9));
}

TEST_F(SchlamSystemResultTest, FourthOrderRecoveryPredictsPastTwoFramesAndReacquiresTracking) {
    settings.motion.predictionModel = schlam::config::MotionPredictionModel::FOURTH_ORDER;
    addTranslatedTextureFrames(7);
    for (int i : {2, 3, 4}) dataset->images[i] = cv::Mat(256, 256, CV_8UC1, cv::Scalar(0));
    const auto result = run();
    ASSERT_EQ(result.finalTrackingState, TrackingState::TRACKING);
    ASSERT_EQ(result.predictedPath.size(), 4);
    EXPECT_EQ(result.predictedPath[2].timestamp, MemoryDataset::timestamp(5));
    EXPECT_EQ(result.predictedPath[3].timestamp, MemoryDataset::timestamp(6));
    const Eigen::Vector3d initialVelocity = result.predictedPath[1].position - result.predictedPath[0].position;
    for (int i : {3, 4, 5}) {
        EXPECT_TRUE(plotter->initialPredictions.at(MemoryDataset::timestamp(i)).isApprox(
            result.predictedPath[1].position + (i - 1) * initialVelocity, 1e-9));
    }
}

TEST_F(SchlamSystemResultTest, LinearRecoveryAlsoUsesTheConfiguredTimeHorizon) {
    settings.motion.predictionModel = schlam::config::MotionPredictionModel::LINEAR;
    addTranslatedTextureFrames(6);
    for (int i : {2, 3, 4}) dataset->images[i] = cv::Mat(256, 256, CV_8UC1, cv::Scalar(0));
    const auto result = run();
    ASSERT_EQ(result.finalTrackingState, TrackingState::TRACKING);
    ASSERT_EQ(result.predictedPath.size(), 3);
    const Eigen::Vector3d step = result.predictedPath[1].position - result.predictedPath[0].position;
    EXPECT_TRUE(plotter->initialPredictions.at(MemoryDataset::timestamp(5)).isApprox(
        result.predictedPath[1].position + 4 * step, 1e-9));
}

TEST_F(SchlamSystemResultTest, FourthOrderRetriesLinearMatchingBeforeRejectingAFrame) {
    settings.motion.predictionModel = schlam::config::MotionPredictionModel::FOURTH_ORDER;
    settings.tracking.projectionWindowPixels = 3.0;
    addTranslatedTextureFrames(9);
    plotter->beforeFrame = [&](const std::shared_ptr<ImageData>& image) {
        if (image->mTimestamp != MemoryDataset::timestamp(8)) return;
        // Earlier accepted-pose outliers disturb the polynomial while leaving
        // the latest two poses (and therefore linear velocity) unchanged.
        MotionModel disturbedHistory;
        disturbedHistory.addPose(std::make_shared<tft::RigidTransform3D>(
            "pred/cam0", "world", Eigen::Matrix3d::Identity(), dataset->origin), MemoryDataset::timestamp(0));
        for (int i = 1; i <= 7; ++i) {
            const auto& accepted = plotter->frustumPoses.at(MemoryDataset::timestamp(i)).back();
            Eigen::Vector3d position = accepted->mTranslation;
            if (i >= 2 && i <= 5) position.y() += 1.0;
            disturbedHistory.addPose(std::make_shared<tft::RigidTransform3D>(
                accepted->mSource, accepted->mTarget, accepted->mRotation, position), MemoryDataset::timestamp(i));
        }
        motion = std::move(disturbedHistory);
    };
    const auto result = run();
    ASSERT_EQ(result.finalTrackingState, TrackingState::TRACKING);
    ASSERT_EQ(result.predictedPath.size(), 9);
    const auto& attempts = plotter->frustumPoses.at(MemoryDataset::timestamp(8));
    ASSERT_EQ(attempts.size(), 3); // Fourth-order guess, linear retry, accepted pose.
    const Eigen::Vector3d linear = 2 * result.predictedPath[7].position - result.predictedPath[6].position;
    EXPECT_GT((attempts[0]->mTranslation - linear).norm(), 0.01);
    EXPECT_TRUE(attempts[1]->mTranslation.isApprox(linear, 1e-10));
    EXPECT_TRUE(attempts[2]->mTranslation.isApprox(result.predictedPath.back().position, 1e-10));
}

TEST_F(SchlamSystemResultTest, DisabledPredictionUsesTheLastAcceptedPoseDuringRecovery) {
    settings.motion.predictionModel = schlam::config::MotionPredictionModel::NONE;
    addTranslatedTextureFrames(2);
    dataset->images.emplace_back(256, 256, CV_8UC1, cv::Scalar(0));
    dataset->images.emplace_back(256, 256, CV_8UC1, cv::Scalar(0));
    const auto result = run();
    EXPECT_EQ(result.finalTrackingState, TrackingState::RECOVERING);
    ASSERT_EQ(result.predictedPath.size(), 2);
    for (int i : {2, 3}) EXPECT_TRUE(plotter->initialPredictions.at(MemoryDataset::timestamp(i)).isApprox(
        result.predictedPath.back().position, 1e-12));
}

TEST_F(SchlamSystemResultTest, RejectedOptimizationsExpireDespiteEnoughDescriptorMatches) {
    settings.tracking.minimumPoseInliers = 10000;
    settings.tracking.recoveryTimeoutSeconds = 0.1;
    addTranslatedTextureFrames(6);
    const auto result = run();
    ASSERT_EQ(result.finalTrackingState, TrackingState::LOST);
    ASSERT_EQ(result.predictedPath.size(), 2);
    EXPECT_EQ(result.processedFrames, 5);
    EXPECT_EQ(dataset->index, 5);
    ASSERT_GE(plotter->matchingCounts.size(), 3);
    EXPECT_GE(plotter->matchingCounts.back(), 12);
    EXPECT_FALSE(plotter->initialPredictions.contains(MemoryDataset::timestamp(4)));
}

TEST_F(SchlamSystemResultTest, RecoveryTimeoutUsesDatasetTimeWithIrregularFrameSpacing) {
    settings.tracking.recoveryTimeoutSeconds = 0.2;
    addTranslatedTextureFrames(2);
    dataset->images.emplace_back(256, 256, CV_8UC1, cv::Scalar(0));
    dataset->images.emplace_back(256, 256, CV_8UC1, cv::Scalar(0));
    dataset->images.emplace_back(256, 256, CV_8UC1, cv::Scalar(0));
    const auto start = MemoryDataset::timestamp(0);
    dataset->imageTimestamps = {start, start + std::chrono::milliseconds(50),
        start + std::chrono::milliseconds(60), start + std::chrono::milliseconds(500),
        start + std::chrono::milliseconds(510)};
    const auto result = run();
    EXPECT_EQ(result.finalTrackingState, TrackingState::LOST);
    EXPECT_EQ(result.processedFrames, 4);
    EXPECT_EQ(dataset->index, 4);
    EXPECT_FALSE(plotter->initialPredictions.contains(dataset->imageTimestamps[3]));
}

TEST_F(SchlamSystemResultTest, AcceptedPoseRestartsTheRecoveryWindow) {
    settings.tracking.recoveryTimeoutSeconds = 0.1;
    addTranslatedTextureFrames(8);
    for (int i : {2, 4, 6}) dataset->images[i] = cv::Mat(256, 256, CV_8UC1, cv::Scalar(0));
    const auto result = run();
    EXPECT_EQ(result.finalTrackingState, TrackingState::TRACKING);
    ASSERT_EQ(result.predictedPath.size(), 5);
    EXPECT_EQ(result.predictedPath.back().timestamp, MemoryDataset::timestamp(7));
    EXPECT_EQ(result.processedFrames, 8);
}

TEST_F(SchlamSystemResultTest, GroundTruthRetainsSensorTimestampAndCameraExtrinsics) {
    dataset->images.assign(1, cv::Mat(256, 256, CV_8UC1, cv::Scalar(0)));
    dataset->gtTimestampOffset = std::chrono::milliseconds(-5);
    dataset->orientation = Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    cameraOffset = Eigen::Vector3d(0.2, -0.3, 0.1);
    cameraRotation = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()).toRotationMatrix();
    const auto result = run();
    ASSERT_EQ(result.groundTruthPath.size(), 1);
    const auto& gt = result.groundTruthPath.front();
    EXPECT_EQ(gt.timestamp, MemoryDataset::timestamp(0) + dataset->gtTimestampOffset);
    EXPECT_TRUE(gt.position.isApprox(dataset->origin + dataset->orientation * cameraOffset));
    EXPECT_TRUE(gt.rotation.isApprox(dataset->orientation * cameraRotation));
}

TEST_F(SchlamSystemResultTest, GroundTruthSamplesSkippedBySynchronizationRemainInResult) {
    dataset->images.assign(2, cv::Mat(256, 256, CV_8UC1, cv::Scalar(0)));
    dataset->groundTruthTimestamps = {
        MemoryDataset::timestamp(0) - std::chrono::milliseconds(10),
        MemoryDataset::timestamp(0) - std::chrono::milliseconds(5),
        MemoryDataset::timestamp(0), MemoryDataset::timestamp(1)};
    const auto result = run();
    ASSERT_EQ(result.groundTruthPath.size(), 4);
    for (std::size_t i = 0; i < result.groundTruthPath.size(); ++i) {
        EXPECT_EQ(result.groundTruthPath[i].timestamp, dataset->groundTruthTimestamps[i]);
    }
    EXPECT_EQ(result.processedFrames, 2);
}
}  // namespace
