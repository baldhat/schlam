#include "schlam/SlamSystem.h"

#include "random.hpp"
#include "schlam/Map.h"
#include "schlam/ORBFeatureDetector.h"
#include "schlam/Optimizer.h"
#include "tft/transformer.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>

class SlamSystemTest : public ::testing::Test {
protected:
    const Eigen::Matrix3d intrinsics = (Eigen::Matrix3d() <<
        500, 0, 376, 0, 500, 240, 0, 0, 1).finished();
    const std::mt19937 originalEngine = utils::randomEngine();
    const cv::RNG originalOpenCVEngine = cv::theRNG();

    void SetUp() override {
        utils::randomEngine().seed(utils::kRandomSeed);
        cv::theRNG() = cv::RNG(utils::kRandomSeed);
    }

    void TearDown() override {
        utils::randomEngine() = originalEngine;
        cv::theRNG() = originalOpenCVEngine;
    }

    auto makeSystem(std::shared_ptr<tft::Transformer> transformer = std::make_shared<tft::Transformer>()) {
        auto detector = std::make_shared<ORBFeatureDetector>(3000, nullptr, 1);
        return std::make_unique<SlamSystem>(std::move(transformer),
            detector, detector, std::make_shared<Optimizer>());
    }

    SensorPacket packet(cv::Mat image, int index, bool externalPose = false) {
        auto imageData = std::make_shared<ImageData>(
            utils::TTimestamp(std::chrono::milliseconds(index * 50)), image, intrinsics, "cam0");
        return {imageData, externalPose ? std::make_shared<tft::RigidTransform3D>(
            "cam0", "world", Eigen::Matrix3d::Identity(), Eigen::Vector3d(index * 0.3, 0, 0)) : nullptr};
    }

    cv::Mat blank() {
        return cv::Mat(480, 752, CV_8UC1, cv::Scalar(100));
    }

    cv::Mat scene(int view) {
        cv::Mat image = blank();
        cv::RNG rng(12345);
        // Textured patches lie at four different depths. A horizontal camera
        // displacement of 0.3 gives exact pixel shifts of 12, 16, 20, or 24.
        for (int row = 0; row < 6; ++row) {
            for (int col = 0; col < 9; ++col) {
                cv::Mat patch(40, 40, CV_8UC1);
                rng.fill(patch, cv::RNG::UNIFORM, 0, 256);
                const int shift = 12 + 4 * ((row + col) % 4);
                const cv::Rect destination(80 + 64 * col - view * shift, 48 + 64 * row, 40, 40);
                const auto clipped = destination & cv::Rect(0, 0, image.cols, image.rows);
                if (clipped.empty()) continue;
                patch(cv::Rect(clipped.x - destination.x, clipped.y - destination.y,
                               clipped.width, clipped.height)).copyTo(image(clipped));
            }
        }
        return image;
    }
};

TEST_F(SlamSystemTest, ConstructorRejectsMissingComponents) {
    auto transformer = std::make_shared<tft::Transformer>();
    auto detector = std::make_shared<ORBFeatureDetector>(1000, nullptr, 1);
    auto optimizer = std::make_shared<Optimizer>();
    EXPECT_THROW(SlamSystem(nullptr, detector, detector, optimizer), std::invalid_argument);
    EXPECT_THROW(SlamSystem(transformer, nullptr, detector, optimizer), std::invalid_argument);
    EXPECT_THROW(SlamSystem(transformer, detector, nullptr, optimizer), std::invalid_argument);
    EXPECT_THROW(SlamSystem(transformer, detector, detector, nullptr), std::invalid_argument);
}

TEST_F(SlamSystemTest, BlankImagesRemainInitializingWithoutAMapOrAcceptedPose) {
    auto system = makeSystem();
    EXPECT_EQ(system->getState(), SlamState::Initializing);
    EXPECT_TRUE(system->getMap().expired());
    const auto first = system->process(packet(blank(), 0));
    const auto second = system->process(packet(blank(), 1));
    for (const auto& result : {first, second}) {
        EXPECT_EQ(result.state, SlamState::Initializing);
        EXPECT_EQ(result.cameraToWorld, nullptr);
        EXPECT_EQ(result.predictedCameraToWorld, nullptr);
        ASSERT_NE(result.frame, nullptr);
        EXPECT_EQ(result.frame->mMapToCamera, nullptr);
        EXPECT_EQ(result.diagnostics.featureCount, 0);
        EXPECT_TRUE(result.diagnostics.referenceChanged);
        EXPECT_FALSE(result.diagnostics.mapInitialized);
        EXPECT_GT(result.diagnostics.processingTime.count(), 0);
    }
    EXPECT_EQ(second.matchReferenceFrame, first.frame);
    EXPECT_TRUE(system->getMap().expired());
}

TEST_F(SlamSystemTest, InvalidPacketsDoNotConsumeTheTimestampOrChangeState) {
    auto system = makeSystem();
    EXPECT_THROW(system->process({}), std::invalid_argument);
    EXPECT_THROW(system->process(packet(cv::Mat(), 0)), std::invalid_argument);
    EXPECT_THROW(system->process(packet(cv::Mat(64, 64, CV_8UC3), 0)), std::invalid_argument);
    auto invalid = packet(blank(), 0);
    auto image = blank();
    invalid.image = std::make_shared<ImageData>(utils::TTimestamp{}, image,
        Eigen::Matrix3d::Zero(), "cam0");
    EXPECT_THROW(system->process(invalid), std::invalid_argument);
    invalid = packet(blank(), 0);
    invalid.initializationCameraToWorld = tft::RigidTransform3D::identity("other_camera", "world");
    EXPECT_THROW(system->process(invalid), std::invalid_argument);
    invalid.initializationCameraToWorld = std::make_shared<tft::RigidTransform3D>(
        "cam0", "world", Eigen::Matrix3d::Identity(),
        Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0, 0));
    EXPECT_THROW(system->process(invalid), std::invalid_argument);
    EXPECT_EQ(system->getState(), SlamState::Initializing);
    EXPECT_TRUE(system->getMap().expired());
    EXPECT_NO_THROW(system->process(packet(blank(), 0)));
}

TEST_F(SlamSystemTest, RejectsRepeatedOrDecreasingImageTimestamps) {
    auto system = makeSystem();
    system->process(packet(blank(), 1));
    EXPECT_THROW(system->process(packet(blank(), 1)), std::invalid_argument);
    EXPECT_THROW(system->process(packet(blank(), 0)), std::invalid_argument);
    EXPECT_NO_THROW(system->process(packet(blank(), 2)));
}

TEST_F(SlamSystemTest, IdenticalViewsDoNotInitializeDespiteEnoughFeatureMatches) {
    auto system = makeSystem();
    system->process(packet(scene(0), 0));
    const auto result = system->process(packet(scene(0), 1));
    EXPECT_GE(result.diagnostics.matchCount, 100);
    EXPECT_EQ(result.state, SlamState::Initializing);
    EXPECT_FALSE(result.diagnostics.referenceChanged);
    EXPECT_FALSE(result.diagnostics.mapInitialized);
    EXPECT_EQ(result.cameraToWorld, nullptr);
    EXPECT_TRUE(system->getMap().expired());
}

TEST_F(SlamSystemTest, InitializesWithoutGroundTruthInArbitraryUnits) {
    auto system = makeSystem();
    system->process(packet(scene(0), 0));
    const auto result = system->process(packet(scene(1), 1));
    ASSERT_EQ(result.state, SlamState::Tracking);
    ASSERT_NE(result.cameraToWorld, nullptr);
    const auto map = system->getMap().lock();
    ASSERT_NE(map, nullptr);
    EXPECT_TRUE(result.diagnostics.mapInitialized);
    EXPECT_TRUE(result.diagnostics.keyFrameAdded);
    EXPECT_NEAR(result.cameraToWorld->mTranslation.norm(), 1, 1e-6);
    EXPECT_EQ(map->mKeyFrames.size(), 2);
    EXPECT_EQ(map->mPoints.size(), result.diagnostics.inlierCount);
    EXPECT_TRUE(map->mKeyFrames.at(0).mMapToCamera->matrix().isIdentity(1e-12));
}

TEST_F(SlamSystemTest, MapLockProtectsReadsAndWeakHandleExpiresAfterLastOwner) {
    auto system = makeSystem();
    EXPECT_TRUE(system->getMap().expired());
    system->process(packet(scene(0), 0));
    ASSERT_EQ(system->process(packet(scene(1), 1)).state, SlamState::Tracking);
    const auto handle = system->getMap();
    {
        const auto map = handle.lock();
        ASSERT_NE(map, nullptr);
        const auto pointCount = map->getSnapshot().points.size();
        system.reset();
        // A reader that already locked the handle may finish after system shutdown.
        EXPECT_FALSE(handle.expired());
        EXPECT_EQ(map->getSnapshot().points.size(), pointCount);
        EXPECT_EQ(map->getSnapshot().keyFrames.size(), 2);
    }
    EXPECT_TRUE(handle.expired());
    EXPECT_EQ(handle.lock(), nullptr);
}

TEST_F(SlamSystemTest, ZeroExternalBaselineDefersInitialization) {
    auto system = makeSystem();
    system->process(packet(scene(0), 0, true));
    auto stationaryExternalPose = packet(scene(1), 1, true);
    stationaryExternalPose.initializationCameraToWorld = tft::RigidTransform3D::identity("cam0", "world");
    const auto result = system->process(stationaryExternalPose);
    EXPECT_EQ(result.state, SlamState::Initializing);
    EXPECT_FALSE(result.diagnostics.mapInitialized);
    EXPECT_EQ(result.cameraToWorld, nullptr);
    EXPECT_EQ(result.frame->mMapToCamera, nullptr);
    EXPECT_TRUE(system->getMap().expired());
}

TEST_F(SlamSystemTest, TracksLosesRecoversAndAddsKeyframesWithoutAViewerOrDataLoader) {
    auto transformer = std::make_shared<tft::Transformer>();
    auto system = makeSystem(transformer);
    system->process(packet(scene(0), 0, true));
    const auto initialized = system->process(packet(scene(1), 1, true));
    ASSERT_EQ(initialized.state, SlamState::Tracking);
    ASSERT_NE(initialized.cameraToWorld, nullptr);
    EXPECT_NEAR((initialized.cameraToWorld->mTranslation - Eigen::Vector3d(0.3, 0, 0)).norm(), 0, 0.02);
    const auto map = system->getMap().lock();
    ASSERT_NE(map, nullptr);

    const auto tracked = system->process(packet(scene(2), 2));
    ASSERT_EQ(tracked.state, SlamState::Tracking);
    ASSERT_NE(tracked.cameraToWorld, nullptr);
    EXPECT_NEAR((tracked.cameraToWorld->mTranslation - Eigen::Vector3d(0.6, 0, 0)).norm(), 0, 0.03);
    EXPECT_GE(tracked.diagnostics.inlierCount, 50);
    EXPECT_EQ(tracked.inliers.size(), tracked.diagnostics.matchCount);

    const auto lost = system->process(packet(blank(), 3));
    EXPECT_EQ(lost.state, SlamState::Lost);
    EXPECT_EQ(system->getState(), SlamState::Lost);
    EXPECT_EQ(lost.cameraToWorld, nullptr);
    EXPECT_EQ(lost.frame->mMapToCamera, nullptr);
    EXPECT_NE(lost.predictedCameraToWorld, nullptr);
    EXPECT_FALSE(lost.diagnostics.keyFrameAdded);
    EXPECT_EQ(system->getMap().lock(), map);
    EXPECT_EQ(map->mKeyFrames.size(), 2);

    const auto recovered = system->process(packet(scene(4), 4));
    ASSERT_EQ(recovered.state, SlamState::Tracking);
    ASSERT_NE(recovered.cameraToWorld, nullptr);
    EXPECT_NEAR((recovered.cameraToWorld->mTranslation - Eigen::Vector3d(1.2, 0, 0)).norm(), 0, 0.05);
    EXPECT_EQ(system->getMap().lock(), map);

    bool addedKeyframe = false;
    for (int index = 5; index <= 13; ++index) {
        const auto result = system->process(packet(scene(4), index));
        ASSERT_EQ(result.state, SlamState::Tracking) << "Frame " << index;
        ASSERT_NE(result.cameraToWorld, nullptr);
        addedKeyframe |= result.diagnostics.keyFrameAdded;
    }
    EXPECT_TRUE(addedKeyframe);
    EXPECT_GE(map->mKeyFrames.size(), 3);
    EXPECT_EQ(transformer->getRootedTransforms().size(), 3);
    EXPECT_EQ(map->getSnapshot().points.size(), map->mPoints.size());
    // Earlier results retain their accepted pose after further processing.
    EXPECT_NEAR((tracked.cameraToWorld->mTranslation - Eigen::Vector3d(0.6, 0, 0)).norm(), 0, 0.03);
}
