#include "schlam/MotionModel.h"

#include <gtest/gtest.h>
#include <Eigen/Geometry>

#include <chrono>
#include <numbers>
#include <stdexcept>

namespace {
utils::TTimestamp at(std::chrono::nanoseconds offset) {
    // Use a realistic epoch so sub-frame intervals cannot be lost by converting
    // absolute timestamps to floating-point seconds before subtracting them.
    return utils::TTimestamp(std::chrono::seconds(1700000000) + offset);
}

std::shared_ptr<tft::RigidTransform3D> pose(const Eigen::Vector3d& position,
                           const Eigen::Matrix3d& rotation = Eigen::Matrix3d::Identity(),
                           const std::string& source = "gt/cam0",
                           const std::string& target = "world") {
    return std::make_shared<tft::RigidTransform3D>(source, target, rotation, position);
}
}

using namespace std::chrono_literals;

TEST(MotionModelTest, EmptyHistoryCannotPredict) {
    MotionModel model;
    EXPECT_THROW(model.estimateLinearAt(at(0ns)), std::logic_error);
}

TEST(MotionModelTest, SinglePoseAssumesNoMotion) {
    MotionModel model;
    const auto initial = pose({1, 2, 3},
        Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitY()).toRotationMatrix(), "cam_ref", "map");
    model.addPose(initial, at(10ms));

    const auto predicted = model.estimateLinearAt(at(35ms));
    EXPECT_TRUE(predicted->matrix().isApprox(initial->matrix(), 1e-12));
    EXPECT_EQ(predicted->mSource, "cam_ref");
    EXPECT_EQ(predicted->mTarget, "map");
}

TEST(MotionModelTest, ExtrapolatesLinearAndAngularVelocityAtFractionalFrameIntervals) {
    MotionModel model;
    const Eigen::Matrix3d base = Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitX()).toRotationMatrix();
    const auto rotate = [&base](double angle) -> Eigen::Matrix3d {
        return base * Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    };
    model.addPose(pose({1, 2, 3}, rotate(0.2), "cam_ref"), at(0ns));
    model.addPose(pose({2, 4, 2}, rotate(0.4), "cam_pred0"), at(50ms));

    const auto predicted = model.estimateLinearAt(at(75ms));
    EXPECT_TRUE(predicted->mTranslation.isApprox(Eigen::Vector3d(2.5, 5, 1.5), 1e-12));
    EXPECT_TRUE(predicted->mRotation.isApprox(rotate(0.5), 1e-12));
    EXPECT_TRUE((predicted->mRotation.transpose() * predicted->mRotation).isIdentity(1e-12));
    EXPECT_NEAR(predicted->mRotation.determinant(), 1.0, 1e-12);
    EXPECT_EQ(predicted->mSource, "cam_pred0");
    EXPECT_EQ(predicted->mTarget, "world");

    const auto midpoint = model.estimateLinearAt(at(25ms));
    EXPECT_TRUE(midpoint->mTranslation.isApprox(Eigen::Vector3d(1.5, 3, 2.5), 1e-12));
    EXPECT_TRUE(midpoint->mRotation.isApprox(rotate(0.3), 1e-12));
    EXPECT_TRUE(model.estimateLinearAt(at(50ms))->matrix().isApprox(
        pose({2, 4, 2}, rotate(0.4))->matrix(), 1e-12));
}

TEST(MotionModelTest, PredictionUsesLatestMotionAndActualSampleInterval) {
    MotionModel model;
    model.addPose(pose({-10, 0, 0}), at(0ns));
    model.addPose(pose({1, 0, 0}), at(50ms));
    model.addPose(pose({3, 0, 0}), at(150ms));

    EXPECT_TRUE(model.estimateLinearAt(at(175ms))->mTranslation.isApprox(
        Eigen::Vector3d(3.5, 0, 0), 1e-12));
    EXPECT_TRUE(model.estimateLinearAt(at(200ms))->mTranslation.isApprox(
        Eigen::Vector3d(4, 0, 0), 1e-12));
}

TEST(MotionModelTest, LimitsExtrapolationAfterTrackingGap) {
    MotionModel model;
    const auto rotate = [](double radians) {
        return Eigen::AngleAxisd(radians, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    };
    model.addPose(pose({0, 0, 0}, rotate(0.0)), at(0ns));
    model.addPose(pose({1, 0, 0}, rotate(0.2)), at(50ms));

    // A long gap must not send the next search window many frames ahead.
    const auto predicted = model.estimateLinearAt(at(500ms));
    EXPECT_TRUE(predicted->mTranslation.isApprox(Eigen::Vector3d(3, 0, 0), 1e-12));
    EXPECT_TRUE(predicted->mRotation.isApprox(rotate(0.6), 1e-12));

    MotionModel cautious(0.5);
    cautious.addPose(pose({0, 0, 0}), at(0ns));
    cautious.addPose(pose({1, 0, 0}), at(50ms));
    EXPECT_TRUE(cautious.estimateLinearAt(at(500ms))->mTranslation.isApprox(
        Eigen::Vector3d(1.5, 0, 0), 1e-12));
}

TEST(MotionModelTest, RotationTakesShortestPathAcrossAngleWraparound) {
    MotionModel model;
    const auto rotate = [](double degrees) {
        return Eigen::AngleAxisd(degrees * std::numbers::pi / 180.0,
            Eigen::Vector3d::UnitZ()).toRotationMatrix();
    };
    model.addPose(pose(Eigen::Vector3d::Zero(), rotate(170)), at(0ns));
    model.addPose(pose(Eigen::Vector3d::Zero(), rotate(-170)), at(50ms));

    EXPECT_TRUE(model.estimateLinearAt(at(100ms))->mRotation.isApprox(rotate(-150), 1e-12));
}

TEST(MotionModelTest, StationaryPosesRemainStationary) {
    MotionModel model;
    const auto initial = pose({1, -2, 3});
    model.addPose(initial, at(0ns));
    model.addPose(initial, at(50ms));

    EXPECT_TRUE(model.estimateLinearAt(at(100ms))->matrix().isApprox(initial->matrix(), 1e-12));
}

TEST(MotionModelTest, RejectsNonIncreasingTimestampsWithoutChangingHistory) {
    MotionModel model;
    model.addPose(pose({0, 0, 0}), at(0ns));
    model.addPose(pose({1, 0, 0}), at(50ms));
    EXPECT_THROW(model.addPose(pose({100, 0, 0}), at(50ms)), std::invalid_argument);
    EXPECT_THROW(model.addPose(pose({100, 0, 0}), at(25ms)), std::invalid_argument);

    EXPECT_TRUE(model.estimateLinearAt(at(100ms))->mTranslation.isApprox(
        Eigen::Vector3d(2, 0, 0), 1e-12));
}

TEST(MotionModelTest, RejectsDifferentTargetFramesWithoutChangingHistory) {
    MotionModel model;
    model.addPose(pose({0, 0, 0}), at(0ns));
    model.addPose(pose({1, 0, 0}), at(50ms));
    EXPECT_THROW(model.addPose(pose({100, 0, 0}, Eigen::Matrix3d::Identity(), "gt/cam0", "map"),
        at(100ms)), std::invalid_argument);

    const auto predicted = model.estimateLinearAt(at(100ms));
    EXPECT_EQ(predicted->mTarget, "world");
    EXPECT_TRUE(predicted->mTranslation.isApprox(Eigen::Vector3d(2, 0, 0), 1e-12));
}
