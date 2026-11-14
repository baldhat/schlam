#include "schlam/MotionModel.h"
#include "tft/rigid_transform_3d.hpp"

#include <gtest/gtest.h>
#include <Eigen/Geometry>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <limits>
#include <random>

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

TEST(MotionModelTest, FourthOrderRequiresHistoryAndHoldsASinglePose) {
    MotionModel model;
    EXPECT_THROW(model.estimateFourthOrderAt(at(0ns)), std::logic_error);
    const auto accepted = pose({1, 2, 3}, Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()).toRotationMatrix());
    model.addPose(accepted, at(0ns));
    EXPECT_TRUE(model.estimateFourthOrderAt(at(100ms))->matrix().isApprox(accepted->matrix(), 1e-12));
}

TEST(MotionModelTest, FourthOrderImprovesSmoothCurvedMotionWithIrregularTimestamps) {
    MotionModel model;
    const Eigen::Vector3d origin(3, -2, 5), velocity(1, -2, 0.5), acceleration(2, 1, -1);
    const Eigen::Vector3d jerk(-3, 4, 2), snap(24, -12, 8);
    const auto position = [&](double t) -> Eigen::Vector3d {
        return origin + velocity * t + acceleration * t * t / 2 + jerk * t * t * t / 6 + snap * std::pow(t, 4) / 24;
    };
    for (const auto time : {-550ms, -490ms, -440ms, -370ms, -310ms, -260ms,
                            -220ms, -170ms, -130ms, -90ms, -40ms, 0ms}) {
        model.addPose(pose(position(std::chrono::duration<double>(time).count())), at(time));
    }
    const auto prediction = model.estimateFourthOrderAt(at(40ms), 500ms);
    const double error = (prediction->mTranslation - position(0.04)).norm();
    const double linearError = (model.estimateLinearAt(at(40ms))->mTranslation - position(0.04)).norm();
    EXPECT_LT(error, linearError * 0.75);
    EXPECT_TRUE(model.estimateFourthOrderAt(at(0ns))->mTranslation.isApprox(origin, 1e-12));
}

TEST(MotionModelTest, FourthOrderUsesOnlyTheTwelveLatestAcceptedPoses) {
    MotionModel model;
    model.addPose(pose({10000, -10000, 10000}), at(-1s));
    for (int i = 0; i < 12; ++i) model.addPose(pose({static_cast<double>(i), 0, 0}), at(50ms * i));
    EXPECT_TRUE(model.estimateFourthOrderAt(at(600ms))->mTranslation.isApprox(Eigen::Vector3d(12, 0, 0), 1e-10));
    // Retaining more history must not change the original linear predictor.
    EXPECT_TRUE(model.estimateLinearAt(at(600ms))->mTranslation.isApprox(Eigen::Vector3d(12, 0, 0), 1e-12));
}

TEST(MotionModelTest, FourthOrderUsesLinearPredictionDuringWarmup) {
    for (int samples = 2; samples <= 7; ++samples) {
        MotionModel model;
        const auto coordinate = [samples](double t) { return 1 + t + std::pow(t, samples - 1); };
        for (int i = 0; i < samples; ++i) {
            model.addPose(pose({coordinate(i * 0.05), 0, 0}), at(50ms * i));
        }
        EXPECT_TRUE(model.estimateFourthOrderAt(at(50ms * samples))->matrix().isApprox(
            model.estimateLinearAt(at(50ms * samples))->matrix(), 1e-12));
    }
}

TEST(MotionModelTest, FourthOrderPredictsRotationInALocalChartAndPreservesSO3) {
    MotionModel model;
    const Eigen::Matrix3d base = Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitX()).toRotationMatrix();
    const auto rotation = [&](double t) -> Eigen::Matrix3d {
        const Eigen::Vector3d vector(0.2 * t + 0.3 * t * t - 0.2 * std::pow(t, 3) + 0.1 * std::pow(t, 4),
            -0.1 * t + 0.05 * std::pow(t, 3), 0.15 * t * t);
        if (vector.norm() == 0.0) return base;
        return base * Eigen::AngleAxisd(vector.norm(), vector.normalized()).toRotationMatrix();
    };
    for (int i = -11; i <= 0; ++i) {
        const auto time = 50ms * i;
        model.addPose(pose(Eigen::Vector3d::Zero(), rotation(std::chrono::duration<double>(time).count())), at(time));
    }
    const auto predicted = model.estimateFourthOrderAt(at(50ms));
    const double error = Eigen::AngleAxisd(rotation(0.05).transpose() * predicted->mRotation).angle();
    const double linearError = Eigen::AngleAxisd(rotation(0.05).transpose() * model.estimateLinearAt(at(50ms))->mRotation).angle();
    EXPECT_LT(error, linearError);
    EXPECT_TRUE((predicted->mRotation.transpose() * predicted->mRotation).isIdentity(1e-12));
    EXPECT_NEAR(predicted->mRotation.determinant(), 1.0, 1e-12);
}

TEST(MotionModelTest, FourthOrderUnwrapsRotationHistoryAcrossPi) {
    MotionModel model;
    const auto rotate = [](double degrees) {
        return Eigen::AngleAxisd(degrees * std::numbers::pi / 180, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    };
    for (int i = 0; i < 12; ++i) model.addPose(pose(Eigen::Vector3d::Zero(), rotate(60 * i)), at(50ms * i));
    EXPECT_TRUE(model.estimateFourthOrderAt(at(600ms))->mRotation.isApprox(rotate(720), 1e-10));
}

TEST(MotionModelTest, ExplicitRecoveryHorizonOverridesIntervalCapWithoutAdvancingHistory) {
    MotionModel model;
    model.addPose(pose({0, 0, 0}), at(0ns));
    model.addPose(pose({1, 0, 0}), at(50ms));
    for (const auto time : {150ms, 200ms, 250ms, 500ms}) {
        const double expected = 1 + std::min(0.2, std::chrono::duration<double>(time - 50ms).count()) / 0.05;
        EXPECT_NEAR(model.estimateFourthOrderAt(at(time), 200ms)->mTranslation.x(), expected, 1e-10);
        EXPECT_NEAR(model.estimateLinearAt(at(time), 200ms)->mTranslation.x(), expected, 1e-12);
    }
    EXPECT_NEAR(model.estimateLinearAt(at(500ms))->mTranslation.x(), 3, 1e-12);
    EXPECT_NEAR(model.estimateFourthOrderAt(at(500ms))->mTranslation.x(), 3, 1e-12);
    EXPECT_NEAR(model.estimateFourthOrderAt(at(500ms), 0ms)->mTranslation.x(), 1, 1e-12);
}

TEST(MotionModelTest, BothPredictorsRejectInvalidTimeHorizons) {
    MotionModel model;
    model.addPose(pose(Eigen::Vector3d::Zero()), at(0ns));
    for (double value : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        const std::chrono::duration<double> horizon(value);
        EXPECT_THROW(model.estimateFourthOrderAt(at(100ms), horizon), std::invalid_argument);
        EXPECT_THROW(model.estimateLinearAt(at(100ms), horizon), std::invalid_argument);
    }
}

TEST(MotionModelTest, FourthOrderReducesNoisyStationaryAndConstantVelocityPredictionErrors) {
    // Independent noise on accepted poses previously produced ~7x the linear
    // prediction error even one frame ahead. Exercise both position and SO(3).
    for (bool moving : {false, true}) {
        SCOPED_TRACE(moving);
        MotionModel model;
        std::mt19937 generator(1729);
        std::normal_distribution<double> noise(0.0, 1.0);
        const Eigen::Vector3d origin(1, -2, 3), velocity(0.5, -0.3, 0.1);
        const Eigen::Vector3d angularVelocity(0.05, -0.08, 0.12);
        const auto rotation = [&](double t) -> Eigen::Matrix3d {
            return Eigen::AngleAxisd(moving ? angularVelocity.norm() * t : 0.0, angularVelocity.normalized()).toRotationMatrix();
        };
        double squaredPositionError = 0, squaredLinearPositionError = 0;
        double squaredRotationError = 0, squaredLinearRotationError = 0;
        for (int i = 0; i < 500; ++i) {
            const double time = 0.05 * i;
            const Eigen::Vector3d positionNoise(noise(generator), noise(generator), noise(generator));
            const Eigen::Vector3d rotationNoise(noise(generator), noise(generator), noise(generator));
            const Eigen::Vector3d position = origin + velocity * (moving ? time : 0.0) + 0.002 * positionNoise;
            const Eigen::Matrix3d orientation = rotation(time) *
                Eigen::AngleAxisd(0.001 * rotationNoise.norm(), rotationNoise.normalized()).toRotationMatrix();
            model.addPose(pose(position, orientation), at(50ms * i));
            if (i < 12) continue;
            const auto prediction = model.estimateFourthOrderAt(at(50ms * (i + 1)));
            const auto linear = model.estimateLinearAt(at(50ms * (i + 1)));
            const Eigen::Vector3d expected = origin + velocity * (moving ? time + 0.05 : 0.0);
            squaredPositionError += (prediction->mTranslation - expected).squaredNorm();
            squaredLinearPositionError += (linear->mTranslation - expected).squaredNorm();
            squaredRotationError += std::pow(Eigen::AngleAxisd(rotation(time + 0.05).transpose() * prediction->mRotation).angle(), 2);
            squaredLinearRotationError += std::pow(Eigen::AngleAxisd(rotation(time + 0.05).transpose() * linear->mRotation).angle(), 2);
            EXPECT_TRUE((prediction->mRotation.transpose() * prediction->mRotation).isIdentity(1e-12));
        }
        EXPECT_LT(std::sqrt(squaredPositionError / squaredLinearPositionError), 0.9);
        EXPECT_LT(std::sqrt(squaredRotationError / squaredLinearRotationError), 0.9);
    }
}

TEST(MotionModelTest, FourthOrderBoundsNoisyCorrectionsWithIrregularTimestamps) {
    MotionModel model;
    std::chrono::milliseconds time(0);
    Eigen::Vector3d previous = Eigen::Vector3d::Zero();
    for (int i = 0; i < 100; ++i) {
        const auto interval = (i % 3 == 0) ? 10ms : ((i % 3 == 1) ? 90ms : 50ms);
        time += interval;
        const Eigen::Vector3d position(0.001 * time.count(), i % 2 ? 0.02 : -0.02, 0.0);
        const auto orientation = Eigen::AngleAxisd(0.0001 * time.count() + (i % 2 ? 0.01 : -0.01),
            Eigen::Vector3d::UnitZ()).toRotationMatrix();
        model.addPose(pose(position, orientation), at(time));
        if (i >= 4) {
            const auto prediction = model.estimateFourthOrderAt(at(time + interval));
            const auto linear = model.estimateLinearAt(at(time + interval));
            EXPECT_LE((prediction->mTranslation - linear->mTranslation).norm(), 0.5 * (position - previous).norm() + 1e-12);
            const auto earlier = model.estimateLinearAt(at(time - interval));
            const double angularStep = Eigen::AngleAxisd(earlier->mRotation.transpose() * orientation).angle();
            EXPECT_LE(Eigen::AngleAxisd(linear->mRotation.transpose() * prediction->mRotation).angle(), 0.5 * angularStep + 1e-12);
        }
        previous = position;
    }
}

TEST(MotionModelTest, FourthOrderCoastsLinearlyDuringLongRecoveryWithoutAdvancingHistory) {
    MotionModel model;
    for (int i = 0; i < 12; ++i) {
        const double t = i * 0.05;
        model.addPose(pose({t + t * t, 0, 0},
            Eigen::AngleAxisd(0.1 * t + 0.05 * t * t, Eigen::Vector3d::UnitY()).toRotationMatrix()), at(50ms * i));
    }
    const auto next = model.estimateFourthOrderAt(at(600ms));
    EXPECT_GT(next->mTranslation.x(), model.estimateLinearAt(at(600ms))->mTranslation.x());
    // Recovery must not extrapolate jerk/snap out to ten frames. At two or
    // more frames it uses the original linear model and the explicit horizon.
    for (const auto gap : {100ms, 150ms, 500ms, 1000ms}) {
        EXPECT_TRUE(model.estimateFourthOrderAt(at(550ms + gap), 500ms)->matrix().isApprox(
            model.estimateLinearAt(at(550ms + gap), 500ms)->matrix(), 1e-12));
    }
    EXPECT_TRUE(model.estimateFourthOrderAt(at(600ms))->matrix().isApprox(next->matrix(), 1e-12));
}
