//
// Created by baldhat on 10/11/26.
//

#include "MotionModel.h"
#include "tft/rigid_transform_3d.hpp"

#include <Eigen/Geometry>
#include <Eigen/QR>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <numbers>

namespace {
constexpr std::size_t motionHistorySize = 12;
constexpr std::size_t minimumFitSamples = 8;
constexpr double derivativeRegularization = 1e-4;
constexpr double maximumCorrectionFraction = 0.5;
}

MotionModel::MotionModel(double maxExtrapolationIntervals)
    : mMaxExtrapolationIntervals(maxExtrapolationIntervals) {
    if (!std::isfinite(maxExtrapolationIntervals) || maxExtrapolationIntervals <= 0.0) {
        throw std::invalid_argument("MotionModel extrapolation limit must be positive and finite");
    }
}

void MotionModel::addPose(const std::shared_ptr<tft::RigidTransform3D> transform,
                          const utils::TTimestamp& timestamp) {
    if (!transform || !transform->mTranslation.allFinite() || !transform->mRotation.allFinite()) {
        throw std::invalid_argument("MotionModel pose must be finite and non-null");
    }
    if (!mPoses.empty()) {
        const auto& [lastPose, lastTimestamp] = mPoses.back();
        if (timestamp <= lastTimestamp) {
            throw std::invalid_argument("MotionModel timestamps must be strictly increasing");
        }
        if (transform->mTarget != lastPose->mTarget) {
            throw std::invalid_argument("MotionModel poses must share a target coordinate frame");
        }
    }

    mPoses.emplace_back(transform, timestamp);
    if (mPoses.size() > motionHistorySize) mPoses.erase(mPoses.begin());
}

std::shared_ptr<tft::RigidTransform3D> MotionModel::estimateLinearAt(
        const utils::TTimestamp& timestamp, std::optional<std::chrono::duration<double>> maximumExtrapolation) const {
    if (maximumExtrapolation && (!std::isfinite(maximumExtrapolation->count()) || maximumExtrapolation->count() < 0.0)) {
        throw std::invalid_argument("MotionModel prediction horizon must be nonnegative and finite");
    }
    if (mPoses.empty()) {
        throw std::logic_error("Cannot estimate motion without a pose");
    }

    const auto& [lastPose, lastTimestamp] = mPoses.back();
    if (mPoses.size() == 1) {
        return lastPose;
    }

    const auto& [previousPose, previousTimestamp] = mPoses[mPoses.size() - 2];
    const double interval = std::chrono::duration<double>(lastTimestamp - previousTimestamp).count();
    const double horizon = maximumExtrapolation ? maximumExtrapolation->count() : mMaxExtrapolationIntervals * interval;
    const double elapsed = std::clamp(std::chrono::duration<double>(timestamp - lastTimestamp).count(), -horizon, horizon);
    const double factor = elapsed / interval;

    const Eigen::Vector3d translation = lastPose->mTranslation +
        factor * (lastPose->mTranslation - previousPose->mTranslation);

    // Scale the shortest relative rotation, preserving SO(3).
    const Eigen::AngleAxisd relativeRotation(previousPose->mRotation.transpose() * lastPose->mRotation);
    const Eigen::Matrix3d rotation = lastPose->mRotation *
        Eigen::AngleAxisd(factor * relativeRotation.angle(), relativeRotation.axis()).toRotationMatrix();

    return std::make_shared<tft::RigidTransform3D>(lastPose->mSource, lastPose->mTarget, rotation, translation);
}

std::shared_ptr<tft::RigidTransform3D> MotionModel::estimateFourthOrderAt(
        const utils::TTimestamp& timestamp, std::optional<std::chrono::duration<double>> maximumExtrapolation) const {
    const auto linear = estimateLinearAt(timestamp, maximumExtrapolation);
    const auto& [lastPose, lastTimestamp] = mPoses.back();
    // Sparse polynomial interpolation amplifies noise even during warmup.
    if (mPoses.size() < minimumFitSamples || timestamp == lastTimestamp) return linear;

    const std::size_t count = mPoses.size();
    const auto& [previousPose, previousTimestamp] = mPoses[count - 2];
    const double interval = std::chrono::duration<double>(lastTimestamp - previousTimestamp).count();
    const double horizon = maximumExtrapolation ? maximumExtrapolation->count() : mMaxExtrapolationIntervals * interval;
    const double elapsed = std::clamp(std::chrono::duration<double>(timestamp - lastTimestamp).count(), -horizon, horizon);
    if (elapsed == 0.0 || std::abs(elapsed) >= 2.0 * interval) return linear;

    // Higher derivatives only adjust the next frame. During a gap their
    // correction fades to zero, while linear motion uses the recovery horizon.
    const double correctionTime = std::clamp(elapsed, -interval, interval);
    const double correctionWeight = std::min(1.0, 2.0 - std::abs(elapsed) / interval);
    const double span = std::chrono::duration<double>(lastTimestamp - std::get<1>(mPoses.front())).count();
    Eigen::Matrix<double, Eigen::Dynamic, 5> design = Eigen::Matrix<double, Eigen::Dynamic, 5>::Zero(count + 3, 5);
    Eigen::Matrix<double, Eigen::Dynamic, 6> samples = Eigen::Matrix<double, Eigen::Dynamic, 6>::Zero(count + 3, 6);

    // Express samples relative to the latest accepted pose. Select the
    // nearest equivalent rotation-vector branch while walking backwards, so a
    // window spanning +/-pi does not create artificial angular derivatives.
    Eigen::Vector3d nextRotationVector = Eigen::Vector3d::Zero();
    for (std::size_t reverse = count; reverse > 0; --reverse) {
        const auto i = reverse - 1;
        const auto& [sample, time] = mPoses[i];
        const double normalizedTime = std::chrono::duration<double>(time - lastTimestamp).count() / span;
        double power = 1.0;
        for (std::size_t column = 0; column < 5; ++column) {
            design(i, column) = power;
            power *= normalizedTime;
        }
        const Eigen::AngleAxisd relative(lastPose->mRotation.transpose() * sample->mRotation);
        Eigen::Vector3d axis = relative.axis();
        if (relative.angle() < 1e-12 && nextRotationVector.norm() > 1e-12) {
            axis = nextRotationVector.normalized();
        }
        const double turns = std::round((nextRotationVector.dot(axis) - relative.angle()) / (2 * std::numbers::pi));
        nextRotationVector = axis * (relative.angle() + turns * 2 * std::numbers::pi);
        samples.block<1, 3>(i, 0) = (sample->mTranslation - lastPose->mTranslation).transpose();
        samples.block<1, 3>(i, 3) = nextRotationVector.transpose();
    }

    // Minimize sample residuals plus lambda * sum((k! * coefficient_k)^2)
    // for k=2..4. Time is normalized by the history span, making the penalty
    // independent of FPS and epoch. Constant position and velocity are unpenalized.
    // Fit the intercept as well: forcing it through a noisy latest pose couples
    // that error into every derivative. Only f(dt)-f(0) is used below, preserving
    // the accepted pose as the exact prediction anchor.
    for (int k = 2, factorial = 2; k <= 4; ++k) {
        design(count + k - 2, k) = std::sqrt(derivativeRegularization) * factorial;
        factorial *= k + 1;
    }
    const auto fit = design.colPivHouseholderQr();
    if (fit.rank() != 5) return linear;
    const Eigen::Matrix<double, 5, 6> coefficients = fit.solve(samples);
    Eigen::Matrix<double, 6, 1> fittedIncrement = Eigen::Matrix<double, 6, 1>::Zero();
    const double normalizedElapsed = correctionTime / span;
    double power = normalizedElapsed;
    for (int k = 1; k <= 4; ++k) {
        fittedIncrement += coefficients.row(k).transpose() * power;
        power *= normalizedElapsed;
    }
    if (!fittedIncrement.allFinite()) return linear;

    Eigen::Matrix<double, 6, 1> linearStep;
    linearStep.head<3>() = lastPose->mTranslation - previousPose->mTranslation;
    const Eigen::AngleAxisd relative(previousPose->mRotation.transpose() * lastPose->mRotation);
    linearStep.tail<3>() = relative.axis() * relative.angle();
    Eigen::Matrix<double, 6, 1> correction = fittedIncrement - linearStep * (correctionTime / interval);
    // Bound translation and rotation separately, using relative motion scales
    // rather than absolute world units (monocular scale is arbitrary).
    for (int offset : {0, 3}) {
        const double limit = maximumCorrectionFraction * linearStep.segment<3>(offset).stableNorm() *
            (std::abs(correctionTime) / interval);
        const double magnitude = correction.segment<3>(offset).stableNorm();
        if (magnitude > limit) correction.segment<3>(offset) *= limit / magnitude;
    }
    correction *= correctionWeight;
    const Eigen::Vector3d rotationVector = linearStep.tail<3>() * (elapsed / interval) + correction.tail<3>();
    const double angle = rotationVector.stableNorm();
    const Eigen::Vector3d position = linear->mTranslation + correction.head<3>();
    if (!position.allFinite() || !std::isfinite(angle)) return linear;
    Eigen::Matrix3d rotation = lastPose->mRotation;
    if (angle > 0.0) rotation *= Eigen::AngleAxisd(angle, rotationVector / angle).toRotationMatrix();
    return std::make_shared<tft::RigidTransform3D>(lastPose->mSource, lastPose->mTarget, rotation, position);
}
