//
// Created by baldhat on 10/11/26.
//

#include "MotionModel.h"
#include "tft/rigid_transform_3d.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>


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
    if (mPoses.size() > 2) mPoses.erase(mPoses.begin());
}

std::shared_ptr<tft::RigidTransform3D> MotionModel::estimateLinearAt(const utils::TTimestamp& timestamp) const {
    if (mPoses.empty()) {
        throw std::logic_error("Cannot estimate motion without a pose");
    }

    const auto& [lastPose, lastTimestamp] = mPoses.back();
    if (mPoses.size() == 1) {
        return lastPose;
    }

    const auto& [previousPose, previousTimestamp] = mPoses[mPoses.size() - 2];
    const double interval = std::chrono::duration<double>(lastTimestamp - previousTimestamp).count();
    const double elapsed = std::chrono::duration<double>(timestamp - lastTimestamp).count();
    const double factor = std::clamp(elapsed / interval,
                                     -mMaxExtrapolationIntervals, mMaxExtrapolationIntervals);

    const Eigen::Vector3d translation = lastPose->mTranslation +
        factor * (lastPose->mTranslation - previousPose->mTranslation);

    // Scale the shortest relative rotation, preserving SO(3).
    const Eigen::AngleAxisd relativeRotation(previousPose->mRotation.transpose() * lastPose->mRotation);
    const Eigen::Matrix3d rotation = lastPose->mRotation *
        Eigen::AngleAxisd(factor * relativeRotation.angle(), relativeRotation.axis()).toRotationMatrix();

    return std::make_shared<tft::RigidTransform3D>(lastPose->mSource, lastPose->mTarget, rotation, translation);
}
