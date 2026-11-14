//
// Created by baldhat on 10/11/26.
//

#ifndef SCHLAM_MOTIONMODEL_H
#define SCHLAM_MOTIONMODEL_H

#include <tuple>
#include <vector>
#include <memory>
#include <chrono>
#include <optional>

#include "utils.hpp"

namespace tft {
    class RigidTransform3D;
}

class MotionModel {
public:
    // Limit predictions to this many intervals between the last two accepted poses.
    explicit MotionModel(double maxExtrapolationIntervals = 2.0);

    // Add chronological poses expressed in a common target coordinate frame.
    void addPose(const std::shared_ptr<tft::RigidTransform3D> transform, const utils::TTimestamp& timestamp);
    // Extrapolate the latest two poses within the configured horizon; with one pose, assume no motion.
    // An explicit time horizon overrides the interval limit during recovery.
    std::shared_ptr<tft::RigidTransform3D> estimateLinearAt(
        const utils::TTimestamp& timestamp,
        std::optional<std::chrono::duration<double>> maximumExtrapolation = std::nullopt) const;

    // Smooth up to twelve accepted poses with a regularized fourth-order fit.
    // Use linear prediction until eight samples are available. Limit corrections
    // relative to linear motion and fade them out over two frame intervals.
    // Rotations use a local SO(3) log/exp chart. Queries never advance history.
    std::shared_ptr<tft::RigidTransform3D> estimateFourthOrderAt(
        const utils::TTimestamp& timestamp,
        std::optional<std::chrono::duration<double>> maximumExtrapolation = std::nullopt) const;

private:
    double mMaxExtrapolationIntervals;
    std::vector<std::tuple<std::shared_ptr<tft::RigidTransform3D>, utils::TTimestamp>> mPoses;
};


#endif //SCHLAM_MOTIONMODEL_H
