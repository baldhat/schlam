//
// Created by baldhat on 10/11/26.
//

#ifndef SCHLAM_MOTIONMODEL_H
#define SCHLAM_MOTIONMODEL_H

#include <tuple>
#include <vector>

#include <Eigen/Core>
#include "utils.hpp"
#include "tft/rigid_transform_3d.hpp"


class MotionModel {
public:
    // Add chronological poses expressed in a common target coordinate frame.
    void addPose(const std::shared_ptr<tft::RigidTransform3D> transform, const utils::TTimestamp& timestamp);
    // Extrapolate the latest two poses; with one pose, assume no motion.
    std::shared_ptr<tft::RigidTransform3D>  estimateLinearAt(const utils::TTimestamp& timestamp);

private:
    std::vector<std::tuple<std::shared_ptr<tft::RigidTransform3D>, utils::TTimestamp> > mPoses;
};


#endif //SCHLAM_MOTIONMODEL_H
