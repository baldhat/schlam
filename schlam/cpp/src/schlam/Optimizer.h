//
// Created by baldhat on 2/26/26.
//

#ifndef SCHLAM_OPTIMIZER_H
#define SCHLAM_OPTIMIZER_H

#include "KeyPoint.h"

#include <memory>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/optimization_algorithm_property.h>

class Optimizer {
public:
    Optimizer();

    // Anchors monocular scale to the initial translation's baseline length.
    std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<Eigen::Vector3d> >
    optimizePoseAndPoints(const std::vector<std::vector<std::shared_ptr<KeyPoint>>>& aKeyPoints,
             const Eigen::Matrix3d& aRotation,
             const Eigen::Vector3d& aTranslation,
             const std::vector<Eigen::Vector3d>& aPoints,
             const Eigen::Matrix3d& aIntrinsics);

    // The input and returned pose map aPoints' coordinate frame into the camera.
    // Observation uncertainty assumes a 1.2x pyramid and one-pixel sigma at level 0.
    std::tuple<Eigen::Matrix3d, Eigen::Vector3d>
    optimizePose(const std::vector<std::shared_ptr<KeyPoint>>& aKeyPoints,
                        const Eigen::Matrix3d &aRotation,
                        const Eigen::Vector3d &aTranslation,
                        const std::vector<Eigen::Vector3d> &aPoints,
                        const Eigen::Matrix3d &aIntrinsics);

};


#endif //SCHLAM_OPTIMIZER_H
