//
// Created by baldhat on 2/26/26.
//

#ifndef SCHLAM_OPTIMIZER_H
#define SCHLAM_OPTIMIZER_H

#include "KeyPoint.h"

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/optimization_algorithm_property.h>

class Optimizer {
public:
    Optimizer();


    std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<Eigen::Vector3d> >
    optimize(const std::vector<std::vector<KeyPoint*>>& aKeyPoints,
             const Eigen::Matrix3d& aRotation,
             const Eigen::Vector3d& aTranslation,
             const std::vector<Eigen::Vector3d>& aPoints,
             const Eigen::Matrix3d& aIntrinsics);

};


#endif //SCHLAM_OPTIMIZER_H