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


    std::tuple<Eigen::Matrix3f, Eigen::Vector3f, std::vector<Eigen::Vector3f> >
    optimize(const std::vector<std::vector<KeyPoint>>& aKeyPoints,
             const Eigen::Matrix3f& aRotation,
             const Eigen::Vector3f& aTranslation,
             const std::vector<Eigen::Vector3f>& aPoints,
             const Eigen::Matrix3f& aIntrinsics);

};


#endif //SCHLAM_OPTIMIZER_H