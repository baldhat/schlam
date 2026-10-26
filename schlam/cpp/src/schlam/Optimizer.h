//
// Created by baldhat on 2/26/26.
//

#ifndef SCHLAM_OPTIMIZER_H
#define SCHLAM_OPTIMIZER_H

#include "KeyPoint.h"

#include <memory>
#include <tuple>
#include <vector>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/optimization_algorithm_property.h>

#include "KeyFrame.h"

class Map;

struct PoseOptimizationResult {
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    // Indexes the input observations. Rejected first-pass matches stay rejected.
    std::vector<bool> inliers;
};

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
    // The inlier mask is checked again against the final optimized pose.
    PoseOptimizationResult
    optimizePoseWithInliers(const std::vector<std::shared_ptr<KeyPoint>>& aKeyPoints,
                           const Eigen::Matrix3d& aRotation,
                           const Eigen::Vector3d& aTranslation,
                           const std::vector<Eigen::Vector3d>& aPoints,
                           const Eigen::Matrix3d& aIntrinsics);

    // Compatibility interface for callers that only need the optimized pose.
    std::tuple<Eigen::Matrix3d, Eigen::Vector3d>
    optimizePose(const std::vector<std::shared_ptr<KeyPoint>>& aKeyPoints,
                        const Eigen::Matrix3d &aRotation,
                        const Eigen::Vector3d &aTranslation,
                        const std::vector<Eigen::Vector3d> &aPoints,
                        const Eigen::Matrix3d &aIntrinsics);

    // Keeps initialization keyframe 0 fixed and preserves the baseline to keyframe 1.
    // Returns true when finite optimized poses and points were written to the map.
    // The final pass can also remove points with fewer than two inlier observations
    // or less than 80% inlier observations.
    bool optimizeFullMap(Map* map, bool removeOutlierPoints = false);

    // Bounds the local poses and fixed anchors; skips problems without a monocular scale anchor.
    bool optimizeCovisibilityGraph(Map* map, KeyFrame& keyframe,
                                  std::size_t maxNumVariable = 20, std::size_t maxNumFixed = 40);

    void stopCovisibility();

    bool stopCovisibility_{false};
};


#endif //SCHLAM_OPTIMIZER_H
