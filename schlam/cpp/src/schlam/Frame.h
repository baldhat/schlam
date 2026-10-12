//
// Created by baldhat on 2/26/26.
//

#ifndef SCHLAM_FRAME_H
#define SCHLAM_FRAME_H

#include "KeyPoint.h"

#include <eigen3/Eigen/Core>
#include <memory>
#include <opencv2/core/mat.hpp>

#include "QuadTreeNode.h"
#include "data/image_data.hpp"
#include "tft/rigid_transform_3d.hpp"

class Frame {
public:
    Frame(const std::shared_ptr<ImageData> aImageData);

    const tft::RigidTransform3D& getMapToCamera() const;

    utils::TTimestamp mTimestamp;
    cv::Mat mImage;
    Eigen::Matrix3d mIntrinsics;
    std::shared_ptr<QuadTreeNode> mKeypointTree;
    // Accepted pose from the map coordinate frame into this camera; null until tracked.
    std::shared_ptr<tft::RigidTransform3D> mMapToCamera;
};


#endif //SCHLAM_FRAME_H
