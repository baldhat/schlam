//
// Created by baldhat on 2/26/26.
//

#ifndef SCHLAM_FRAME_H
#define SCHLAM_FRAME_H

#include "KeyPoint.h"

#include <eigen3/Eigen/Core>
#include <opencv2/core/mat.hpp>

#include "QuadTreeNode.h"
#include "data/image_data.hpp"

class Frame {
public:
    Frame(const std::shared_ptr<ImageData> aImageData);

    utils::TTimestamp mTimestamp;
    cv::Mat mImage;
    Eigen::Matrix3d mIntrinsics;
    std::shared_ptr<QuadTreeNode> mKeypointTree;
    Eigen::Matrix3d mRotation;
    Eigen::Vector3d mPosition;
};


#endif //SCHLAM_FRAME_H
