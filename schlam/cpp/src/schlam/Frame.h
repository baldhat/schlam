//
// Created by baldhat on 2/26/26.
//

#ifndef SCHLAM_FRAME_H
#define SCHLAM_FRAME_H

#include "KeyPoint.h"

#include <eigen3/Eigen/Core>
#include <opencv2/core/mat.hpp>

#include "QuadTreeNode.h"

class ImageData;

class Frame {
public:
    Frame(const std::shared_ptr<ImageData> aImageData);

    cv::Mat mImage;
    Eigen::Matrix3d mIntrinsics;
    std::unique_ptr<QuadTreeNode> mKeypointTree;
    Eigen::Matrix3d mRotation;
    Eigen::Vector3d mPosition;
};


#endif //SCHLAM_FRAME_H