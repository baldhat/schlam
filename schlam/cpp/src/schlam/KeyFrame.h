//
// Created by baldhat on 10/12/26.
//

#ifndef SCHLAM_KEYFRAME_H
#define SCHLAM_KEYFRAME_H
#include <vector>

#include "Frame.h"
#include "tft/rigid_transform_3d.hpp"


class KeyFrame {
public:
    KeyFrame(
      const KeyFrameId id,
      const Frame& frame);

    const KeyFrameId mId;
    const utils::TTimestamp mTimestamp;
    const Eigen::Matrix3d mIntrinsics;
    std::shared_ptr<QuadTreeNode> mKeypointTree;
    const std::vector<std::shared_ptr<KeyPoint>> mKeyPoints;
    std::shared_ptr<tft::RigidTransform3D> mMapToCamera;
    cv::Mat mImageData;
};


#endif //SCHLAM_KEYFRAME_H
