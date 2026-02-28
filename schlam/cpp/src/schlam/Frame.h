//
// Created by baldhat on 2/26/26.
//

#ifndef SCHLAM_FRAME_H
#define SCHLAM_FRAME_H

#include "KeyPoint.h"

#include <eigen3/Eigen/Core>

struct Frame {
    std::vector<KeyPoint> mKeyPoints;
    Eigen::Matrix3f mRotation;
    Eigen::Vector3f mPosition;
};


#endif //SCHLAM_FRAME_H