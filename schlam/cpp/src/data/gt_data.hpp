
#pragma once

#include "../utils.hpp"

#include <eigen3/Eigen/Core>

class GTData {
public:
  GTData(const utils::TTimestamp &aTimestamp,
          const Eigen::Vector3d &aPosition, 
          const Eigen::Matrix3d &aRotation,
          const Eigen::Vector3d &aTranslationalVelocity,
          const std::string &aCF);

  const utils::TTimestamp mTimestamp;
  const Eigen::Vector3d mPosition;
  const Eigen::Matrix3d mRotation;
  const Eigen::Vector3d mTranslationalVelocity;
  const std::string mCoordinateFrame;
};