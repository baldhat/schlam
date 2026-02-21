
#pragma once

#include "../utils.hpp"

#include <eigen3/Eigen/Core>

class GTData {
public:
  GTData(const utils::TTimestamp &aTimestamp,
          const Eigen::Vector3f &aPosition,
          const Eigen::Matrix3f &aRotation,
          const Eigen::Vector3f &aTranslationalVelocity,
          const std::string &aCF);

  const utils::TTimestamp mTimestamp;
  const Eigen::Vector3f mPosition;
  const Eigen::Matrix3f mRotation;
  const Eigen::Vector3f mTranslationalVelocity;
  const std::string mCoordinateFrame;
};