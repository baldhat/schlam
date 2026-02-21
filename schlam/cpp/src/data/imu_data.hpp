
#pragma once

#include "../utils.hpp"

#include <eigen3/Eigen/Core>

class IMUData {
public:
  IMUData(const utils::TTimestamp &aTimestamp,
          const Eigen::Vector3f &aAcceleration,
          const Eigen::Vector3f &aAngularVelocity, const std::string &aCF,
          const double mGyroscopeNoiseDensity,
          const double mGyroscopeRandomWalk,
          const double mAccelerometerNoiseDensity,
          const double mAccelerometerRandomWalk);

  const utils::TTimestamp mTimestamp;
  const Eigen::Vector3f mAcceleration;
  const Eigen::Vector3f mRotationalVelocity;
  const std::string mCoordinateFrame;
  const double mGyroscopeNoiseDensity;
  const double mGyroscopeRandomWalk;
  const double mAccelerometerNoiseDensity;
  const double mAccelerometerRandomWalk;
};
