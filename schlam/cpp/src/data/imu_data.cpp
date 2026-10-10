
#include "imu_data.hpp"

IMUData::IMUData(const utils::TTimestamp &aTimestamp,
                 const Eigen::Vector3d &aAcceleration,
                 const Eigen::Vector3d &aRotationalVelocity,
                 const std::string &aCF, const double aGyroscopeNoiseDensity,
                 const double aGyroscopeRandomWalk,
                 const double aAccelerometerNoiseDensity,
                 const double aAccelerometerRandomWalk)
    : mTimestamp(aTimestamp), mAcceleration(aAcceleration),
      mRotationalVelocity(aRotationalVelocity), mCoordinateFrame(aCF),
      mGyroscopeNoiseDensity(aGyroscopeNoiseDensity),
      mGyroscopeRandomWalk(aGyroscopeRandomWalk),
      mAccelerometerNoiseDensity(aAccelerometerNoiseDensity),
      mAccelerometerRandomWalk(aAccelerometerRandomWalk) {}