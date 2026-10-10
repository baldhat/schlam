
#include "gt_data.hpp"

GTData::GTData(const utils::TTimestamp &aTimestamp,
                 const Eigen::Vector3d &aPosition,
                 const Eigen::Matrix3d &aRotation,
                 const Eigen::Vector3d &aTranslationalVelocity,
                 const std::string &aCF)
    : mTimestamp(aTimestamp), mPosition(aPosition), mRotation(aRotation),
      mTranslationalVelocity(aTranslationalVelocity), mCoordinateFrame(aCF) {}