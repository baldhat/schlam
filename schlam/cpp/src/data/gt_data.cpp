
#include "gt_data.hpp"

GTData::GTData(const utils::TTimestamp &aTimestamp,
                 const Eigen::Vector3f &aPosition,
                 const Eigen::Matrix3f &aRotation,
                 const Eigen::Vector3f &aTranslationalVelocity,
                 const std::string &aCF)
    : mTimestamp(aTimestamp), mPosition(aPosition), mRotation(aRotation),
      mTranslationalVelocity(aTranslationalVelocity), mCoordinateFrame(aCF) {}