#pragma once

#include "transform.hpp"

#include <string>
#include <memory>

#include <eigen3/Eigen/Core>

namespace tft {
/**
 * @brief A Transform object describes how to express data described in the
 * target coordinate frame in the source coordinate frame and reversed.
 */
class RigidTransform3D : public Transform {
public:
  RigidTransform3D(const std::string &aSource, const std::string &aTarget,
                   const Eigen::Matrix3d &aRotation,
                   const Eigen::Vector3d &aTranslation);

  ~RigidTransform3D() = default;

  Transformable3D apply(const Transformable3D &aTransformable) override;
  Transformable3D apply(const Transformable3D &&aTransformable) override;
  Transformable3D applyInverse(const Transformable3D &aTransformable) override;
  Transformable3D applyInverse(const Transformable3D &&aTransformable) override;
  std::shared_ptr<RigidTransform3D> inverse();

  static std::shared_ptr<RigidTransform3D> identity(const std::string& source="world", const std::string& target="world");

  const Eigen::Matrix3d rotation;
  const Eigen::Vector3d translation;

private:
};

std::shared_ptr<RigidTransform3D> operator*(const std::shared_ptr<RigidTransform3D> a, const std::shared_ptr<RigidTransform3D> b);

} // namespace tft