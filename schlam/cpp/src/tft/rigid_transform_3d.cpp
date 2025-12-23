
#include "rigid_transform_3d.hpp"
#include <memory>
#include <stdexcept>

namespace tft {
RigidTransform3D::RigidTransform3D(const std::string &aSource,
                                   const std::string &aTarget,
                                   const Eigen::Matrix3d &aRotation,
                                   const Eigen::Vector3d &aTranslation)
    : Transform(aSource, aTarget), rotation(aRotation),
      translation(aTranslation) {}

Transformable3D RigidTransform3D::apply(const Transformable3D &aTransformable) {
  if (aTransformable.getCF() != source) {
    throw std::runtime_error(
        "Cannot apply transform from CF '" + source + "' to CF '" + target +
        "' to Transformable3D in CF '" + aTransformable.getCF() + "'!");
  }
  return Transformable3D(rotation * aTransformable.vector + translation,
                         target);
}

Transformable3D
RigidTransform3D::apply(const Transformable3D &&aTransformable) {
  if (aTransformable.getCF() != source) {
    throw std::runtime_error(
        "Cannot apply transform from CF '" + source + "' to CF '" + target +
        "' to Transformable3D in CF '" + aTransformable.getCF() + "'!");
  }
  return Transformable3D(rotation * aTransformable.vector + translation,
                         target);
}

Transformable3D
RigidTransform3D::applyInverse(const Transformable3D &aTransformable) {
  if (aTransformable.getCF() != target) {
    throw std::runtime_error(
        "Cannot apply transform from CF '" + target + "' to CF '" + source +
        "' to Transformable3D in CF '" + aTransformable.getCF() + "'!");
  }
  return Transformable3D(
      rotation.transpose() * (aTransformable.vector - translation), source);
}

Transformable3D
RigidTransform3D::applyInverse(const Transformable3D &&aTransformable) {
  if (aTransformable.getCF() != target) {
    throw std::runtime_error(
        "Cannot apply transform from CF '" + target + "' to CF '" + source +
        "' to Transformable3D in CF '" + aTransformable.getCF() + "'!");
  }
  return Transformable3D(
      rotation.transpose() * (aTransformable.vector - translation), source);
}

std::shared_ptr<RigidTransform3D> RigidTransform3D::inverse() {
  return std::make_shared<RigidTransform3D>(
      target, source, rotation.transpose(),
      -rotation.transpose() * translation);
}

std::shared_ptr<RigidTransform3D> RigidTransform3D::identity(const std::string& source, const std::string& target) {
    return std::make_shared<RigidTransform3D>(
      source, target, Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 0));
}

std::shared_ptr<RigidTransform3D>
operator*(const std::shared_ptr<RigidTransform3D> a,
          const std::shared_ptr<RigidTransform3D> b) {
  return std::make_shared<RigidTransform3D>(
      b->source, a->target, a->rotation * b->rotation,
      a->rotation * b->translation + a->translation);
}
} // namespace tft