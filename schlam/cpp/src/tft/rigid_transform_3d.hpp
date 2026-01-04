#pragma once

#include "transformable_3d.hpp"

#include <string>
#include <memory>

#include <eigen3/Eigen/Core>

namespace tft {
/**
 * @brief A Transform object describes how to express data described in the
 * target coordinate frame in the source coordinate frame and reversed.
 */
class RigidTransform3D {
public:
    RigidTransform3D(const std::string &aSource, const std::string &aTarget,
                     const Eigen::Matrix3d &aRotation,
                     const Eigen::Vector3d &aTranslation);

    ~RigidTransform3D() = default;

    Transformable3D apply(const Transformable3D &aTransformable);
    Transformable3D apply(const Transformable3D &&aTransformable);
    Transformable3D applyInverse(const Transformable3D &aTransformable);
    Transformable3D applyInverse(const Transformable3D &&aTransformable);
    std::shared_ptr<RigidTransform3D> inverse() const;
    Eigen::Matrix4d matrix() const;

    static std::shared_ptr<RigidTransform3D> identity(const std::string& source="world", const std::string& target="world");

    const Eigen::Matrix3d mRotation;
    const Eigen::Vector3d mTranslation;
    const std::string mSource;
    const std::string mTarget;

private:
};

std::shared_ptr<RigidTransform3D> operator*(const std::shared_ptr<RigidTransform3D> a, const std::shared_ptr<RigidTransform3D> b);

} // namespace tft