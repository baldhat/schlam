#include "rigid_transform_3d.hpp"
#include <memory>
#include <stdexcept>

namespace tft {
    RigidTransform3D::RigidTransform3D(const std::string &aSource,
                                       const std::string &aTarget,
                                       const Eigen::Matrix3d &aRotation,
                                       const Eigen::Vector3d &aTranslation)
        : mSource(aSource), mTarget(aTarget), mRotation(aRotation),
          mTranslation(aTranslation) {
    }

    Transformable3D RigidTransform3D::apply(const Transformable3D &aTransformable) {
        if (aTransformable.getCF() != mSource) {
            throw std::runtime_error(
                "Cannot apply transform from CF '" + mSource + "' to CF '" + mTarget +
                "' to Transformable3D in CF '" + aTransformable.getCF() + "'!");
        }
        return Transformable3D(mRotation * aTransformable.vector + mTranslation,
                               mTarget);
    }

    Transformable3D
    RigidTransform3D::apply(const Transformable3D &&aTransformable) {
        if (aTransformable.getCF() != mSource) {
            throw std::runtime_error(
                "Cannot apply transform from CF '" + mSource + "' to CF '" + mTarget +
                "' to Transformable3D in CF '" + aTransformable.getCF() + "'!");
        }
        return Transformable3D(mRotation * aTransformable.vector + mTranslation,
                               mTarget);
    }

    Transformable3D
    RigidTransform3D::applyInverse(const Transformable3D &aTransformable) {
        if (aTransformable.getCF() != mTarget) {
            throw std::runtime_error(
                "Cannot apply transform from CF '" + mTarget + "' to CF '" + mSource +
                "' to Transformable3D in CF '" + aTransformable.getCF() + "'!");
        }
        return Transformable3D(
            mRotation.transpose() * (aTransformable.vector - mTranslation), mSource);
    }

    Transformable3D
    RigidTransform3D::applyInverse(const Transformable3D &&aTransformable) {
        if (aTransformable.getCF() != mTarget) {
            throw std::runtime_error(
                "Cannot apply transform from CF '" + mTarget + "' to CF '" + mSource +
                "' to Transformable3D in CF '" + aTransformable.getCF() + "'!");
        }
        return Transformable3D(
            mRotation.transpose() * (aTransformable.vector - mTranslation), mSource);
    }

    std::shared_ptr<RigidTransform3D> RigidTransform3D::inverse() const {
        return std::make_shared<RigidTransform3D>(
            mTarget, mSource, mRotation.transpose(),
            -mRotation.transpose() * mTranslation);
    }

    std::shared_ptr<RigidTransform3D>
    RigidTransform3D::identity(const std::string &aSource, const std::string &aTarget) {
        return std::make_shared<RigidTransform3D>(
            aSource, aTarget, Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 0));
    }

    Eigen::Matrix4d RigidTransform3D::matrix() const {
        Eigen::Matrix4d m;
        auto& r = mRotation;
        auto& t = mTranslation;
        m << r(0, 0), r(0, 1), r(0, 2), t(0),
             r(1, 0), r(1, 1), r(1, 2), t(1),
             r(2, 0), r(2, 1), r(2, 2), t(2),
             0, 0, 0, 1;
        return m;
    }

std::shared_ptr<RigidTransform3D>
operator*(const std::shared_ptr<RigidTransform3D> a,
          const std::shared_ptr<RigidTransform3D> b) {
  return std::make_shared<RigidTransform3D>(
      b->mSource, a->mTarget, a->mRotation * b->mRotation,
      a->mRotation * b->mTranslation + a->mTranslation);
}
} // namespace tft
