#include <gtest/gtest.h>

#include "random.hpp"
#include "schlam/Ransac.h"
#include "tft/rigid_transform_3d.hpp"

#include <random>

class ReconstructionTest : public ::testing::Test {
protected:
    const Eigen::Matrix3d intrinsics = (Eigen::Matrix3d() <<
        500, 0, 320, 0, 700, 240, 0, 0, 1).finished();
    const Eigen::Matrix3d rotation =
        (Eigen::AngleAxisd(0.03, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(-0.02, Eigen::Vector3d::UnitZ())).toRotationMatrix();
    const Eigen::Vector3d translation = Eigen::Vector3d(-0.98, 0.12, 0.15).normalized();
    std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matches;
    std::vector<Eigen::Vector3d> cloud;
    std::vector<bool> expectedMask;
    const std::mt19937 originalEngine = utils::randomEngine();
    const cv::RNG originalOpenCVEngine = cv::theRNG();

    void SetUp() override {
        utils::randomEngine().seed(utils::kRandomSeed);
        cv::theRNG() = cv::RNG(utils::kRandomSeed);
    }

    void TearDown() override {
        utils::randomEngine() = originalEngine;
        cv::theRNG() = originalOpenCVEngine;
    }

    void scene(size_t nearCount = 120, size_t farCount = 0, size_t outlierStride = 0) {
        std::mt19937 engine(42);
        std::uniform_real_distribution<double> x(-2, 2), y(-1.5, 1.5), z(4, 10);
        for (size_t i = 0; i < nearCount + farCount; ++i) {
            const Eigen::Vector3d point(x(engine), y(engine), i < nearCount ? z(engine) : 1000);
            cloud.push_back(point);
            const Eigen::Vector3d pixel1 = intrinsics * point;
            const Eigen::Vector3d pixel2 = intrinsics * (rotation * point + translation);
            const bool inlier = outlierStride == 0 || i % outlierStride != 2;
            matches[0].push_back(std::make_shared<KeyPoint>(
                pixel1.x() / pixel1.z(), pixel1.y() / pixel1.z()));
            matches[1].push_back(std::make_shared<KeyPoint>(
                pixel2.x() / pixel2.z() + (inlier ? 0 : 80),
                pixel2.y() / pixel2.z() + (inlier ? 0 : -50)));
            expectedMask.push_back(inlier && i < nearCount);
        }
    }

    auto normalizedPoints() const {
        const Eigen::Matrix3d inverseIntrinsics = intrinsics.inverse();
        std::array<std::vector<Eigen::Vector3d>, 2> points;
        for (size_t camera = 0; camera < points.size(); ++camera) {
            for (const auto& keypoint : matches[camera]) {
                points[camera].push_back(inverseIntrinsics *
                    Eigen::Vector3d(keypoint->getImgX(), keypoint->getImgY(), 1));
            }
        }
        return points;
    }

    void verify(const decltype(reconstructInitial(matches[0], matches[1], intrinsics))& result,
                double poseTolerance = 1e-6, double pointTolerance = 1e-5) {
        ASSERT_TRUE(result.has_value());
        const auto& [recoveredRotation, recoveredTranslation, recoveredCloud, mask] = *result;
        EXPECT_NEAR((recoveredRotation - rotation).norm(), 0, poseTolerance);
        EXPECT_NEAR((recoveredTranslation - translation).norm(), 0, poseTolerance);
        EXPECT_NEAR(recoveredTranslation.norm(), 1, 1e-6);
        ASSERT_EQ(recoveredCloud.size(), cloud.size());
        ASSERT_EQ(mask.size(), expectedMask.size());
        for (size_t i = 0; i < mask.size(); ++i) {
            SCOPED_TRACE(i);
            EXPECT_EQ(mask[i], expectedMask[i]);
            if (mask[i]) {
                EXPECT_NEAR((recoveredCloud[i] - cloud[i]).norm(), 0, pointTolerance);
            }
        }
    }
};

TEST_F(ReconstructionTest, RecoversPoseAndReferenceFrameCloudWithDistinctFocalLengths) {
    scene();
    verify(reconstructInitial(matches[0], matches[1], intrinsics));
}

TEST_F(ReconstructionTest, RejectsMismatchesWithoutChangingCloudAndMaskOrder) {
    scene(150, 0, 6);
    // A RANSAC minimal solution can have small residuals within the pixel threshold.
    // Allow that estimation error while requiring the exact correspondence mask.
    verify(reconstructInitial(matches[0], matches[1], intrinsics), 5e-3, 0.3);
}

TEST_F(ReconstructionTest, TriangulationReturnsCoordinatesInBothCameraFrames) {
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.topLeftCorner<3, 3>() = rotation;
    transform.topRightCorner<3, 1>() = translation;
    for (const Eigen::Vector3d& point : {
             Eigen::Vector3d(0.4, -0.6, 5), Eigen::Vector3d(-1, 0.2, 10),
             Eigen::Vector3d(0, 0, 1000), Eigen::Vector3d(-0.6, 0.3, -6)}) {
        const Eigen::Vector3d point2 = rotation * point + translation;
        const Eigen::Vector3d ray1 = point / point.z();
        const Eigen::Vector3d ray2 = point2 / point2.z();
        const auto [recovered1, recovered2] = triangulate(ray1, ray2, transform);
        EXPECT_NEAR((recovered1 - point).norm(), 0, 1e-7);
        EXPECT_NEAR((recovered2 - point2).norm(), 0, 1e-7);
    }
}

TEST_F(ReconstructionTest, KnownPoseTriangulationPreservesOrderAndRejectsMismatchesAndFarPoints) {
    scene(30, 10, 6);
    const auto transform = std::make_shared<tft::RigidTransform3D>(
        "camera1", "camera2", rotation, translation);
    const auto result = triangulate(matches[0], matches[1], intrinsics, transform);
    ASSERT_TRUE(result.has_value());
    const auto& [points, mask] = *result;
    ASSERT_EQ(points.size(), cloud.size());
    ASSERT_EQ(mask.size(), expectedMask.size());
    for (size_t i = 0; i < mask.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(mask[i], expectedMask[i]);
        if (mask[i]) EXPECT_NEAR((points[i] - cloud[i]).norm(), 0, 1e-7);
    }
}

TEST_F(ReconstructionTest, KnownPoseTriangulationWorksWithOneMatch) {
    scene(1);
    const auto transform = std::make_shared<tft::RigidTransform3D>(
        "camera1", "camera2", rotation, translation);
    const auto result = triangulate(matches[0], matches[1], intrinsics, transform);
    ASSERT_TRUE(result.has_value());
    const auto& [points, mask] = *result;
    ASSERT_EQ(points.size(), 1);
    EXPECT_EQ(mask, std::vector<bool>({true}));
    EXPECT_NEAR((points[0] - cloud[0]).norm(), 0, 1e-7);
}

TEST_F(ReconstructionTest, KnownPoseTriangulationFiltersEpipolarOutliersBeforeTriangulation) {
    const auto transform = std::make_shared<tft::RigidTransform3D>(
        "camera1", "camera2", Eigen::Matrix3d::Identity(), Eigen::Vector3d(-1, 0, 0));
    for (const double offset : {0.0, 2.5, 3.0}) {
        matches[0].push_back(std::make_shared<KeyPoint>(320, 240));
        matches[1].push_back(std::make_shared<KeyPoint>(220, 240 + offset));
    }
    const auto result = triangulate(matches[0], matches[1], intrinsics, transform);
    ASSERT_TRUE(result.has_value());
    const auto& [points, mask] = *result;
    ASSERT_EQ(points.size(), 3);
    EXPECT_EQ(mask, std::vector<bool>({true, true, false}));
    EXPECT_NEAR((points[0] - Eigen::Vector3d(0, 0, 5)).norm(), 0, 1e-7);
    EXPECT_TRUE(points[2].isZero(0));
}

TEST_F(ReconstructionTest, KnownPoseEpipolarFilteringIsIndependentOfBaselineScale) {
    scene(10);
    for (const double scale : {1e-6, 1000.0}) {
        SCOPED_TRACE(scale);
        const auto transform = std::make_shared<tft::RigidTransform3D>(
            "camera1", "camera2", rotation, scale * translation);
        const auto result = triangulate(matches[0], matches[1], intrinsics, transform);
        ASSERT_TRUE(result.has_value());
        const auto& [points, mask] = *result;
        ASSERT_EQ(points.size(), cloud.size());
        EXPECT_EQ(mask, expectedMask);
        for (size_t i = 0; i < points.size(); ++i) {
            EXPECT_NEAR((points[i] / scale - cloud[i]).norm(), 0, 1e-7);
        }
    }
}

TEST_F(ReconstructionTest, KnownPoseTriangulationRejectsPointsBehindEitherCamera) {
    const auto transform = std::make_shared<tft::RigidTransform3D>(
        "camera1", "camera2", Eigen::Matrix3d::Identity(), Eigen::Vector3d(-1, 0, -5));
    cloud = {{0.4, -0.6, -2}, {0.4, -0.6, 2}, {0.4, -0.6, 10}};
    for (const auto& point : cloud) {
        const Eigen::Vector3d pixel1 = intrinsics * point;
        const Eigen::Vector3d pixel2 = intrinsics * (point + transform->mTranslation);
        matches[0].push_back(std::make_shared<KeyPoint>(
            pixel1.x() / pixel1.z(), pixel1.y() / pixel1.z()));
        matches[1].push_back(std::make_shared<KeyPoint>(
            pixel2.x() / pixel2.z(), pixel2.y() / pixel2.z()));
    }
    const auto result = triangulate(matches[0], matches[1], intrinsics, transform);
    ASSERT_TRUE(result.has_value());
    const auto& [points, mask] = *result;
    ASSERT_EQ(points.size(), cloud.size());
    EXPECT_EQ(mask, std::vector<bool>({false, false, true}));
    EXPECT_NEAR((points[2] - cloud[2]).norm(), 0, 1e-7);
}

TEST_F(ReconstructionTest, KnownPoseTriangulationReturnsNoResultWithoutUsablePoints) {
    scene(0, 10);
    const auto transform = std::make_shared<tft::RigidTransform3D>(
        "camera1", "camera2", rotation, translation);
    EXPECT_FALSE(triangulate(matches[0], matches[1], intrinsics, transform).has_value());
    EXPECT_FALSE(triangulate(matches[0], matches[0], intrinsics,
        tft::RigidTransform3D::identity("camera1", "camera2")).has_value());
}

TEST_F(ReconstructionTest, KnownPoseTriangulationRejectsEmptyOrUnpairedInputsAndMissingPose) {
    const auto transform = std::make_shared<tft::RigidTransform3D>(
        "camera1", "camera2", rotation, translation);
    EXPECT_FALSE(triangulate(matches[0], matches[1], intrinsics, transform).has_value());
    scene(2);
    EXPECT_FALSE(triangulate(matches[0], matches[1], intrinsics, nullptr).has_value());
    matches[1].pop_back();
    EXPECT_FALSE(triangulate(matches[0], matches[1], intrinsics, transform).has_value());
}

TEST_F(ReconstructionTest, FarPointsSupportPoseWithoutEnteringTheMap) {
    scene(100, 20);
    Eigen::Matrix3d crossTranslation;
    crossTranslation << 0, -translation.z(), translation.y(),
        translation.z(), 0, -translation.x(), -translation.y(), translation.x(), 0;
    verify(recoverPoseFromEssential(crossTranslation * rotation, normalizedPoints(), matches,
        std::vector<bool>(cloud.size(), true), intrinsics));
}

TEST_F(ReconstructionTest, IdenticalViewsDoNotInitialize) {
    scene();
    EXPECT_FALSE(reconstructInitial(matches[0], matches[0], intrinsics).has_value());
}

TEST_F(ReconstructionTest, InsufficientMatchesDoNotInitialize) {
    EXPECT_FALSE(reconstructInitial(matches[0], matches[1], intrinsics).has_value());
    scene(7);
    EXPECT_FALSE(reconstructInitial(matches[0], matches[1], intrinsics).has_value());
}

TEST_F(ReconstructionTest, EmptyEssentialSearchClearsOutputs) {
    scene();
    Eigen::Matrix3d essential = Eigen::Matrix3d::Identity();
    std::vector<bool> inliers(cloud.size(), true);
    double score = 42;
    findEssential(normalizedPoints(), {}, inliers, score, essential, 1.0 / intrinsics(0, 0));
    EXPECT_DOUBLE_EQ(score, 0);
    EXPECT_TRUE(essential.isZero(0));
    EXPECT_EQ(inliers, std::vector<bool>(cloud.size(), false));
}
