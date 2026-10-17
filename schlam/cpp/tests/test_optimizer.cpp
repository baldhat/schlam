#include "schlam/Optimizer.h"
#include "schlam/Map.h"
#include "tft/transformer.hpp"

#include <gtest/gtest.h>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <stdexcept>

class PoseOptimizerTest : public ::testing::Test {
protected:
    Optimizer optimizer;
    Eigen::Matrix3d intrinsics;
    Eigen::Matrix3d rotation =
        (Eigen::AngleAxisd(0.17, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(-0.08, Eigen::Vector3d::UnitX())).toRotationMatrix();
    Eigen::Vector3d translation{0.3, -0.2, 0.4};
    std::vector<Eigen::Vector3d> mapPoints;
    std::vector<std::shared_ptr<KeyPoint>> observations;

    void SetUp() override {
        intrinsics << 520, 0, 320, 0, 410, 240, 0, 0, 1;
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 5; ++x) {
                const uint32_t u = 120 + 75 * x;
                const uint32_t v = 90 + 60 * y;
                const double depth = 4.0 + 0.4 * (x + 2 * y);
                const Eigen::Vector3d cameraPoint(
                    (u - intrinsics(0, 2)) * depth / intrinsics(0, 0),
                    (v - intrinsics(1, 2)) * depth / intrinsics(1, 1), depth);
                mapPoints.push_back(rotation.transpose() * (cameraPoint - translation));
                observations.push_back(std::make_shared<KeyPoint>(u, v));
            }
        }
    }

    Eigen::Matrix3d initialRotation() const {
        return rotation * Eigen::AngleAxisd(0.06, Eigen::Vector3d::UnitY()).toRotationMatrix();
    }

    Eigen::Vector3d initialTranslation() const {
        return translation + Eigen::Vector3d(-0.12, 0.1, 0.08);
    }
};

TEST_F(PoseOptimizerTest, RecoversPoseWithDistinctFocalLengthsAndPreservesMapPoints) {
    const auto originalPoints = mapPoints;
    const auto [optimizedRotation, optimizedTranslation] = optimizer.optimizePose(
        observations, initialRotation(), initialTranslation(), mapPoints, intrinsics);

    EXPECT_TRUE(optimizedRotation.isApprox(rotation, 1e-7));
    EXPECT_TRUE(optimizedTranslation.isApprox(translation, 1e-6));
    ASSERT_EQ(mapPoints.size(), originalPoints.size());
    for (std::size_t j = 0; j < mapPoints.size(); ++j) {
        EXPECT_TRUE(mapPoints[j].isApprox(originalPoints[j], 0));
    }
}

TEST_F(PoseOptimizerTest, MapToCameraResultConvertsBackToCameraToWorld) {
    auto transformer = std::make_shared<tft::Transformer>();
    const auto mapToWorld = std::make_shared<tft::RigidTransform3D>("map", "world",
        Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitZ()).toRotationMatrix(),
        Eigen::Vector3d(3, -2, 1));
    const auto initialMapToCamera = std::make_shared<tft::RigidTransform3D>("map", "camera",
        initialRotation(), initialTranslation());
    const auto predictedCameraToWorld = mapToWorld * initialMapToCamera->inverse();
    transformer->registerTransform(mapToWorld);
    transformer->registerTransform(predictedCameraToWorld);

    const auto mapToCamera = transformer->findTransform("map", "camera");
    const auto [optimizedRotation, optimizedTranslation] = optimizer.optimizePose(
        observations, mapToCamera->mRotation, mapToCamera->mTranslation, mapPoints, intrinsics);
    const auto optimizedMapToCamera = std::make_shared<tft::RigidTransform3D>("map", "camera",
        optimizedRotation, optimizedTranslation);
    const auto optimizedCameraToWorld = mapToWorld * optimizedMapToCamera->inverse();
    const auto trueMapToCamera = std::make_shared<tft::RigidTransform3D>("map", "camera",
        rotation, translation);
    const auto expectedCameraToWorld = mapToWorld * trueMapToCamera->inverse();

    EXPECT_EQ(optimizedCameraToWorld->mSource, "camera");
    EXPECT_EQ(optimizedCameraToWorld->mTarget, "world");
    EXPECT_TRUE(optimizedCameraToWorld->matrix().isApprox(expectedCameraToWorld->matrix(), 1e-6));
    transformer->registerTransform(optimizedCameraToWorld);
    EXPECT_TRUE(transformer->findTransform("camera", "world")->matrix().isApprox(
        expectedCameraToWorld->matrix(), 1e-6));
}

TEST_F(PoseOptimizerTest, ExactInitialPoseRemainsUnchanged) {
    const auto [optimizedRotation, optimizedTranslation] = optimizer.optimizePose(
        observations, rotation, translation, mapPoints, intrinsics);
    EXPECT_TRUE(optimizedRotation.isApprox(rotation, 1e-10));
    EXPECT_TRUE(optimizedTranslation.isApprox(translation, 1e-10));
}

TEST_F(PoseOptimizerTest, RejectsMismatchedObservationAndRefinesInliers) {
    const auto& observation = observations.back();
    observations.back() = std::make_shared<KeyPoint>(
        observation->getImgX() + 80, observation->getImgY() - 40);

    const auto [optimizedRotation, optimizedTranslation] = optimizer.optimizePose(
        observations, initialRotation(), initialTranslation(), mapPoints, intrinsics);

    EXPECT_TRUE(optimizedRotation.isApprox(rotation, 1e-7));
    EXPECT_TRUE(optimizedTranslation.isApprox(translation, 1e-6));
}

TEST_F(PoseOptimizerTest, FineScaleObservationsDominateNoisyCoarseScaleObservations) {
    const auto preciseCount = observations.size();
    const Eigen::Vector3d biasedTranslation = translation + Eigen::Vector3d(0.008, 0, 0);
    for (std::size_t j = 0; j < preciseCount; ++j) {
        const auto& preciseObservation = observations[j];
        const double depth = (rotation * mapPoints[j] + translation).z() + 0.2;
        const Eigen::Vector3d cameraPoint(
            (preciseObservation->getImgX() + 15 - intrinsics(0, 2)) * depth / intrinsics(0, 0),
            (preciseObservation->getImgY() + 10 - intrinsics(1, 2)) * depth / intrinsics(1, 1), depth);
        const Eigen::Vector3d point = rotation.transpose() * (cameraPoint - translation);
        const Eigen::Vector3d biasedProjection = intrinsics * (rotation * point + biasedTranslation);
        mapPoints.push_back(point);
        observations.push_back(std::make_shared<KeyPoint>(
            biasedProjection.x() / biasedProjection.z(), biasedProjection.y() / biasedProjection.z(), 7));
    }

    const auto [optimizedRotation, optimizedTranslation] = optimizer.optimizePose(
        observations, initialRotation(), initialTranslation(), mapPoints, intrinsics);

    // Coarse measurements have a systematic bias, but the precise measurements
    // should still determine the pose to substantially better than a pixel.
    double totalPreciseError = 0;
    for (std::size_t j = 0; j < preciseCount; ++j) {
        const Eigen::Vector3d projection = intrinsics * (optimizedRotation * mapPoints[j] + optimizedTranslation);
        const Eigen::Vector2d measurement(observations[j]->getImgX(), observations[j]->getImgY());
        totalPreciseError += (projection.head<2>() / projection.z() - measurement).norm();
    }
    EXPECT_LT(totalPreciseError / preciseCount, 0.15);
}

TEST_F(PoseOptimizerTest, RetainsNoisyCoarseObservationsWhileRejectingGrossMismatch) {
    const auto originalPoints = mapPoints;
    const auto originalObservations = observations;
    mapPoints.clear();
    observations.clear();
    for (std::size_t j = 0; j < originalPoints.size(); ++j) {
        // Symmetric independent measurements have their least-squares optimum
        // at the true projection, despite each being three pixels away from it.
        for (double offset : {-3.0, 3.0}) {
            mapPoints.push_back(originalPoints[j]);
            observations.push_back(std::make_shared<KeyPoint>(
                originalObservations[j]->getImgX() + offset, originalObservations[j]->getImgY(), 4));
        }
    }
    mapPoints.push_back(originalPoints.front());
    observations.push_back(std::make_shared<KeyPoint>(
        originalObservations.front()->getImgX() + 80, originalObservations.front()->getImgY() - 40, 4));

    const auto [optimizedRotation, optimizedTranslation] = optimizer.optimizePose(
        observations, initialRotation(), initialTranslation(), mapPoints, intrinsics);

    EXPECT_TRUE(optimizedRotation.isApprox(rotation, 1e-7));
    EXPECT_TRUE(optimizedTranslation.isApprox(translation, 1e-6));
}

TEST_F(PoseOptimizerTest, TooFewInliersSkipSecondPass) {
    observations.resize(2);
    mapPoints.resize(2);

    const auto [optimizedRotation, optimizedTranslation] = optimizer.optimizePose(
        observations, rotation, translation, mapPoints, intrinsics);

    EXPECT_TRUE(optimizedRotation.isApprox(rotation, 1e-10));
    EXPECT_TRUE(optimizedTranslation.isApprox(translation, 1e-10));
}

TEST_F(PoseOptimizerTest, AllPointsBehindCameraSkipSecondPass) {
    for (auto& point : mapPoints) {
        const Eigen::Vector3d cameraPoint = rotation * point + translation;
        point = rotation.transpose() * (-cameraPoint - translation);
    }

    const auto [optimizedRotation, optimizedTranslation] = optimizer.optimizePose(
        observations, rotation, translation, mapPoints, intrinsics);

    EXPECT_TRUE(optimizedRotation.isApprox(rotation, 1e-10));
    EXPECT_TRUE(optimizedTranslation.isApprox(translation, 1e-10));
}

TEST_F(PoseOptimizerTest, EmptyMatchesPreserveInitialPose) {
    const auto [optimizedRotation, optimizedTranslation] = optimizer.optimizePose(
        {}, initialRotation(), initialTranslation(), {}, intrinsics);
    EXPECT_TRUE(optimizedRotation.isApprox(initialRotation(), 0));
    EXPECT_TRUE(optimizedTranslation.isApprox(initialTranslation(), 0));
}

TEST_F(PoseOptimizerTest, MismatchedCorrespondenceCountsAreRejected) {
    observations.pop_back();
    EXPECT_THROW(optimizer.optimizePose(observations, rotation, translation, mapPoints, intrinsics),
        std::invalid_argument);
}

TEST_F(PoseOptimizerTest, NullObservationIsRejected) {
    observations[0].reset();
    EXPECT_THROW(optimizer.optimizePose(observations, rotation, translation, mapPoints, intrinsics),
        std::invalid_argument);
}

class JointOptimizerScaleTest : public ::testing::Test {
protected:
    Optimizer optimizer;
    Eigen::Matrix3d intrinsics;
    const Eigen::Vector3d trueTranslation{-2, 0, 0};
    std::vector<Eigen::Vector3d> mapPoints;
    std::vector<std::vector<std::shared_ptr<KeyPoint>>> observations{2};

    void SetUp() override {
        intrinsics << 400, 0, 320, 0, 400, 240, 0, 0, 1;
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 6; ++x) {
                const uint32_t u = 180 + 55 * x;
                const uint32_t v = 150 + 45 * y;
                const uint32_t disparity = 40 + 5 * ((x * x + 3 * y) % 9);
                const double depth = 800.0 / disparity;
                mapPoints.emplace_back((u - 320.0) * depth / 400.0,
                    (v - 240.0) * depth / 400.0, depth);
                observations[0].push_back(std::make_shared<KeyPoint>(u, v));
                observations[1].push_back(std::make_shared<KeyPoint>(u - disparity, v));
            }
        }
    }
};

TEST_F(JointOptimizerScaleTest, PreservesInitialBaselineAcrossScalesWhileRefiningPoseAndPoints) {
    const Eigen::Matrix3d initialRotation =
        Eigen::AngleAxisd(0.025, Eigen::Vector3d::UnitY()).toRotationMatrix();
    for (double inputScale : {0.5, 1.0, 3.0}) {
        SCOPED_TRACE(inputScale);
        const Eigen::Vector3d initialTranslation = inputScale * Eigen::Vector3d(-2, 0.12, 0.08);
        auto initialPoints = mapPoints;
        for (auto& point : initialPoints) {
            point *= 1.3 * inputScale;
        }
        const auto [rotation, translation, points] = optimizer.optimizePoseAndPoints(
            observations, initialRotation, initialTranslation, initialPoints, intrinsics);

        EXPECT_NEAR(translation.norm(), initialTranslation.norm(), 1e-6 * inputScale);
        EXPECT_TRUE(rotation.isApprox(Eigen::Matrix3d::Identity(), 1e-5));
        EXPECT_TRUE(translation.normalized().isApprox(trueTranslation.normalized(), 1e-5));
        ASSERT_EQ(points.size(), mapPoints.size());
        const double expectedScale = initialTranslation.norm() / trueTranslation.norm();
        for (std::size_t j = 0; j < points.size(); ++j) {
            EXPECT_TRUE(points[j].isApprox(expectedScale * mapPoints[j], 1e-5)) << "Point " << j;
        }
    }
}

TEST_F(JointOptimizerScaleTest, RecoversPoseAndPointsWithDistinctFocalLengths) {
    const Eigen::Matrix3d initialRotation =
        Eigen::AngleAxisd(0.025, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Vector3d initialTranslation(-2, 0.12, 0.08);
    const double expectedScale = initialTranslation.norm() / trueTranslation.norm();

    for (double fy : {250.0, 650.0}) {
        SCOPED_TRACE(fy);
        intrinsics(1, 1) = fy;
        auto expectedPoints = mapPoints;
        auto initialPoints = mapPoints;
        for (std::size_t j = 0; j < mapPoints.size(); ++j) {
            // Keep the integer observations exact for the new vertical focal length.
            expectedPoints[j].y() *= 400.0 / fy;
            initialPoints[j] = 1.3 * expectedPoints[j];
        }

        const auto [rotation, translation, points] = optimizer.optimizePoseAndPoints(
            observations, initialRotation, initialTranslation, initialPoints, intrinsics);

        EXPECT_NEAR(translation.norm(), initialTranslation.norm(), 1e-6);
        EXPECT_TRUE(rotation.isApprox(Eigen::Matrix3d::Identity(), 1e-5));
        EXPECT_TRUE(translation.normalized().isApprox(trueTranslation.normalized(), 1e-5));
        ASSERT_EQ(points.size(), expectedPoints.size());
        for (std::size_t j = 0; j < points.size(); ++j) {
            EXPECT_TRUE(points[j].isApprox(expectedScale * expectedPoints[j], 1e-5)) << "Point " << j;
        }
    }
}

TEST_F(JointOptimizerScaleTest, UsesEachFramesOctaveForNoisyObservations) {
    const std::size_t noisyPoint = 7;
    for (std::size_t noisyFrame : {0, 1}) {
        SCOPED_TRACE(noisyFrame);
        auto mixedObservations = observations;
        const auto& observation = observations[noisyFrame][noisyPoint];
        mixedObservations[noisyFrame][noisyPoint] = std::make_shared<KeyPoint>(
            observation->getImgX(), observation->getImgY() + 3, 7);
        const auto [rotation, translation, points] = optimizer.optimizePoseAndPoints(
            mixedObservations, Eigen::Matrix3d::Identity(), trueTranslation, mapPoints, intrinsics);

        EXPECT_NEAR(translation.norm(), trueTranslation.norm(), 1e-6);
        for (std::size_t frame : {0, 1}) {
            const Eigen::Vector3d cameraPoint = frame == 0
                ? points[noisyPoint] : Eigen::Vector3d(rotation * points[noisyPoint] + translation);
            const Eigen::Vector3d projection = intrinsics * cameraPoint;
            const Eigen::Vector2d measurement(mixedObservations[frame][noisyPoint]->getImgX(),
                mixedObservations[frame][noisyPoint]->getImgY());
            const double error = (projection.head<2>() / projection.z() - measurement).norm();
            if (frame == noisyFrame) {
                EXPECT_GT(error, 1.5);
            } else {
                // The precise observation should dominate the reconstructed point.
                EXPECT_LT(error, 0.5);
            }
        }
    }
}

TEST_F(JointOptimizerScaleTest, RejectsWrongMatchFromEitherFrameAndRefinesInliers) {
    const std::size_t badPoint = 7;
    const Eigen::Matrix3d initialRotation =
        Eigen::AngleAxisd(0.025, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Vector3d initialTranslation(-2, 0.12, 0.08);
    const double expectedScale = initialTranslation.norm() / trueTranslation.norm();
    auto initialPoints = mapPoints;
    for (auto& point : initialPoints) point *= 1.3;

    for (std::size_t badFrame : {0, 1}) {
        SCOPED_TRACE(badFrame);
        auto mismatchedObservations = observations;
        const auto& observation = observations[badFrame][badPoint];
        mismatchedObservations[badFrame][badPoint] = std::make_shared<KeyPoint>(
            observation->getImgX(), observation->getImgY() + 60);
        const auto [rotation, translation, points] = optimizer.optimizePoseAndPoints(
            mismatchedObservations, initialRotation, initialTranslation, initialPoints, intrinsics);

        ASSERT_EQ(points.size(), mapPoints.size());
        EXPECT_NEAR(translation.norm(), initialTranslation.norm(), 1e-6);
        EXPECT_TRUE(rotation.isApprox(Eigen::Matrix3d::Identity(), 1e-5));
        EXPECT_TRUE(translation.normalized().isApprox(trueTranslation.normalized(), 1e-5));
        for (std::size_t j = 0; j < points.size(); ++j) {
            if (j == badPoint) {
                EXPECT_TRUE(points[j].isApprox(initialPoints[j], 0));
            } else {
                EXPECT_TRUE(points[j].isApprox(expectedScale * mapPoints[j], 1e-5)) << "Point " << j;
            }
        }
    }
}

TEST_F(JointOptimizerScaleTest, RejectsPointsBehindCamerasWithZeroReprojectionError) {
    for (std::size_t j = 0; j < mapPoints.size(); ++j) {
        mapPoints[j] *= -1;
        for (std::size_t frame : {0, 1}) {
            const Eigen::Vector3d cameraPoint = frame == 0
                ? mapPoints[j] : Eigen::Vector3d(mapPoints[j] + trueTranslation);
            const Eigen::Vector3d projection = intrinsics * cameraPoint;
            observations[frame][j] = std::make_shared<KeyPoint>(
                projection.x() / projection.z(), projection.y() / projection.z());
        }
    }
    const auto [rotation, translation, points] = optimizer.optimizePoseAndPoints(
        observations, Eigen::Matrix3d::Identity(), trueTranslation, mapPoints, intrinsics);

    EXPECT_TRUE(rotation.isApprox(Eigen::Matrix3d::Identity(), 1e-10));
    EXPECT_TRUE(translation.isApprox(trueTranslation, 1e-10));
    ASSERT_EQ(points.size(), mapPoints.size());
    for (std::size_t j = 0; j < points.size(); ++j) {
        EXPECT_TRUE(points[j].isApprox(mapPoints[j], 0));
    }
}

TEST_F(JointOptimizerScaleTest, TooFewInliersPreserveExactInputPose) {
    mapPoints.resize(2);
    for (auto& frameObservations : observations) frameObservations.resize(2);
    const auto [rotation, translation, points] = optimizer.optimizePoseAndPoints(
        observations, Eigen::Matrix3d::Identity(), trueTranslation, mapPoints, intrinsics);

    EXPECT_TRUE(rotation.isApprox(Eigen::Matrix3d::Identity(), 1e-10));
    EXPECT_TRUE(translation.isApprox(trueTranslation, 1e-10));
    ASSERT_EQ(points.size(), mapPoints.size());
    for (std::size_t j = 0; j < points.size(); ++j) {
        EXPECT_TRUE(points[j].isApprox(mapPoints[j], 1e-10));
    }
}

TEST_F(JointOptimizerScaleTest, EmptyMatchesPreservePoseAndClearInlierMask) {
    const auto [rotation, translation, points] = optimizer.optimizePoseAndPoints(
        {{}, {}}, Eigen::Matrix3d::Identity(), trueTranslation, {}, intrinsics);

    EXPECT_TRUE(rotation.isApprox(Eigen::Matrix3d::Identity(), 0));
    EXPECT_TRUE(translation.isApprox(trueTranslation, 0));
    EXPECT_TRUE(points.empty());
}

TEST_F(JointOptimizerScaleTest, InvalidCorrespondenceCountsAreRejected) {
    EXPECT_THROW(optimizer.optimizePoseAndPoints({}, Eigen::Matrix3d::Identity(),
        trueTranslation, mapPoints, intrinsics), std::invalid_argument);
    EXPECT_THROW(optimizer.optimizePoseAndPoints({observations[0]}, Eigen::Matrix3d::Identity(),
        trueTranslation, mapPoints, intrinsics), std::invalid_argument);
    observations[1].pop_back();
    EXPECT_THROW(optimizer.optimizePoseAndPoints(observations, Eigen::Matrix3d::Identity(),
        trueTranslation, mapPoints, intrinsics), std::invalid_argument);
}

TEST_F(JointOptimizerScaleTest, InvalidObservationDataIsRejected) {
    auto invalidObservations = observations;
    invalidObservations[0][0].reset();
    EXPECT_THROW(optimizer.optimizePoseAndPoints(invalidObservations, Eigen::Matrix3d::Identity(),
        trueTranslation, mapPoints, intrinsics), std::invalid_argument);

    invalidObservations = observations;
    invalidObservations[1][0] = std::make_shared<KeyPoint>(
        std::numeric_limits<double>::quiet_NaN(), 200);
    EXPECT_THROW(optimizer.optimizePoseAndPoints(invalidObservations, Eigen::Matrix3d::Identity(),
        trueTranslation, mapPoints, intrinsics), std::invalid_argument);

    mapPoints[0].z() = std::numeric_limits<double>::infinity();
    EXPECT_THROW(optimizer.optimizePoseAndPoints(observations, Eigen::Matrix3d::Identity(),
        trueTranslation, mapPoints, intrinsics), std::invalid_argument);
}

TEST_F(JointOptimizerScaleTest, ZeroBaselineIsRejected) {
    EXPECT_THROW(optimizer.optimizePoseAndPoints(observations, Eigen::Matrix3d::Identity(),
        Eigen::Vector3d::Zero(), mapPoints, intrinsics), std::invalid_argument);
}

TEST_F(JointOptimizerScaleTest, NonFiniteBaselineIsRejected) {
    for (double invalid : {std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(optimizer.optimizePoseAndPoints(observations, Eigen::Matrix3d::Identity(),
            Eigen::Vector3d(invalid, 0, 0), mapPoints, intrinsics), std::invalid_argument);
    }
}

class CovisibilityOptimizerTest : public ::testing::Test {
protected:
    Map map{"map"};
    Optimizer optimizer;
    std::array<KeyFrameId, 4> frameIds{};
    std::vector<MapPointId> pointIds;
    Eigen::Matrix3d intrinsics = (Eigen::Matrix3d() <<
        500, 0, 320, 0, 500, 240, 0, 0, 1).finished();

    void SetUp() override {
        std::vector<Eigen::Vector3d> positions;
        for (int i = 0; i < 20; ++i) {
            positions.emplace_back(-0.6 + 0.3 * (i % 5), -0.4 + 0.25 * (i / 5), 4.0 + 0.1 * i);
        }

        for (int camera = 0; camera < 4; ++camera) {
            cv::Mat image(480, 640, CV_8UC1, cv::Scalar(0));
            auto imageData = std::make_shared<ImageData>(
                utils::TTimestamp(std::chrono::milliseconds(camera)), image, intrinsics, "camera");
            auto frame = std::make_shared<Frame>(imageData);
            frame->mMapToCamera = std::make_shared<tft::RigidTransform3D>(
                "map", "camera", Eigen::Matrix3d::Identity(),
                Eigen::Vector3d(-0.3 * camera, 0, 0));
            std::vector<KeyPoint> features;
            for (const auto& position : positions) {
                const auto cameraPoint = frame->mMapToCamera->apply(position);
                features.emplace_back(500 * cameraPoint.x() / cameraPoint.z() + 320,
                                      500 * cameraPoint.y() / cameraPoint.z() + 240);
            }
            frame->mKeypointTree = std::make_shared<QuadTreeNode>(0, 640, 0, 480, features);
            frameIds[camera] = map.addKeyFrame(frame);
        }

        const auto& features = map.mKeyFrames.at(frameIds[0]).mKeyPoints;
        for (std::size_t i = 0; i < positions.size(); ++i) {
            const auto pointId = map.addMapPoint(positions[i], features[i], frameIds[0]);
            pointIds.push_back(pointId);
            for (const auto frameId : frameIds) map.addObservation(frameId, i, pointId);
        }
    }
};

TEST_F(CovisibilityOptimizerTest, SharedCountsAreSymmetricAndDuplicateObservationsAreIdempotent) {
    for (const auto first : frameIds) {
        const auto& frame = map.mKeyFrames.at(first);
        EXPECT_FALSE(frame.mSharedPointCount.contains(first));
        EXPECT_EQ(frame.mObservedPoints.size(), pointIds.size());
        for (const auto second : frameIds) {
            if (first != second) EXPECT_EQ(frame.mSharedPointCount.at(second), pointIds.size());
        }
    }

    map.addObservation(frameIds[3], 0, pointIds[0]);
    EXPECT_EQ(map.mKeyFrames.at(frameIds[3]).mObservedPoints.size(), pointIds.size());
    EXPECT_EQ(map.mKeyFrames.at(frameIds[3]).mSharedPointCount.at(frameIds[0]), pointIds.size());

    const auto otherPoint = map.addMapPoint(Eigen::Vector3d(0, 0, 5),
                                            map.mKeyFrames.at(frameIds[3]).mKeyPoints[0], frameIds[3]);
    EXPECT_THROW(map.addObservation(frameIds[3], 0, otherPoint), std::invalid_argument);
    EXPECT_TRUE(map.mPoints.at(otherPoint).mObservations.empty());
}

TEST_F(CovisibilityOptimizerTest, RemovingPointClearsFeaturesAndSharedCounts) {
    const auto removedId = pointIds.front();
    ASSERT_TRUE(map.removePoint(removedId));
    EXPECT_FALSE(map.mPoints.contains(removedId));
    EXPECT_FALSE(map.removePoint(removedId));

    for (const auto frameId : frameIds) {
        const auto& frame = map.mKeyFrames.at(frameId);
        EXPECT_FALSE(frame.mKeyPoints[0]->getMatched());
        EXPECT_FALSE(frame.mObservedPointByFeature[0].has_value());
        EXPECT_EQ(std::count(frame.mObservedPoints.begin(), frame.mObservedPoints.end(), removedId), 0);
        for (const auto otherId : frameIds) {
            if (frameId != otherId) EXPECT_EQ(frame.mSharedPointCount.at(otherId), pointIds.size() - 1);
        }
    }

    const auto replacementId = map.addMapPoint(Eigen::Vector3d(0, 0, 5),
                                                map.mKeyFrames.at(frameIds[0]).mKeyPoints[0], frameIds[3]);
    for (const auto frameId : frameIds) map.addObservation(frameId, 0, replacementId);
    for (const auto frameId : frameIds) {
        const auto& frame = map.mKeyFrames.at(frameId);
        EXPECT_TRUE(frame.mKeyPoints[0]->getMatched());
        EXPECT_EQ(frame.mObservedPointByFeature[0], replacementId);
        EXPECT_EQ(frame.mObservedPoints.size(), pointIds.size());
    }
}

TEST_F(CovisibilityOptimizerTest, CullingUsesPointRemovalWithoutInvalidatingIteration) {
    map.mPoints.at(pointIds[0]).mVisibleInTracking = 8;
    map.cullMapPoints();
    EXPECT_FALSE(map.mPoints.contains(pointIds[0]));
    EXPECT_EQ(map.mPoints.size(), pointIds.size() - 1);
    EXPECT_EQ(map.mKeyFrames.at(frameIds[0]).mSharedPointCount.at(frameIds[1]), pointIds.size() - 1);
}

TEST_F(CovisibilityOptimizerTest, CullsOldPointWithoutLaterTrackingInliers) {
    for (int camera = 4; camera < 6; ++camera) {
        cv::Mat image(480, 640, CV_8UC1, cv::Scalar(0));
        auto imageData = std::make_shared<ImageData>(
            utils::TTimestamp(std::chrono::milliseconds(camera)), image, intrinsics, "camera");
        auto frame = std::make_shared<Frame>(imageData);
        frame->mMapToCamera = std::make_shared<tft::RigidTransform3D>(
            "map", "camera", Eigen::Matrix3d::Identity(), Eigen::Vector3d(-0.3 * camera, 0, 0));
        frame->mKeypointTree = std::make_shared<QuadTreeNode>(
            0, 640, 0, 480, std::vector<KeyPoint>{});
        map.addKeyFrame(frame);
    }

    auto& stalePoint = map.mPoints.at(pointIds[0]);
    stalePoint.addTriangulationObservation(stalePoint.mDescriptor);
    EXPECT_EQ(stalePoint.mNumInlierObservations, 0);
    stalePoint.mVisibleInTracking = 8;
    ASSERT_GT(map.nextKeyFrameId, stalePoint.mCreatorId + 4);

    map.cullMapPoints();
    EXPECT_FALSE(map.mPoints.contains(pointIds[0]));
    EXPECT_EQ(map.mKeyFrames.at(frameIds[0]).mSharedPointCount.at(frameIds[1]), pointIds.size() - 1);
}

TEST_F(CovisibilityOptimizerTest, CullingWaitsForOpportunitiesAndKeepsSuccessfulPoints) {
    auto& point = map.mPoints.at(pointIds[0]);
    point.mVisibleInTracking = 7;
    map.cullMapPoints();
    ASSERT_TRUE(map.mPoints.contains(pointIds[0]));

    point.addInlierObservation(point.mDescriptor);
    point.addInlierObservation(point.mDescriptor);
    point.mVisibleInTracking = 8;
    map.cullMapPoints();
    ASSERT_TRUE(map.mPoints.contains(pointIds[0]));

    point.mVisibleInTracking = 9;
    map.cullMapPoints();
    EXPECT_FALSE(map.mPoints.contains(pointIds[0]));
}

TEST_F(CovisibilityOptimizerTest, RemovingLastSharedPointErasesCovisibilityEdges) {
    for (const auto pointId : pointIds) EXPECT_TRUE(map.removePoint(pointId));
    EXPECT_TRUE(map.mPoints.empty());
    for (const auto frameId : frameIds) {
        const auto& frame = map.mKeyFrames.at(frameId);
        EXPECT_TRUE(frame.mSharedPointCount.empty());
        EXPECT_TRUE(frame.mObservedPoints.empty());
        for (const auto& feature : frame.mKeyPoints) EXPECT_FALSE(feature->getMatched());
    }
}

TEST_F(CovisibilityOptimizerTest, BoundedLocalAdjustmentRefinesCurrentPoseAndKeepsBoundaryFixed) {
    auto& current = map.mKeyFrames.at(frameIds[3]);
    map.updateKeyFramePose(frameIds[3], tft::RigidTransform3D(
        "map", "camera", Eigen::Matrix3d::Identity(), Eigen::Vector3d(-1.0, 0.02, 0)));
    const auto before = current.mMapToCamera->mTranslation;
    const auto fixed0 = map.mKeyFrames.at(frameIds[0]).mMapToCamera->matrix();
    const auto fixed1 = map.mKeyFrames.at(frameIds[1]).mMapToCamera->matrix();
    const auto excluded = map.mKeyFrames.at(frameIds[2]).mMapToCamera->matrix();

    ASSERT_TRUE(optimizer.optimizeCovisibilityGraph(&map, current, 1, 2));
    const Eigen::Vector3d expected(-0.9, 0, 0);
    EXPECT_LT((current.mMapToCamera->mTranslation - expected).norm(),
              (before - expected).norm());
    EXPECT_TRUE(map.mKeyFrames.at(frameIds[0]).mMapToCamera->matrix().isApprox(fixed0, 0));
    EXPECT_TRUE(map.mKeyFrames.at(frameIds[1]).mMapToCamera->matrix().isApprox(fixed1, 0));
    EXPECT_TRUE(map.mKeyFrames.at(frameIds[2]).mMapToCamera->matrix().isApprox(excluded, 0));
}

TEST_F(CovisibilityOptimizerTest, SelectedReferenceNeighborRemainsFixed) {
    const auto& first = map.mKeyFrames.at(frameIds[0]);
    map.updateKeyFramePose(frameIds[0], tft::RigidTransform3D(
        "map", "camera", Eigen::Matrix3d::Identity(), Eigen::Vector3d(-0.12, 0.02, 0)));
    const auto before = first.mMapToCamera->matrix();
    const auto fixed1 = map.mKeyFrames.at(frameIds[1]).mMapToCamera->matrix();
    const auto fixed2 = map.mKeyFrames.at(frameIds[2]).mMapToCamera->matrix();

    ASSERT_TRUE(optimizer.optimizeCovisibilityGraph(&map, map.mKeyFrames.at(frameIds[3]), 2, 2));
    EXPECT_TRUE(first.mMapToCamera->matrix().isApprox(before, 0));
    EXPECT_TRUE(map.mKeyFrames.at(frameIds[1]).mMapToCamera->matrix().isApprox(fixed1, 0));
    EXPECT_TRUE(map.mKeyFrames.at(frameIds[2]).mMapToCamera->matrix().isApprox(fixed2, 0));
}

TEST_F(CovisibilityOptimizerTest, FinalFullMapPassRemovesOutlierPointsAndTheirObservations) {
    cv::Mat image(480, 640, CV_8UC1, cv::Scalar(0));
    auto imageData = std::make_shared<ImageData>(
        utils::TTimestamp(std::chrono::milliseconds(4)), image, intrinsics, "camera");
    auto fifthFrame = std::make_shared<Frame>(imageData);
    fifthFrame->mMapToCamera = std::make_shared<tft::RigidTransform3D>(
        "map", "camera", Eigen::Matrix3d::Identity(), Eigen::Vector3d(-1.2, 0, 0));
    std::vector<KeyPoint> fifthFeatures;
    for (int i = 0; i < 20; ++i) {
        const Eigen::Vector3d position(-0.6 + 0.3 * (i % 5), -0.4 + 0.25 * (i / 5), 4.0 + 0.1 * i);
        const auto cameraPoint = fifthFrame->mMapToCamera->apply(position);
        fifthFeatures.emplace_back(500 * cameraPoint.x() / cameraPoint.z() + 320,
                                   500 * cameraPoint.y() / cameraPoint.z() + 240);
    }
    fifthFrame->mKeypointTree = std::make_shared<QuadTreeNode>(0, 640, 0, 480, fifthFeatures);
    const auto fifthFrameId = map.addKeyFrame(fifthFrame);
    for (std::size_t i = 0; i < pointIds.size(); ++i)
        map.addObservation(fifthFrameId, i, pointIds[i]);

    map.mKeyFrames.at(frameIds[1]).mKeyPoints.at(0)->scaleBy(1.3, 1.0);
    map.mKeyFrames.at(frameIds[2]).mKeyPoints.at(0)->scaleBy(1.5, 1.0);
    map.mKeyFrames.at(frameIds[3]).mKeyPoints.at(0)->scaleBy(1.7, 1.0);
    map.mKeyFrames.at(frameIds[3]).mKeyPoints.at(1)->scaleBy(1.5, 1.0);

    ASSERT_TRUE(optimizer.optimizeFullMap(&map, true));
    EXPECT_FALSE(map.mPoints.contains(pointIds[0]));
    EXPECT_TRUE(map.mPoints.contains(pointIds[1]));
    EXPECT_TRUE(map.mPoints.contains(pointIds[2]));
    EXPECT_EQ(map.mPoints.size(), pointIds.size() - 1);
    for (const auto frameId : {frameIds[0], frameIds[1], frameIds[2], frameIds[3], fifthFrameId}) {
        const auto& frame = map.mKeyFrames.at(frameId);
        EXPECT_FALSE(frame.mObservedPointByFeature.at(0).has_value());
        EXPECT_FALSE(frame.mKeyPoints.at(0)->getMatched());
        EXPECT_EQ(frame.mObservedPointByFeature.at(1), pointIds[1]);
        for (const auto otherId : {frameIds[0], frameIds[1], frameIds[2], frameIds[3], fifthFrameId}) {
            if (otherId != frameId)
                EXPECT_EQ(frame.mSharedPointCount.at(otherId), pointIds.size() - 1);
        }
    }
}
