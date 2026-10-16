#include <gtest/gtest.h>
#include "random.hpp"
#include "schlam/Ransac.h"

TEST(RansacSamplingTest, SharedSeedReplaysSuccessiveSubsets) {
    auto& engine = utils::randomEngine();
    const auto originalEngine = engine;

    engine.seed(utils::kRandomSeed);
    const auto first = getSparseSubset(8, 100);
    const auto second = getSparseSubset(8, 100);

    engine.seed(utils::kRandomSeed);
    EXPECT_EQ(getSparseSubset(8, 100), first);
    EXPECT_EQ(getSparseSubset(8, 100), second);
    EXPECT_NE(first, second);

    engine.seed(utils::kRandomSeed + 1);
    EXPECT_NE(getSparseSubset(8, 100), first);

    engine = originalEngine;
}

class RansacTest : public ::testing::Test {
  protected:
    RansacTest() {

    }

    ~RansacTest() {

    }

    void SetUp() override {}
    void TearDown() override {}

    
};


TEST_F(RansacTest, calculateEssential) {
// 2D Coordinates (u, v, 1)
    Eigen::Matrix<double, 8, 3> points1;
    Eigen::Matrix<double, 8, 3> points2;

    // We generate 2D points that correspond to a 1-unit translation in X
    // Camera 1 is at (0,0,0). Camera 2 is at (1,0,0).
    // Let's assume a focal length of 1 for "normalized" coordinates.
    
    // Structure: {u, v, 1}
    // Point 1 in world is at (0, 0, 5) -> proj1: (0, 0), proj2: (-1/5, 0)
    points1.row(0) << 0.0f, 0.0f, 1.0f;
    points2.row(0) << -0.2f, 0.0f, 1.0f;

    // Point 2 in world is at (1, 1, 5) -> proj1: (1/5, 1/5), proj2: (0, 1/5)
    points1.row(1) << 0.2f, 0.2f, 1.0f;
    points2.row(1) << 0.0f, 0.2f, 1.0f;

    // Point 3: (1, -1, 5)
    points1.row(2) << 0.2f, -0.2f, 1.0f;
    points2.row(2) << 0.0f, -0.2f, 1.0f;

    // Point 4: (-1, 1, 5)
    points1.row(3) << -0.2f, 0.2f, 1.0f;
    points2.row(3) << -0.4f, 0.2f, 1.0f;

    // Point 5: (-1, -1, 5)
    points1.row(4) << -0.2f, -0.2f, 1.0f;
    points2.row(4) << -0.4f, -0.2f, 1.0f;

    // Point 6: (0, 0, 10)
    points1.row(5) << 0.0f, 0.0f, 1.0f;
    points2.row(5) << -0.1f, 0.0f, 1.0f;

    // Point 7: (2, 2, 10)
    points1.row(6) << 0.2f, 0.2f, 1.0f;
    points2.row(6) << 0.1f, 0.2f, 1.0f;

    // Point 8: (1.5, -0.8, 10), breaking the symmetry to give eight independent constraints.
    points1.row(7) << 0.15f, -0.08f, 1.0f;
    points2.row(7) << 0.05f, -0.08f, 1.0f;

    std::array<Eigen::Matrix<double, 8, 3>, 2> input = {points1, points2};
    auto essential = computeEssential(input);

    // Expected Essential Matrix for pure X-translation:
    // E = [t]x * R = [1, 0, 0]x * I
    Eigen::Matrix3d expectedE;
    expectedE << 0,  0,  0,
                 0,  0, -1,
                 0,  1,  0;

    // Normalize both for comparison (E is defined up to scale)
    essential.normalize();
    expectedE.normalize();

    // The Epipolar Constraint: x2' * E * x1 = 0
    // Verify every correspondence satisfies the constraint with the result.
    for (Eigen::Index i = 0; i < points1.rows(); ++i) {
        double constraint = points2.row(i) * essential * points1.row(i).transpose();
        EXPECT_NEAR(constraint, 0.0f, 1e-4) << "Point " << i;
    }

    // Verify matrix structure (handling sign ambiguity)
    bool match = essential.isApprox(expectedE, 1e-2) || 
                 essential.isApprox(-expectedE, 1e-2);
    EXPECT_TRUE(match) << "Computed E:\n" << essential << "\nExpected E:\n" << expectedE;
}

class RansacParallaxTest : public ::testing::Test {
protected:
    const Eigen::Matrix3d intrinsics = (Eigen::Matrix3d() <<
        1000, 0, 1000, 0, 1000, 1000, 0, 0, 1).finished();
    const Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d translation{-1, 0, 0};
    std::vector<KeyPoint> keypoints1, keypoints2;
    std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matches;
    std::array<std::vector<Eigen::Vector3d>, 2> points;

    void scene(size_t nearCount, size_t farCount) {
        const size_t count = nearCount + farCount;
        keypoints1.reserve(count);
        keypoints2.reserve(count);
        const Eigen::Matrix3d inverseIntrinsics = intrinsics.inverse();
        for (size_t i = 0; i < count; ++i) {
            const auto x = static_cast<uint32_t>(800 + (i % 10) * 40);
            const auto y = static_cast<uint32_t>(800 + (i / 10) * 20);
            // A unit baseline gives depths of 20 and 1000 for these disparities.
            const uint32_t disparity = i < nearCount ? 50 : 1;
            keypoints1.emplace_back(x, y);
            keypoints2.emplace_back(x - disparity, y);
            matches[0].push_back(std::make_shared<KeyPoint>(keypoints1.back()));
            matches[1].push_back(std::make_shared<KeyPoint>(keypoints2.back()));
            points[0].push_back(inverseIntrinsics * Eigen::Vector3d(x, y, 1));
            points[1].push_back(inverseIntrinsics * Eigen::Vector3d(x - disparity, y, 1));
        }
    }

    auto recover() {
        Eigen::Matrix3d essential;
        essential << 0, 0, 0, 0, 0, 1, 0, -1, 0;
        return recoverPoseFromEssential(essential, points, matches,
            std::vector<bool>(matches[0].size(), true), intrinsics);
    }
};

TEST_F(RansacParallaxTest, LowParallaxInlierSupportsPose) {
    const Eigen::Vector3d point1(0, 0, 1000), point2(-1, 0, 1000);
    const auto keypoint1 = std::make_shared<KeyPoint>(1000, 1000);
    const auto keypoint2 = std::make_shared<KeyPoint>(999, 1000);
    const auto [valid, cosine] = isValid(point1, point2, rotation, translation,
        intrinsics, keypoint1, keypoint2);
    EXPECT_TRUE(valid);
    EXPECT_GE(cosine, 0.99998);
}

TEST_F(RansacParallaxTest, LowParallaxWithBadReprojectionIsRejected) {
    const Eigen::Vector3d point1(0, 0, 1000), point2(-1, 0, 1000);
    const auto keypoint1 = std::make_shared<KeyPoint>(1000, 1000);
    const auto keypoint2 = std::make_shared<KeyPoint>(1004, 1000);
    EXPECT_FALSE(isValid(point1, point2, rotation, translation,
        intrinsics, keypoint1, keypoint2).first);
}

TEST_F(RansacParallaxTest, HighParallaxPointBehindCameraIsRejected) {
    const Eigen::Vector3d point1(0, 0, -20), point2(-1, 0, -20);
    const auto keypoint1 = std::make_shared<KeyPoint>(1000, 1000);
    const auto keypoint2 = std::make_shared<KeyPoint>(1050, 1000);
    EXPECT_FALSE(isValid(point1, point2, rotation, translation,
        intrinsics, keypoint1, keypoint2).first);
}

TEST_F(RansacParallaxTest, DegeneratePointsDoNotSupportPose) {
    const auto keypoint1 = std::make_shared<KeyPoint>(1000, 1000);
    const auto keypoint2 = std::make_shared<KeyPoint>(999, 1000);
    for (const Eigen::Vector3d point1 : {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1000, 0, 0)}) {
        const Eigen::Vector3d point2 = point1 + translation;
        EXPECT_FALSE(isValid(point1, point2, rotation, translation,
            intrinsics, keypoint1, keypoint2).first);
    }
}

TEST_F(RansacParallaxTest, PoseSupportIncludesFarPointsButTriangulatedMaskDoesNot) {
    scene(100, 20);
    const auto result = recover();
    ASSERT_TRUE(result.has_value());
    const auto& [recoveredRotation, recoveredTranslation, cloud, triangulated] = *result;
    EXPECT_TRUE(recoveredRotation.isApprox(rotation, 1e-4f));
    EXPECT_TRUE(recoveredTranslation.isApprox(translation, 1e-4f));
    ASSERT_EQ(cloud.size(), 120);
    ASSERT_EQ(triangulated.size(), 120);
    for (size_t i = 0; i < triangulated.size(); ++i) {
        EXPECT_EQ(triangulated[i], i < 100) << "Point " << i;
    }
}

TEST_F(RansacParallaxTest, LowParallaxOnlyCannotInitialize) {
    scene(0, 120);
    EXPECT_FALSE(recover().has_value());
}

class RansacZeroScoreTest : public ::testing::Test {
protected:
    const Eigen::Matrix3d intrinsics = (Eigen::Matrix3d() <<
        458.654f, 0, 367.215f, 0, 457.296f, 248.375f, 0, 0, 1).finished();
    // Undistorted EuRoC V1_01_easy frames 200 and 220.
    std::array<KeyPoint, 8> keypoints1 = {
        KeyPoint(594, 90), KeyPoint(186, 284), KeyPoint(192, 267), KeyPoint(320, 379),
        KeyPoint(364, 386), KeyPoint(227, 446), KeyPoint(461, 272), KeyPoint(447, 288)
    };
    std::array<KeyPoint, 8> keypoints2 = {
        KeyPoint(511, 58), KeyPoint(157, 276), KeyPoint(165, 238), KeyPoint(308, 353),
        KeyPoint(356, 355), KeyPoint(222, 433), KeyPoint(418, 236), KeyPoint(409, 252)
    };
    std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matches;
    std::array<std::vector<Eigen::Vector3d>, 2> points;

    void SetUp() override {
        const Eigen::Matrix3d inverseIntrinsics = intrinsics.inverse();
        for (size_t i = 0; i < keypoints1.size(); ++i) {
            matches[0].push_back(std::make_shared<KeyPoint>(keypoints1[i]));
            matches[1].push_back(std::make_shared<KeyPoint>(keypoints2[i]));
            for (size_t camera = 0; camera < matches.size(); ++camera) {
                const auto& keypoint = matches[camera][i];
                points[camera].push_back(inverseIntrinsics * Eigen::Vector3d(
                    keypoint->getImgX(), keypoint->getImgY(), 1));
            }
        }
    }

    void expectZeroScoreOutputs(decltype(&findEssential) findModel) {
        Eigen::Matrix3d model = Eigen::Matrix3d::Identity();
        std::vector<bool> inliers(keypoints1.size(), true);
        double score = 42;
        // With double precision this sample can score a model, so use an empty
        // candidate list to exercise the zero-score output contract directly.
        findModel(points, {}, inliers, score, model, 1.0 / intrinsics(0, 0));
        ASSERT_EQ(score, 0.0);
        EXPECT_TRUE(model.isZero(0));
        EXPECT_EQ(inliers, std::vector<bool>(keypoints1.size(), false));
    }
};

TEST_F(RansacZeroScoreTest, EssentialSearchClearsUnscoredOutputs) {
    expectZeroScoreOutputs(&findEssential);
}

TEST_F(RansacZeroScoreTest, HomographySearchClearsUnscoredOutputs) {
    expectZeroScoreOutputs(&findHomography);
}

TEST_F(RansacZeroScoreTest, ReconstructionRejectsUnscoredModels) {
    EXPECT_FALSE(reconstructInitial(matches[0], matches[1], intrinsics).has_value());
}
