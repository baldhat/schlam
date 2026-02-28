#include <gtest/gtest.h>
#include "schlam/Ransac.h"

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
    Eigen::Matrix<float, 8, 3> points1;
    Eigen::Matrix<float, 8, 3> points2;

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

    // Point 8: (-2, -2, 10)
    points1.row(7) << -0.2f, -0.2f, 1.0f;
    points2.row(7) << -0.3f, -0.2f, 1.0f;

    std::array<Eigen::Matrix<float, 8, 3>, 2> input = {points1, points2};
    auto essential = computeEssential(input);

    // Expected Essential Matrix for pure X-translation:
    // E = [t]x * R = [1, 0, 0]x * I
    Eigen::Matrix3f expectedE;
    expectedE << 0,  0,  0,
                 0,  0, -1,
                 0,  1,  0;

    // Normalize both for comparison (E is defined up to scale)
    essential.normalize();
    expectedE.normalize();

    // The Epipolar Constraint: x2' * E * x1 = 0
    // Verify at least one point satisfies the constraint with the result
    float constraint = points2.row(0) * essential * points1.row(0).transpose();
    EXPECT_NEAR(constraint, 0.0f, 1e-4);

    // Verify matrix structure (handling sign ambiguity)
    bool match = essential.isApprox(expectedE, 1e-2) || 
                 essential.isApprox(-expectedE, 1e-2);
    EXPECT_TRUE(match) << "Computed E:\n" << essential << "\nExpected E:\n" << expectedE;
}
