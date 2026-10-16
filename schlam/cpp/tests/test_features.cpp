#include <gtest/gtest.h>

#include "schlam/FastFeatureDetector.h"
#include "schlam/Matcher.h"
#include "schlam/ORBFeatureDetector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

class ORBFeatureDetectorTest : public ::testing::Test {
protected:
    ORBFeatureDetector detector{1000, nullptr, 1};

    auto filter(std::vector<KeyPoint> points, uint32_t count) {
        return detector.filterWithOctree(points, 100, 100, count);
    }

    auto detect(const cv::Mat& image) {
        auto tree = detector.calcFeatures({image});
        detector.addOrientation({image}, tree.get());
        detector.addDescriptors({image}, tree.get());
        return tree;
    }

    auto detectPyramid(cv::Mat image, uint32_t count = 1000, uint8_t levels = 8) {
        ORBFeatureDetector multilevel(count, nullptr, levels);
        auto data = std::make_shared<ImageData>(utils::TTimestamp{}, image,
            Eigen::Matrix3d::Identity(), "pyramid");
        Frame frame(data);
        multilevel.detectFeatures(frame);
        return std::pair(frame.mKeypointTree, multilevel.mNumFeaturesPerLevel);
    }

    auto selectPyramid(const std::vector<cv::Mat>& pyramid, uint32_t count) {
        ORBFeatureDetector multilevel(count, nullptr, static_cast<uint8_t>(pyramid.size()));
        return multilevel.calcFeatures(pyramid);
    }

    void describe(const cv::Mat& image, QuadTreeNode& tree) {
        detector.addDescriptors({image}, &tree);
    }

    auto harris(const cv::Mat& image, std::vector<KeyPoint> points) {
        detector.computeHarrisResponse(image, points, 7);
        return points;
    }

    auto describeRescaledCorner(const cv::Mat& image, uint8_t level) {
        ORBFeatureDetector multilevel(1000, nullptr, 8);
        const auto pyramid = buildPyramid(image, 8, 1.0 / 1.2);
        QuadTreeNode expected(0, pyramid[level].cols, 0, pyramid[level].rows,
            {KeyPoint(24, 29)});
        detector.addOrientation({pyramid[level]}, &expected);
        detector.addDescriptors({pyramid[level]}, &expected);

        std::vector<std::shared_ptr<KeyPoint>> points{std::make_shared<KeyPoint>(24, 29)};
        multilevel.rescaleFeatures(points, 1.0 / multilevel.mLevelFactors[level], level);
        QuadTreeNode actual(0, image.cols, 0, image.rows, points, nullptr);
        multilevel.addOrientation(pyramid, &actual);
        multilevel.addDescriptors(pyramid, &actual);
        return std::pair(actual.getFeatures()[0], expected.getFeatures()[0]);
    }
};

TEST(KeyPointTest, ScalingAndGeometryConversionPreserveFractionalPositions) {
    auto point = std::make_shared<KeyPoint>(24.25, 29.75);
    EXPECT_DOUBLE_EQ(point->getImgX(), 24.25);
    EXPECT_DOUBLE_EQ(point->getImgY(), 29.75);
    point->scaleBy(1.2, 1.44);

    const auto positions = toEigen({point});
    ASSERT_EQ(positions.size(), 1);
    EXPECT_NEAR(positions[0].x(), 29.1, 1e-12);
    EXPECT_NEAR(positions[0].y(), 42.84, 1e-12);
    point->scaleBy(1.0 / 1.2, 1.0 / 1.44);
    EXPECT_NEAR(point->getImgX(), 24.25, 1e-12);
    EXPECT_NEAR(point->getImgY(), 29.75, 1e-12);
}

TEST(QuadTreeTest, EmptyTreeCanBeQueriedAndFiltered) {
    QuadTreeNode tree(0, 100, 0, 100, {});
    EXPECT_TRUE(tree.getFeatures().empty());
    EXPECT_TRUE(tree.getFeatures(0, 100, 0, 100).empty());
    tree.divide();
    tree.retainBestFeaturesPerLeaf();
    EXPECT_TRUE(tree.getFeatures().empty());
}

TEST(QuadTreeTest, UnsplitTreeReturnsAllFeaturesInsideWindow) {
    QuadTreeNode tree(0, 100, 0, 100, {KeyPoint(25, 25), KeyPoint(75, 75)});
    EXPECT_EQ(tree.getFeatures().size(), 2);
    EXPECT_EQ(tree.getFeatures(0, 100, 0, 100).size(), 2);
    EXPECT_EQ(tree.getFeatures(20, 30, 20, 30).size(), 1);
}

TEST(QuadTreeTest, WindowContainedWithinChildFindsFeature) {
    QuadTreeNode tree(0, 100, 0, 100, {KeyPoint(25, 25), KeyPoint(75, 75)});
    tree.divide();
    auto points = tree.getFeatures(20, 30, 20, 30);
    ASSERT_EQ(points.size(), 1);
    EXPECT_EQ(points[0]->getImgX(), 25);
    EXPECT_EQ(tree.getFeatures(25, 25, 25, 25).size(), 1);
    EXPECT_TRUE(tree.getFeatures(110, 120, 110, 120).empty());
}

TEST(QuadTreeTest, FeatureIdentityAndLifetimeSurviveSubdivision) {
    auto tree = std::make_unique<QuadTreeNode>(0, 100, 0, 100,
        std::vector<KeyPoint>{KeyPoint(25, 25), KeyPoint(75, 75)});
    auto features = tree->getFeatures();
    std::weak_ptr<KeyPoint> retained = features[0];

    tree->divide();
    const auto window = tree->getFeatures(20, 30, 20, 30);
    ASSERT_EQ(window.size(), 1);
    EXPECT_EQ(window[0], features[0]);
    window[0]->setScore(42);
    EXPECT_EQ(features[0]->getScore(), 42);

    tree.reset();
    EXPECT_FALSE(retained.expired());
    EXPECT_EQ(features[0]->getImgX(), 25);
    EXPECT_EQ(features[1]->getImgY(), 75);
}

TEST(MatcherTest, WindowMatchingCanBeRepeatedAndUsesEachTargetOnce) {
    auto source = std::make_shared<QuadTreeNode>(0, 100, 0, 100,
        std::vector<KeyPoint>{KeyPoint(25, 25), KeyPoint(26, 26)});
    auto target = std::make_shared<QuadTreeNode>(0, 100, 0, 100,
        std::vector<KeyPoint>{KeyPoint(25, 25)});
    // Persistent state from another operation must not affect this matching call.
    target->getFeatures()[0]->setObservation(42);
    const auto [first, firstMask] = matchWindow(source, target, 20, 5);
    const auto [second, secondMask] = matchWindow(source, target, 20, 5);
    EXPECT_EQ(first[0].size(), 1);
    EXPECT_EQ(second[0].size(), 1);
    EXPECT_EQ(firstMask, (std::vector<bool>{true, false}));
    EXPECT_EQ(secondMask, firstMask);
}

TEST(MatcherTest, IdenticalFeaturesMatchThroughDividedTree) {
    auto source = std::make_shared<QuadTreeNode>(0, 100, 0, 100,
        std::vector<KeyPoint>{KeyPoint(25, 25)});
    auto target = std::make_shared<QuadTreeNode>(0, 100, 0, 100,
        std::vector<KeyPoint>{KeyPoint(25, 25), KeyPoint(75, 75)});
    target->divide();
    const auto [matches, matched] = matchWindow(source, target, 20, 5);
    EXPECT_EQ(matches[0].size(), 1);
    EXPECT_EQ(matched, (std::vector<bool>{true}));
}

TEST(MatcherTest, WindowMatchesOwnFeaturesAfterTreesAreDestroyed) {
    auto source = std::make_shared<QuadTreeNode>(0, 100, 0, 100,
        std::vector<KeyPoint>{KeyPoint(25, 25)});
    auto target = std::make_shared<QuadTreeNode>(0, 100, 0, 100,
        std::vector<KeyPoint>{KeyPoint(26, 25), KeyPoint(75, 75)});
    target->divide();
    std::weak_ptr<KeyPoint> sourceFeature = source->getFeatures()[0];
    std::weak_ptr<KeyPoint> targetFeature = target->getFeatures(20, 30, 20, 30)[0];

    auto [matches, matched] = matchWindow(source, target, 20, 5);
    ASSERT_EQ(matches[0].size(), 1);
    ASSERT_EQ(matches[1].size(), 1);
    source.reset();
    target.reset();

    EXPECT_EQ(matches[0][0], sourceFeature.lock());
    EXPECT_EQ(matches[1][0], targetFeature.lock());
    EXPECT_EQ(matches[0][0]->getImgX(), 25);
    EXPECT_EQ(matches[1][0]->getImgX(), 26);
    matches = {};
    EXPECT_TRUE(sourceFeature.expired());
    EXPECT_TRUE(targetFeature.expired());
}

TEST(MatcherTest, GlobalMatchesOwnFeaturesAndUseEachTargetOnce) {
    auto source = std::make_shared<QuadTreeNode>(0, 100, 0, 100,
        std::vector<KeyPoint>{KeyPoint(25, 25), KeyPoint(26, 25)});
    auto target = std::make_shared<QuadTreeNode>(0, 100, 0, 100,
        std::vector<KeyPoint>{KeyPoint(75, 75)});
    auto matches = matchGlobally(source, target, 20);
    ASSERT_EQ(matches[0].size(), 1);
    ASSERT_EQ(matches[1].size(), 1);
    EXPECT_EQ(matches[0][0], source->getFeatures()[0]);
    EXPECT_EQ(matches[1][0], target->getFeatures()[0]);

    source.reset();
    target.reset();
    EXPECT_EQ(matches[0][0]->getImgX(), 25);
    EXPECT_EQ(matches[1][0]->getImgY(), 75);
}

TEST(MatcherTest, ProjectionMatchesOwnProjectedAndDetectedFeatures) {
    cv::Mat image(100, 100, CV_8UC1, cv::Scalar(0));
    Eigen::Matrix3d intrinsics;
    intrinsics << 100, 0, 50, 0, 100, 50, 0, 0, 1;
    auto imageData = std::make_shared<ImageData>(utils::TTimestamp{}, image, intrinsics, "current");
    auto frame = std::make_shared<Frame>(imageData);
    std::array<uint8_t, 32> descriptor{};
    descriptor[0] = 0xa5;
    frame->mKeypointTree = std::make_shared<QuadTreeNode>(0, 100, 0, 100,
        std::vector<KeyPoint>{KeyPoint(51, 50, 0, 0, 0, descriptor), KeyPoint(80, 80)});
    frame->mKeypointTree->divide();
    std::weak_ptr<KeyPoint> observation = frame->mKeypointTree->getFeatures(49, 53, 48, 52)[0];

    auto map = std::make_shared<Map>("world");
    const auto descriptorSource = std::make_shared<KeyPoint>(0, 0, 0, 0, 0, descriptor);
    map->addMapPoint(Eigen::Vector3d(0, 0, -5), descriptorSource, 0);
    map->addMapPoint(Eigen::Vector3d(0, 0, 5), descriptorSource, 0);
    map->addMapPoint(Eigen::Vector3d(10, 0, 5), descriptorSource, 0);
    // Both visible projections compete for the same observation.
    map->addMapPoint(Eigen::Vector3d(0.05, 0, 5), descriptorSource, 0);
    map->addMapPoint(Eigen::Vector3d(1, 1, 5), descriptorSource, 0);
    auto pose = std::make_shared<tft::RigidTransform3D>("current", "world",
        Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    auto transformer = std::make_shared<tft::Transformer>();
    transformer->registerTransform(pose);

    auto [matches, matched] = matchByProjection(map.get(), frame, pose, transformer, 20, 2);
    ASSERT_EQ(matches[0].size(), 1);
    ASSERT_EQ(matches[1].size(), 1);
    EXPECT_EQ(matched, (std::vector<bool>{false, true, false, false, false}));
    std::weak_ptr<KeyPoint> projection = matches[0][0];
    std::weak_ptr<Frame> originalFrame = frame;
    frame.reset();
    map.reset();

    EXPECT_TRUE(originalFrame.expired());
    EXPECT_EQ(matches[0][0]->getImgX(), 50);
    EXPECT_EQ(matches[0][0]->getImgY(), 50);
    EXPECT_EQ(matches[0][0]->getDescriptor(), descriptor);
    EXPECT_EQ(matches[1][0], observation.lock());
    EXPECT_EQ(matches[1][0]->getImgX(), 51);
    EXPECT_EQ(matches[1][0]->getImgY(), 50);
    matches = {};
    EXPECT_TRUE(projection.expired());
    EXPECT_TRUE(observation.expired());
}

TEST(MatcherTest, WindowMaskIncludesUnmatchedSourcesAtTheirOriginalIndices) {
    std::vector<std::shared_ptr<KeyPoint>> source{
        std::make_shared<KeyPoint>(80, 80),
        std::make_shared<KeyPoint>(25, 25),
        std::make_shared<KeyPoint>(26, 25)};
    auto target = std::make_shared<QuadTreeNode>(0, 100, 0, 100,
        std::vector<KeyPoint>{KeyPoint(25, 25)});
    const auto [matches, matched] = matchWindow(source, target, 20, 2);
    ASSERT_EQ(matches[0].size(), 1);
    EXPECT_EQ(matches[0][0], source[1]);
    EXPECT_EQ(matched, (std::vector<bool>{false, true, false}));
}

TEST(MatcherTest, ProjectionMaskPreservesMapPointAndObservationOrder) {
    cv::Mat image(100, 100, CV_8UC1, cv::Scalar(0));
    Eigen::Matrix3d intrinsics;
    intrinsics << 100, 0, 50, 0, 100, 50, 0, 0, 1;
    auto imageData = std::make_shared<ImageData>(utils::TTimestamp{}, image, intrinsics, "current");
    auto frame = std::make_shared<Frame>(imageData);
    frame->mKeypointTree = std::make_shared<QuadTreeNode>(0, 100, 0, 100,
        std::vector<KeyPoint>{KeyPoint(30, 30), KeyPoint(70, 70)});
    frame->mKeypointTree->divide();
    std::array<uint8_t, 32> descriptor{};
    auto map = std::make_shared<Map>("world");
    const auto descriptorSource = std::make_shared<KeyPoint>(0, 0, 0, 0, 0, descriptor);
    map->addMapPoint(Eigen::Vector3d(10, 0, 5), descriptorSource, 0);
    map->addMapPoint(Eigen::Vector3d(1, 1, 5), descriptorSource, 0);
    map->addMapPoint(Eigen::Vector3d(0, 0, -5), descriptorSource, 0);
    map->addMapPoint(Eigen::Vector3d(-1, -1, 5), descriptorSource, 0);
    auto pose = std::make_shared<tft::RigidTransform3D>("current", "world",
        Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    auto transformer = std::make_shared<tft::Transformer>();
    transformer->registerTransform(pose);

    const auto [matches, matched] = matchByProjection(map.get(), frame, pose, transformer, 20, 1);
    ASSERT_EQ(matches[1].size(), 2);
    EXPECT_EQ(matched, (std::vector<bool>{false, true, false, true}));
    std::size_t observationIndex = 0;
    for (std::size_t mapIndex = 0; mapIndex < map->mPoints.size(); ++mapIndex) {
        if (matched[mapIndex]) {
            const Eigen::Vector3d projected = intrinsics * map->mPoints.at(mapIndex).mPosition;
            EXPECT_EQ(matches[1][observationIndex]->getImgX(), projected.x() / projected.z());
            EXPECT_EQ(matches[1][observationIndex]->getImgY(), projected.y() / projected.z());
            ++observationIndex;
        }
    }
}

TEST_F(ORBFeatureDetectorTest, SparseInputPreservesFeatures) {
    auto tree = filter({KeyPoint(25, 25), KeyPoint(75, 75)}, 1000);
    EXPECT_EQ(tree->getFeatures().size(), 2);
    EXPECT_EQ(tree->getFeatures(0, 100, 0, 100).size(), 2);
}

TEST_F(ORBFeatureDetectorTest, FilteringRetainsSingletonLeavesAndHonorsBudget) {
    auto tree = filter({KeyPoint(25, 25, 0, 4), KeyPoint(75, 25, 0, 3),
                        KeyPoint(25, 75, 0, 2), KeyPoint(75, 75, 0, 1)}, 3);
    auto points = tree->getFeatures();
    ASSERT_EQ(points.size(), 3);
    EXPECT_TRUE(std::all_of(points.begin(), points.end(), [](const auto& point) {
        return point->getScore() >= 2;
    }));
}

TEST_F(ORBFeatureDetectorTest, FilteringTerminatesForCoincidentFeatures) {
    auto tree = filter({KeyPoint(25, 25, 0, 1), KeyPoint(25, 25, 0, 3),
                        KeyPoint(25, 25, 0, 2)}, 2);
    ASSERT_EQ(tree->getFeatures().size(), 1);
    EXPECT_EQ(tree->getFeatures()[0]->getScore(), 3);
}

TEST_F(ORBFeatureDetectorTest, EmptyInputAndZeroBudgetReturnEmptyTrees) {
    EXPECT_TRUE(filter({}, 1000)->getFeatures().empty());
    EXPECT_TRUE(filter({KeyPoint(25, 25)}, 0)->getFeatures().empty());
}

TEST_F(ORBFeatureDetectorTest, BlankImageHasNoFeatures) {
    EXPECT_TRUE(detect(cv::Mat(64, 64, CV_8UC1, cv::Scalar(100)))->getFeatures().empty());
}

TEST_F(ORBFeatureDetectorTest, HarrisResponsesMatchOpenCVForCornerEdgeAndFlatRegions) {
    cv::Mat image(80, 80, CV_8UC1, cv::Scalar(0));
    cv::rectangle(image, cv::Rect(24, 24, 32, 32), cv::Scalar(255), cv::FILLED);
    cv::Mat reference;
    cv::cornerHarris(image, reference, 7, 3, 0.04);
    const auto points = harris(image, {KeyPoint(24, 24), KeyPoint(40, 24), KeyPoint(40, 40)});

    EXPECT_GT(points[0].getScore(), 0);
    EXPECT_LT(points[1].getScore(), 0);
    EXPECT_DOUBLE_EQ(points[2].getScore(), 0);
    for (const auto& point : points) {
        const int x = static_cast<int>(point.getImgX());
        const int y = static_cast<int>(point.getImgY());
        EXPECT_TRUE(std::isfinite(point.getScore()));
        EXPECT_NEAR(point.getScore(), reference.at<float>(y, x), 1e-7);
    }
}

TEST_F(ORBFeatureDetectorTest, PyramidRescalingPreservesOrientationAndDescriptorSampling) {
    cv::Mat image(256, 256, CV_8UC1);
    cv::RNG rng(12345);
    rng.fill(image, cv::RNG::UNIFORM, 0, 256);
    for (uint8_t level = 0; level < 8; ++level) {
        SCOPED_TRACE(static_cast<int>(level));
        const auto [actual, expected] = describeRescaledCorner(image, level);
        EXPECT_NEAR(actual->getAngle(), expected->getAngle(), 1e-12);
        EXPECT_EQ(actual->getDescriptor(), expected->getDescriptor());
    }
}

TEST_F(ORBFeatureDetectorTest, PyramidRetainsEveryLevelQuotaAndSupportsFractionalWindowQueries) {
    cv::Mat image(480, 752, CV_8UC1);
    cv::RNG rng(12345);
    rng.fill(image, cv::RNG::UNIFORM, 0, 256);
    const auto [tree, budgets] = detectPyramid(image);
    EXPECT_EQ(budgets, (std::vector<uint32_t>{217, 181, 151, 126, 105, 87, 73, 60}));
    const auto features = tree->getFeatures();
    ASSERT_EQ(features.size(), 1000);
    std::vector<uint32_t> counts(8, 0);
    for (const auto& feature : features) {
        ASSERT_LT(feature->getLevel(), counts.size());
        ++counts[feature->getLevel()];
        const auto nearby = tree->getFeatures(
            feature->getImgX() - 1e-6, feature->getImgX() + 1e-6,
            feature->getImgY() - 1e-6, feature->getImgY() + 1e-6);
        EXPECT_NE(std::find(nearby.begin(), nearby.end(), feature), nearby.end());
    }
    EXPECT_EQ(counts, budgets);
    EXPECT_EQ(tree->getFeatures(0, image.cols, 0, image.rows).size(), features.size());
}

TEST_F(ORBFeatureDetectorTest, SmallPyramidBudgetsDoNotExceedTotalFeatureLimit) {
    cv::Mat image(256, 256, CV_8UC1);
    cv::RNG rng(12345);
    rng.fill(image, cv::RNG::UNIFORM, 0, 256);
    for (uint32_t limit : {0, 7}) {
        SCOPED_TRACE(limit);
        const auto [tree, budgets] = detectPyramid(image, limit);
        std::vector<uint32_t> counts(8, 0);
        for (const auto& feature : tree->getFeatures()) {
            ASSERT_LT(feature->getLevel(), counts.size());
            ++counts[feature->getLevel()];
        }
        uint32_t allocated = 0;
        for (size_t level = 0; level < budgets.size(); ++level) {
            allocated += budgets[level];
            EXPECT_LE(counts[level], budgets[level]);
        }
        EXPECT_EQ(allocated, limit);
        EXPECT_EQ(tree->getFeatures().size(), limit);
    }
}

TEST_F(ORBFeatureDetectorTest, BlankPyramidDoesNotFillMissingLevelQuotas) {
    const auto [tree, budgets] = detectPyramid(cv::Mat(256, 256, CV_8UC1, cv::Scalar(100)));
    EXPECT_TRUE(tree->getFeatures().empty());
    EXPECT_TRUE(tree->getFeatures(0, 256, 0, 256).empty());
}

TEST_F(ORBFeatureDetectorTest, PyramidPreservesCoincidentFeaturesFromDifferentLevels) {
    cv::Mat fine(144, 144, CV_8UC1, cv::Scalar(0));
    cv::Mat coarse(120, 120, CV_8UC1, cv::Scalar(0));
    fine.at<uint8_t>(72, 72) = 255;
    coarse.at<uint8_t>(60, 60) = 255;
    const auto tree = selectPyramid({fine, coarse}, 2);
    const auto features = tree->getFeatures(72 - 1e-6, 72 + 1e-6, 72 - 1e-6, 72 + 1e-6);
    ASSERT_EQ(tree->getFeatures().size(), 2);
    ASSERT_EQ(features.size(), 2);
    EXPECT_NE(features[0]->getLevel(), features[1]->getLevel());
}

TEST_F(ORBFeatureDetectorTest, SparsePyramidPreservesAvailableFeaturesWithinTheirQuota) {
    cv::Mat fine(144, 144, CV_8UC1, cv::Scalar(0));
    cv::Mat coarse(120, 120, CV_8UC1, cv::Scalar(0));
    coarse.at<uint8_t>(60, 60) = 255;
    const auto tree = selectPyramid({fine, coarse}, 1000);
    const auto features = tree->getFeatures();
    ASSERT_EQ(features.size(), 1);
    EXPECT_EQ(features[0]->getLevel(), 1);
    EXPECT_NEAR(features[0]->getImgX(), 72, 1e-12);
    EXPECT_NEAR(features[0]->getImgY(), 72, 1e-12);
}

TEST(FastFeatureDetectorTest, DetectsEveryRotationOfNineSampleArc) {
    const schlam::FastFeatureDetector detector;
    // Standard FAST circle in (row, column) coordinates, independent of the detector's table.
    const std::array<std::array<int, 2>, 16> ring{{
        {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}, {0, 3}, {1, 3}, {2, 2}, {3, 1},
        {3, 0}, {3, -1}, {2, -2}, {1, -3}, {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}
    }};
    for (int value : {0, 200}) {
        for (int start = 0; start < 16; ++start) {
            cv::Mat image(9, 9, CV_8UC1, cv::Scalar(100));
            for (int i = 0; i < 9; ++i) {
                const auto& offset = ring[(start + i) % 16];
                image.at<uint8_t>(4 + offset[0], 4 + offset[1]) = value;
            }
            const auto points = detector.calculateFastFeatures(image);
            EXPECT_TRUE(std::any_of(points.begin(), points.end(), [](const auto& point) {
                return point.getImgX() == 4 && point.getImgY() == 4;
            })) << "Arc start " << start << ", intensity " << value;
        }
    }
}

TEST(FastFeatureDetectorTest, ConstructorThresholdControlsBrightAndDarkCornerDetection) {
    const schlam::FastFeatureDetector belowContrast(19);
    const schlam::FastFeatureDetector atContrast(20);
    for (int value : {80, 120}) {
        cv::Mat image(9, 9, CV_8UC1, cv::Scalar(value));
        image.at<uint8_t>(4, 4) = 100;
        const auto detectsCenter = [&](const auto& detector) {
            const auto points = detector.calculateFastFeatures(image);
            return std::any_of(points.begin(), points.end(), [](const auto& point) {
                return point.getImgX() == 4 && point.getImgY() == 4;
            });
        };
        EXPECT_TRUE(detectsCenter(belowContrast));
        EXPECT_FALSE(detectsCenter(atContrast));
    }
}

TEST(FastFeatureDetectorTest, ConfigurableArcLengthAcceptsExactArcsAndRejectsShorterArcs) {
    const std::array<std::array<int, 2>, 16> ring{{
        {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}, {0, 3}, {1, 3}, {2, 2}, {3, 1},
        {3, 0}, {3, -1}, {2, -2}, {1, -3}, {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}
    }};
    for (int length = 1; length <= 16; ++length) {
        const schlam::FastFeatureDetector detector(7, length);
        for (int value : {0, 200}) {
            for (int start = 0; start < 16; ++start) {
                SCOPED_TRACE(::testing::Message() << "Length " << length
                    << ", start " << start << ", intensity " << value);
                cv::Mat image(9, 9, CV_8UC1, cv::Scalar(100));
                for (int i = 0; i < length; ++i) {
                    const auto& offset = ring[(start + i) % 16];
                    image.at<uint8_t>(4 + offset[0], 4 + offset[1]) = value;
                }
                const auto detectsCenter = [&] {
                    const auto points = detector.calculateFastFeatures(image);
                    return std::any_of(points.begin(), points.end(), [](const auto& point) {
                        return point.getImgX() == 4 && point.getImgY() == 4;
                    });
                };
                EXPECT_TRUE(detectsCenter());
                const auto& last = ring[(start + length - 1) % 16];
                image.at<uint8_t>(4 + last[0], 4 + last[1]) = 100;
                EXPECT_FALSE(detectsCenter());
            }
        }
    }
}

TEST(FastFeatureDetectorTest, ConstructorRejectsInvalidParameters) {
    EXPECT_THROW(schlam::FastFeatureDetector(-1), std::invalid_argument);
    EXPECT_THROW(schlam::FastFeatureDetector(256), std::invalid_argument);
    EXPECT_THROW(schlam::FastFeatureDetector(7, -1), std::invalid_argument);
    EXPECT_THROW(schlam::FastFeatureDetector(7, 0), std::invalid_argument);
    EXPECT_THROW(schlam::FastFeatureDetector(7, 17), std::invalid_argument);
    EXPECT_NO_THROW(schlam::FastFeatureDetector(0, 1));
    EXPECT_NO_THROW(schlam::FastFeatureDetector(255, 16));
}

TEST_F(ORBFeatureDetectorTest, DescriptorUsesNegativeOffsetsAtZeroRotation) {
    cv::Mat image(64, 64, CV_8UC1);
    // Pattern bit 0 compares (8,-3) with (9,5). This ramp gives equal intensities,
    // but truncating the negative y offset incorrectly makes the first sample darker
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            image.at<uint8_t>(y, x) = std::clamp(128 + 8 * (x - 32) - (y - 32), 0, 255);
        }
    }
    QuadTreeNode tree(0, 64, 0, 64, {KeyPoint(32, 32)});
    describe(image, tree);
    EXPECT_EQ(tree.getFeatures()[0]->getDescriptor()[0] & 1, 0);
}

TEST_F(ORBFeatureDetectorTest, RotatedDescriptorsStayInsideImageAtRetainedBorder) {
    cv::Mat image(64, 64, CV_8UC1);
    cv::randu(image, 0, 256);
    for (int cornerX : {19, 44}) {
        for (int cornerY : {19, 44}) {
            for (int rotation = 0; rotation < 8; ++rotation) {
                QuadTreeNode tree(0, 64, 0, 64,
                    {KeyPoint(cornerX, cornerY, 0, 0, rotation * std::numbers::pi / 4)});
                describe(image, tree);
                EXPECT_EQ(tree.getFeatures().size(), 1);
            }
        }
    }
}

TEST_F(ORBFeatureDetectorTest, DetectionExcludesUnsafeDescriptorBorder) {
    cv::Mat image(64, 64, CV_8UC1);
    cv::randu(image, 0, 256);
    const auto tree = detect(image);
    const auto points = tree->getFeatures();
    ASSERT_FALSE(points.empty());
    EXPECT_TRUE(std::all_of(points.begin(), points.end(), [](const auto& point) {
        return point->getImgX() >= 19 && point->getImgX() < 45 &&
               point->getImgY() >= 19 && point->getImgY() < 45;
    }));
}
