#include <gtest/gtest.h>
#include "schlam/feature_detection/xfeat/XFeatMatcher.h"
#include "schlam/feature_detection/xfeat/XFeatDescriptor.h"
#include "schlam/feature_detection/orb/OrbDescriptor.h"
#include "schlam/feature_detection/orb/QuadTreeNode.h"
#include "schlam/Matcher.h"
#include "schlam/Frame.h"
#include "schlam/KeyPoint.h"
#include "schlam/KeyFrame.h"
#include "schlam/Map.h"
#include "schlam/MapPoint.hpp"
#include "tft/transformer.hpp"

#include <cmath>
#include <random>

namespace {
DescriptorPtr descriptor(float x, float y) {
    XFeatDescriptor::Values values{};
    values[0] = x; values[1] = y;
    return std::make_shared<XFeatDescriptor>(values);
}
KeyPoint feature(double x, double y, const DescriptorPtr& d) { return KeyPoint(x,y,0,0,0,d); }
}

TEST(XFeatMatcherTest, RejectsNonMutualPairsRatherThanGreedilyAssigningAnotherTarget) {
    const auto exact = descriptor(1,0), close = descriptor(1,0.1f), other = descriptor(1,0.5f);
    const auto matches = matchXFeat({close,exact}, {exact,other}, 0.18);
    ASSERT_EQ(matches.size(), 1);
    EXPECT_EQ(matches[0], (std::array<std::size_t,2>{1,0}));
}

TEST(XFeatMatcherTest, UsesFirstIndexForTiesAndUpstreamThresholdSemantics) {
    const auto a = descriptor(1,0), b = descriptor(-1,0);
    const auto matches = matchXFeat({a,a}, {a,a}, 0.18);
    ASSERT_EQ(matches.size(), 1);
    EXPECT_EQ(matches[0], (std::array<std::size_t,2>{0,0}));
    EXPECT_TRUE(matchXFeat({a}, {b}, 0.18).empty());
    EXPECT_EQ(matchXFeat({a}, {b}, 1).size(), 1); // min_cossim=0 disables cutoff upstream.
    const auto boundary = descriptor(0.5f, std::sqrt(0.75f));
    EXPECT_TRUE(matchXFeat({a}, {boundary}, 0.5).empty()); // strict greater-than
    EXPECT_TRUE(matchXFeat({}, {a}, 0.18).empty());
    EXPECT_TRUE(matchXFeat({a}, {}, 0.18).empty());
    EXPECT_THROW(matchXFeat({a}, {nullptr}, 0.18), std::invalid_argument);
    EXPECT_THROW(matchXFeat({a}, {std::make_shared<OrbDescriptor>(TDescriptor{})}, 0.18), std::invalid_argument);
}

TEST(XFeatMatcherTest, MatchesScalarCosineMutualArgmaxAcrossMatrixBlocks) {
    std::mt19937 random(42);
    std::normal_distribution<float> normal;
    std::vector<DescriptorPtr> a, b;
    for (int i = 0; i < 270; ++i) {
        XFeatDescriptor::Values values;
        for (auto& value : values) value = normal(random);
        b.push_back(std::make_shared<XFeatDescriptor>(values));
    }
    for (int i = 0; i < 350; ++i) a.push_back(b[(i*17)%b.size()]);
    std::vector<std::array<std::size_t,2>> expected;
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::size_t best = 0;
        for (std::size_t j = 1; j < b.size(); ++j)
            if (a[i]->getDistance(b[j]) < a[i]->getDistance(b[best])) best = j;
        std::size_t reverse = 0;
        for (std::size_t j = 1; j < a.size(); ++j)
            if (b[best]->getDistance(a[j]) < b[best]->getDistance(a[reverse])) reverse = j;
        if (reverse == i && a[i]->getDistance(b[best]) < 0.18) expected.push_back({i,best});
    }
    EXPECT_EQ(matchXFeat(a,b,0.18), expected);
}

TEST(XFeatMatcherTest, InitializationUsesGlobalMutualMatchesAndPreservesSourceMask) {
    const auto exact = descriptor(1,0), close = descriptor(1,0.1f);
    auto source = std::make_shared<QuadTreeNode>(0,100,0,100,
        std::vector<KeyPoint>{feature(5,5,close), feature(6,6,exact)});
    auto target = std::make_shared<QuadTreeNode>(0,100,0,100,
        std::vector<KeyPoint>{feature(95,95,exact)});
    const auto [matches, mask] = matchWindow(source,target,0.18,1);
    ASSERT_EQ(matches[0].size(),1);
    EXPECT_EQ(matches[0][0]->getImgX(),6);
    EXPECT_EQ(matches[1][0]->getImgX(),95);
    EXPECT_EQ(mask,(std::vector<bool>{false,true}));
    const auto global = matchGlobally(source,target,0.18);
    EXPECT_EQ(global,matches);
}

TEST(XFeatMatcherTest, TrackingUsesGlobalMutualMatchingForVisibleMapPoints) {
    const auto exact = descriptor(1,0), close = descriptor(1,0.1f);
    Eigen::Matrix3d k;
    k << 100,0,50,0,100,50,0,0,1;
    cv::Mat image(100,100,CV_8UC1,cv::Scalar(0));
    auto frame = std::make_shared<Frame>(std::make_shared<ImageData>(utils::TTimestamp{},image,k,"current"));
    frame->mKeypointTree = std::make_shared<QuadTreeNode>(0,100,0,100,
        std::vector<KeyPoint>{feature(90,90,exact)});
    Map map("world");
    const auto weak = map.addMapPoint(Eigen::Vector3d(0,0,5),
        std::make_shared<KeyPoint>(feature(50,50,close)),0,utils::TTimestamp{});
    const auto best = map.addMapPoint(Eigen::Vector3d(0,0,6),
        std::make_shared<KeyPoint>(feature(50,50,exact)),0,utils::TTimestamp{});
    auto pose = std::make_shared<tft::RigidTransform3D>("current","world",Eigen::Matrix3d::Identity(),Eigen::Vector3d::Zero());
    auto transformer = std::make_shared<tft::Transformer>();
    transformer->registerTransform(pose);
    auto snapshot = map.getTrackingSnapshot();
    const auto matches = matchMapPointsByProjection(snapshot,frame,pose,transformer,0.18,1);
    ASSERT_EQ(matches.size(),1);
    EXPECT_EQ(matches[0].mapPointId,best);
    EXPECT_NE(matches[0].mapPointId,weak);
    EXPECT_EQ(matches[0].feature->getImgX(),90);
    EXPECT_EQ(matches[0].projectedFeature->getImgX(),50);
}
