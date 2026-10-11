//
// Created by baldhat on 2/8/26.
//

#include "Matcher.h"

#include <vector>
#include <array>
#include <cstdint>
#include <limits>
#include <unordered_set>

#include "Frame.h"
#include "se3_ops.h"
#include "utils.h"
#include "tft/rigid_transform_3d.hpp"
#include "tft/transformer.hpp"


inline int hamming256(const uint8_t *a, const uint8_t *b) {
    const uint64_t *a64 = reinterpret_cast<const uint64_t *>(a);
    const uint64_t *b64 = reinterpret_cast<const uint64_t *>(b);

    return __builtin_popcountll(a64[0] ^ b64[0]) +
           __builtin_popcountll(a64[1] ^ b64[1]) +
           __builtin_popcountll(a64[2] ^ b64[2]) +
           __builtin_popcountll(a64[3] ^ b64[3]);
}

std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matchGlobally(const std::shared_ptr<QuadTreeNode>& aTree1,
                                                     const std::shared_ptr<QuadTreeNode>& aTree2,
                                                     const double aMaxDistance) {
    auto keypoints1 = aTree1->getFeatures();
    auto keypoints2 = aTree2->getFeatures();
    std::vector<std::array<std::uint32_t, 2> > matches;
    std::vector<bool> matchedIndices2(keypoints2.size(), false);

    for (std::uint32_t j = 0; j < keypoints1.size(); ++j) {
        auto descriptor = keypoints1[j]->getDescriptor();
        std::uint32_t bestMatchIndex = -1;
        double bestMatchValue = std::numeric_limits<double>::max();

        for (std::uint32_t i = 0; i < keypoints2.size(); i++) {
            if (matchedIndices2[i]) continue;

            auto distance = hamming256(descriptor.data(), keypoints2[i]->getDescriptor().data());
            if (distance < bestMatchValue) {
                bestMatchIndex = i;
                bestMatchValue = distance;
            }
        }

        if (bestMatchIndex != -1 && bestMatchValue < aMaxDistance) {
            matches.push_back({j, bestMatchIndex});
            matchedIndices2[bestMatchIndex] = true;
        }
    }
    return getMatched(keypoints1, keypoints2, matches);
}

std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> getMatched(const std::vector<std::shared_ptr<KeyPoint>>& aKeypoints1,
                                                  const std::vector<std::shared_ptr<KeyPoint>>& aKeypoints2,
                                                  const std::vector<std::array<std::uint32_t, 2>>& aMatches) {
    std::vector<std::shared_ptr<KeyPoint>> out1, out2;
    for (const auto &[id1, id2]: aMatches) {
        out1.push_back(aKeypoints1[id1]);
        out2.push_back(aKeypoints2[id2]);
    }
    return {out1, out2};
}

std::tuple<std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>, std::vector<bool>>
matchWindow(const std::vector<std::shared_ptr<KeyPoint>>& aKeypoints,
            const std::shared_ptr<QuadTreeNode>& aTree2,
            const double aMaxDistance, const double aWindowSize) {
    std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matches;
    std::vector<bool> matched(aKeypoints.size());
    std::unordered_set<std::shared_ptr<KeyPoint>> matchedTargets;

    for (std::size_t featureIndex = 0; featureIndex < aKeypoints.size(); ++featureIndex) {
        const auto& feature = aKeypoints[featureIndex];
        const auto descriptor = feature->getDescriptor();
        std::shared_ptr<KeyPoint> bestMatch;
        double bestMatchValue = std::numeric_limits<double>::max();

        const auto x{feature->getImgX()}, y{feature->getImgY()};
        const auto x1{x - aWindowSize}, x2{x + aWindowSize}, y1{y - aWindowSize}, y2{y + aWindowSize};
        const auto candidates = aTree2->getFeatures(x1, x2, y1, y2);

        for (const auto& candidate: candidates) {
            if (matchedTargets.contains(candidate)) continue;

            const auto distance = hamming256(descriptor.data(), candidate->getDescriptor().data());
            if (distance < bestMatchValue) {
                bestMatch = candidate;
                bestMatchValue = distance;
            }
        }

        if (bestMatch && bestMatchValue < aMaxDistance) {
            matches[0].push_back(feature);
            matches[1].push_back(bestMatch);
            matchedTargets.insert(bestMatch);
            matched[featureIndex] = true;
        }
    }
    return {matches, matched};
}

std::tuple<std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>, std::vector<bool>>
matchWindow(const std::shared_ptr<QuadTreeNode>& aTree1,
                                                   const std::shared_ptr<QuadTreeNode>& aTree2,
                                                   const double aMaxDistance, const double aWindowSize) {
    const auto features1 = aTree1->getFeatures();
    return matchWindow(features1, aTree2, aMaxDistance, aWindowSize);
}

std::tuple<std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>, std::vector<bool>>
matchByProjection(const std::shared_ptr<Map> map,
                  std::shared_ptr<Frame> frame,
                  std::shared_ptr<tft::RigidTransform3D> estimatedPose,
                  std::shared_ptr<tft::Transformer> aTransformer,
                  const double aMaxDistance, const double aWindowSize) {
    // project map points into frame
    const auto mapToCamera = aTransformer->findTransform(map->mFrame, estimatedPose->mSource);
    std::vector<std::shared_ptr<KeyPoint>> projectedMapKeypoints;
    std::vector<std::size_t> projectedMapIndices;
    projectedMapKeypoints.reserve(map->mPoints.size());
    projectedMapIndices.reserve(map->mPoints.size());
    for (std::size_t mapIndex = 0; mapIndex < map->mPoints.size(); ++mapIndex) {
        const auto& mapPoint = map->mPoints[mapIndex];
        auto pointInCamera = mapToCamera->apply(mapPoint.mPosition);
        auto projected = project(pointInCamera, frame->mIntrinsics);
        if (projected.has_value()
            && projected->x() >= 0 && projected->x() < frame->mImage.cols
            && projected->y() >= 0 && projected->y() < frame->mImage.rows) {
            projectedMapKeypoints.push_back(std::make_shared<KeyPoint>(projected->x(), projected->y(), 0, 0, 0, mapPoint.mDescriptor));
            projectedMapIndices.push_back(mapIndex);
        }
    }

    // find matches by window search
    auto [matches, projectedMatched] = matchWindow(projectedMapKeypoints, frame->mKeypointTree, aMaxDistance, aWindowSize);
    std::vector<bool> matched(map->mPoints.size(), false);
    for (std::size_t j = 0; j < projectedMatched.size(); ++j) {
        matched[projectedMapIndices[j]] = projectedMatched[j];
    }
    return {matches, matched};
}
