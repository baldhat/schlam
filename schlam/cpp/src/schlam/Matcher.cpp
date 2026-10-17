//
// Created by baldhat on 2/8/26.
//

#include "Matcher.h"

#include <algorithm>
#include <vector>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_set>

#include "Frame.h"
#include "se3_ops.h"
#include "utils.h"
#include "tft/rigid_transform_3d.hpp"
#include "tft/transformer.hpp"

namespace {
std::vector<MapPointId> sortedMapPointIds(const Map& map) {
    std::vector<MapPointId> ids;
    ids.reserve(map.mPoints.size());
    for (const auto& [id, point] : map.mPoints) {
        ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

Eigen::Vector3d cameraCenter(const tft::RigidTransform3D& mapToCamera) {
    return -mapToCamera.mRotation.transpose() * mapToCamera.mTranslation;
}

std::optional<Eigen::Vector2d> projectIntoFrame(
    const MapPoint& point, const Frame& frame, const tft::RigidTransform3D& mapToCamera) {
    const Eigen::Vector3d pointInCamera =
        mapToCamera.mRotation * point.mPosition + mapToCamera.mTranslation;
    const auto projected = project(pointInCamera, frame.mIntrinsics);
    if (!projected || projected->x() < 0 || projected->x() >= frame.mImage.cols ||
        projected->y() < 0 || projected->y() >= frame.mImage.rows)
        return std::nullopt;
    return projected;
}
}

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
    return matchGlobally(keypoints1, keypoints2, aMaxDistance);
}

std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matchGlobally(const std::vector<std::shared_ptr<KeyPoint>>& keypoints1,
                                                     const std::vector<std::shared_ptr<KeyPoint>>& keypoints2,
                                                     const double aMaxDistance) {
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

void recordMapPointVisibility(Map* map, const Frame& frame,
                              const tft::RigidTransform3D& mapToCamera) {
    const auto center = cameraCenter(mapToCamera);
    for (auto& [pointId, point] : map->mPoints) {
        if (point.isViewDirectionCompatible(center) &&
            projectIntoFrame(point, frame, mapToCamera))
            ++point.mVisibleInTracking;
    }
}

std::vector<MapPointMatch>
matchMapPointsByProjection(Map* map,
                          const std::shared_ptr<Frame>& frame,
                          const std::shared_ptr<tft::RigidTransform3D>& estimatedPose,
                          const std::shared_ptr<tft::Transformer>& aTransformer,
                          const double aMaxDistance, const double aWindowSize) {
    const auto mapToCamera = aTransformer->findTransform(map->mFrame, estimatedPose->mSource);
    std::vector<MapPointMatch> matches;
    matches.reserve(map->mPoints.size());
    std::unordered_set<std::shared_ptr<KeyPoint>> matchedTargets;
    const auto center = cameraCenter(*mapToCamera);
    for (const auto mapPointId : sortedMapPointIds(*map)) {
        auto& mapPoint = map->mPoints.at(mapPointId);
        if (!mapPoint.isViewDirectionCompatible(center)) continue;
        const auto projected = projectIntoFrame(mapPoint, *frame, *mapToCamera);
        if (!projected) continue;

        const auto candidates = frame->mKeypointTree->getFeatures(
            projected->x() - aWindowSize, projected->x() + aWindowSize,
            projected->y() - aWindowSize, projected->y() + aWindowSize);
        const auto findMatch = [&](const MapPoint::Descriptor& descriptor) {
            std::shared_ptr<KeyPoint> bestMatch;
            double bestDistance = aMaxDistance;
            for (const auto& candidate : candidates) {
                if (matchedTargets.contains(candidate)) continue;
                const auto distance = hamming256(descriptor.data(), candidate->getDescriptor().data());
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestMatch = candidate;
                }
            }
            return bestMatch;
        };

        auto feature = findMatch(mapPoint.mLastInlierDescriptor);
        if (!feature && mapPoint.mLastInlierDescriptor != mapPoint.mDescriptor) {
            feature = findMatch(mapPoint.mDescriptor);
        }
        if (!feature) continue;

        auto projectedFeature = std::make_shared<KeyPoint>(
            projected->x(), projected->y(), 0, 0, 0, mapPoint.mDescriptor);
        matches.push_back({mapPointId, feature, projectedFeature});
        matchedTargets.insert(feature);
    }
    return matches;
}

std::tuple<std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>, std::vector<bool>>
matchByProjection(Map* map,
                  std::shared_ptr<Frame> frame,
                  std::shared_ptr<tft::RigidTransform3D> estimatedPose,
                  std::shared_ptr<tft::Transformer> aTransformer,
                  const double aMaxDistance, const double aWindowSize) {
    const auto associations = matchMapPointsByProjection(
        map, frame, estimatedPose, aTransformer, aMaxDistance, aWindowSize);
    std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matches;
    const auto pointIds = sortedMapPointIds(*map);
    std::vector<bool> matched(map->mPoints.size(), false);
    for (const auto& association : associations) {
        matches[0].push_back(association.projectedFeature);
        matches[1].push_back(association.feature);
        const auto index = std::lower_bound(pointIds.begin(), pointIds.end(), association.mapPointId)
            - pointIds.begin();
        matched[index] = true;
    }
    return {matches, matched};
}

std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matchUnmatchedFeatures(KeyFrame& frame1, KeyFrame& frame2, const double aMaxDistance) {
    auto unmatched1 = frame1.getUnmatchedFeatures();
    auto unmatched2 = frame2.getUnmatchedFeatures();
    auto matches = matchGlobally(unmatched1, unmatched2, aMaxDistance);
    return matches;
}

