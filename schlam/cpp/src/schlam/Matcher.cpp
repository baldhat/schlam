//
// Created by baldhat on 2/8/26.
//

#include "Matcher.h"
#include "Frame.h"
#include "MapPoint.hpp"
#include "helpers/utils.h"
#include "tft/rigid_transform_3d.hpp"
#include "Map.h"
#include "Ransac.h"
#include "tft/transformer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <unordered_set>
#include <tracy/Tracy.hpp>

#include "KeyFrame.h"


namespace {

Eigen::Vector3d cameraCenter(const tft::RigidTransform3D& mapToCamera) {
    return -mapToCamera.mRotation.transpose() * mapToCamera.mTranslation;
}

std::optional<Eigen::Vector2d> projectIntoFrame(const Eigen::Vector3d& pointInCamera, const Frame& frame) {
    const auto projected = project(pointInCamera, frame.mIntrinsics);
    if (!projected || projected->x() < 0 || projected->x() >= frame.mImage.cols ||
        projected->y() < 0 || projected->y() >= frame.mImage.rows)
        return std::nullopt;
    return projected;
}
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
    ZoneScoped;
    std::vector<std::array<std::uint32_t, 2> > matches;
    std::vector<bool> matchedIndices2(keypoints2.size(), false);

    for (std::uint32_t j = 0; j < keypoints1.size(); ++j) {
        const auto descriptor = keypoints1[j]->getDescriptor();
        std::uint32_t bestMatchIndex = -1;
        double bestMatchValue = std::numeric_limits<double>::max();

        for (std::uint32_t i = 0; i < keypoints2.size(); i++) {
            if (matchedIndices2[i]) continue;

            auto distance = descriptor->getDistance(keypoints2[i]->getDescriptor());
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
    ZoneScoped;
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

            auto distance = descriptor->getDistance(candidate->getDescriptor());
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

void recordMapPointVisibility(TrackingSnapshot& snapshot, const Frame& frame,
                              const tft::RigidTransform3D& mapToCamera) {
    const auto center = cameraCenter(mapToCamera);
    for (auto& [pointId, point] : snapshot.mapPoints) {
        auto pointInCamera = mapToCamera.mRotation * point.mPosition + mapToCamera.mTranslation;
        if (point.isViewDirectionCompatible(center) &&
            projectIntoFrame(pointInCamera, frame))
            snapshot.visibleInTracking.insert_or_assign(pointId, ++point.mVisibleInTracking);
    }
}

std::vector<MapPointMatch>
matchMapPointsByProjection(TrackingSnapshot& snapshot,
                          const std::shared_ptr<Frame>& frame,
                          const std::shared_ptr<tft::RigidTransform3D>& estimatedPose,
                          const std::shared_ptr<tft::Transformer>& aTransformer,
                          const double aMaxDistance, const double aWindowSize) {
    ZoneScoped;
    const auto mapToCamera = aTransformer->findTransform(snapshot.frame, estimatedPose->mSource);
    std::vector<MapPointMatch> matches;
    matches.reserve(snapshot.mapPoints.size());
    std::unordered_set<std::shared_ptr<KeyPoint>> matchedTargets;
    std::vector<std::shared_ptr<KeyPoint>> candidates;
    const auto center = cameraCenter(*mapToCamera);
    for (const auto& [pointId, mapPoint] : snapshot.mapPoints) {
        if (!mapPoint.isViewDirectionCompatible(center)) continue;

        auto pointInCamera = mapToCamera->mRotation * mapPoint.mPosition + mapToCamera->mTranslation;
        const auto projected = projectIntoFrame(pointInCamera, *frame);
        if (!projected) continue;

        candidates.clear();
        frame->mKeypointTree->getFeatures(
            projected->x() - aWindowSize, projected->x() + aWindowSize,
            projected->y() - aWindowSize, projected->y() + aWindowSize, candidates);
        const auto findMatch = [&](const DescriptorPtr& descriptor) {
            std::shared_ptr<KeyPoint> bestMatch;
            double bestDistance = aMaxDistance;
            for (const auto& candidate : candidates) {
                if (matchedTargets.contains(candidate)) continue;
                const auto distance = descriptor->getDistance(candidate->getDescriptor());
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
        matches.push_back({pointId, feature, projectedFeature});
        matchedTargets.insert(feature);
    }
    return matches;
}

namespace {

struct EpipolarSearch {
    Eigen::Matrix3d essential;
    Eigen::Matrix3d inverseIntrinsics;
    double inverseSigmaSquared;
    double threshold;

    Eigen::Vector3d normalize(const KeyPoint& point) const {
        return inverseIntrinsics * Eigen::Vector3d(point.getImgX(), point.getImgY(), 1.0);
    }
};

template<class Score, class Distance>
std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matchKeyFramePair(
    const KeyFrame& frame1, const KeyFrame& frame2, Score cutoff, const EpipolarSearch* epipolar,
    const Distance& distance) {
    std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matches;
    std::vector<std::size_t> sources;
    sources.reserve(frame1.mKeyPoints.size());
    for (std::size_t i = 0; i < frame1.mKeyPoints.size(); ++i) {
        if (!frame1.mKeyPoints[i]->getMatched()) sources.push_back(i);
    }
    std::vector<std::uint8_t> eligible;
    eligible.reserve(frame2.mKeyPoints.size());
    for (const auto& point : frame2.mKeyPoints) eligible.push_back(!point->getMatched());
    auto remainingTargets = std::count(eligible.begin(), eligible.end(), std::uint8_t{1});
    if (remainingTargets == 0) return matches;
    std::vector<const KeyPoint*> candidates;
    const auto targetCount = frame2.mKeyPoints.size();

    for (std::size_t source : sources) {
        auto bestTarget = targetCount;
        Score bestDistance = cutoff;
        const auto consider = [&](std::size_t target, const auto& passesGeometry) {
            if (!eligible[target]) return false;
            const auto value = distance(source, target, bestDistance);
            if (!(value < bestDistance) || !passesGeometry(target)) return false;
            bestTarget = target;
            bestDistance = value;
            return std::is_integral_v<Score> && bestDistance == 0;
        };

        if (epipolar) {
            const Eigen::Vector3d p1 = epipolar->normalize(*frame1.mKeyPoints[source]);
            const Eigen::Vector3d line2 = epipolar->essential * p1;
            const double normalSquared = line2.head<2>().squaredNorm();
            if (!(normalSquared >= 1e-9) || !std::isfinite(normalSquared)) continue;
            const Eigen::Vector3d pixelLine = epipolar->inverseIntrinsics.transpose() * line2;
            // Keep a half-pixel margin in the tree search. Apply the same symmetric check as triangulation below.
            const double maxResidual = std::sqrt(epipolar->threshold * (normalSquared / epipolar->inverseSigmaSquared)) +
                0.5 * pixelLine.head<2>().norm();
            if (!pixelLine.allFinite() || !std::isfinite(maxResidual)) continue;

            candidates.clear();
            frame2.mKeypointTree->getFeaturesInBand({pixelLine.x(), pixelLine.y(), pixelLine.z()}, maxResidual, candidates);
            const auto passesGeometry = [&](std::size_t target) {
                const Eigen::Vector3d p2 = epipolar->normalize(*frame2.mKeyPoints[target]);
                const Eigen::Vector3d line1 = epipolar->essential.transpose() * p2;
                return calculateSymmetricErrorEssential(line2, p2, epipolar->inverseSigmaSquared) <= epipolar->threshold &&
                       calculateSymmetricErrorEssential(line1, p1, epipolar->inverseSigmaSquared) <= epipolar->threshold;
            };
            for (const auto* candidate : candidates) {
                const auto index = candidate->getIndex();
                if (index && *index < targetCount && consider(*index, passesGeometry)) break;
            }
        } else {
            for (std::size_t target = 0; target < targetCount; ++target) {
                if (consider(target, [](std::size_t) { return true; })) break;
            }
        }
        if (bestTarget != targetCount) {
            matches[0].push_back(frame1.mKeyPoints[source]);
            matches[1].push_back(frame2.mKeyPoints[bestTarget]);
            eligible[bestTarget] = false;
            if (--remainingTargets == 0) break;
        }
    }
    return matches;
}

std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matchUnmatchedFeatures(
    KeyFrame& frame1, KeyFrame& frame2, double maxDistance, const EpipolarSearch* epipolar) {
    const auto& descriptors1 = frame1.packedOrbDescriptors();
    const auto& descriptors2 = frame2.packedOrbDescriptors();
    if (descriptors1.size() != frame1.mKeyPoints.size() || descriptors2.size() != frame2.mKeyPoints.size()) {
        if (!epipolar) return matchGlobally(frame1.getUnmatchedFeatures(), frame2.getUnmatchedFeatures(), maxDistance);
        return matchKeyFramePair(frame1, frame2, std::min(maxDistance, std::numeric_limits<double>::max()), epipolar,
            [&](std::size_t source, std::size_t target, double) {
                return frame1.mKeyPoints[source]->getDescriptor()->getDistance(frame2.mKeyPoints[target]->getDescriptor());
            });
    }

    if (!(maxDistance > 0)) return {};
    // Hamming distances are integers between 0 and 256. Keep the strict floating-point cutoff.
    const int cutoff = static_cast<int>(std::ceil(std::min(maxDistance, 257.0)));
    return matchKeyFramePair(frame1, frame2, cutoff, epipolar, [&](std::size_t source, std::size_t target, int limit) {
        return OrbDescriptor::boundedDistance(descriptors1[source], descriptors2[target], limit);
    });
}

} // namespace

std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matchUnmatchedFeatures(KeyFrame& frame1, KeyFrame& frame2, double maxDistance) {
    ZoneScoped;
    return matchUnmatchedFeatures(frame1, frame2, maxDistance, nullptr);
}

std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matchUnmatchedFeatures(
    KeyFrame& source, KeyFrame& target, double maxDistance,
    const tft::RigidTransform3D& sourceToTarget, double essentialChi2) {
    ZoneScoped;
    const auto essential = triangulationEssentialMatrix(sourceToTarget);
    if (!essential || !essential->allFinite()) return {};
    if (essentialChi2 == std::numeric_limits<double>::infinity()) {
        return matchUnmatchedFeatures(source, target, maxDistance, nullptr);
    }
    // Keep the normalization used by Reconstructor::triangulate for both views.
    const Eigen::Matrix3d inverseIntrinsics = target.mIntrinsics.inverse();
    const double sigma = 1.0 / target.mIntrinsics(0, 0);
    const double inverseSigmaSquared = 1.0 / (sigma * sigma);
    if (!inverseIntrinsics.allFinite() ||
        !(inverseSigmaSquared > 0) || !std::isfinite(inverseSigmaSquared) ||
        essentialChi2 < 0 || !std::isfinite(essentialChi2)) return {};
    const EpipolarSearch epipolar{*essential, inverseIntrinsics, inverseSigmaSquared, essentialChi2};
    return matchUnmatchedFeatures(source, target, maxDistance, &epipolar);
}

