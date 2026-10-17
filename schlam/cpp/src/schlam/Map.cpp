//
// Created by baldhat on 10/11/26.
//

#include "Map.h"

#include "KeyPoint.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>


Map::Map(const std::string& frameName) : mFrame(frameName) {}

std::vector<MapPointId> Map::addMapPoints(const std::vector<Eigen::Vector3d>& points3d,
        const std::vector<std::shared_ptr<KeyPoint>>& keyPoints, const KeyFrameId creatorId) {
    if (points3d.size() != keyPoints.size() ) {
        throw std::runtime_error("[Map::addMapPoints] 3D points and Keypoints need to be the same number!");
    }
    std::vector<MapPointId> pointIds;
    pointIds.reserve(points3d.size());
    for (std::size_t i = 0; i < points3d.size(); ++i) {
        pointIds.push_back(addMapPoint(points3d[i], keyPoints[i], creatorId));
    }
    return pointIds;
}
MapPointId Map::addMapPoint(const Eigen::Vector3d& point, const std::shared_ptr<KeyPoint>& keyPoint, const KeyFrameId creatorId) {
    auto descriptor = keyPoint->getDescriptor();
    std::lock_guard guard(mMutex);
    const auto pointId = nextMapPointId;
    mPoints.insert({pointId, MapPoint(point, descriptor, pointId, creatorId)});
    nextMapPointId++;
    return pointId;
}

KeyFrameId Map::addKeyFrame(const std::shared_ptr<Frame>& frame) {
    std::lock_guard guard(mMutex);
    const auto keyFrameId = nextKeyFrameId;
    mKeyFrames.insert({keyFrameId, KeyFrame(keyFrameId, *frame)});
    nextKeyFrameId++;
    std::cout << "[Map]: Added new keyframe with id " << keyFrameId << std::endl;
    return keyFrameId;
}

MapSnapshot Map::getSnapshot() const {
    std::lock_guard guard(mMutex);
    MapSnapshot snapshot;
    snapshot.points.reserve(mPoints.size());
    for (const auto& [id, point] : mPoints) {
        snapshot.points.push_back(point.mPosition);
    }
    snapshot.keyFrames.reserve(mKeyFrames.size());
    for (const auto& [id, keyFrame] : mKeyFrames) {
        snapshot.keyFrames.push_back({id, keyFrame.mIntrinsics, keyFrame.mImageData, *keyFrame.mMapToCamera});
    }
    return snapshot;
}

void Map::updateMapPointPosition(MapPointId pointId, const Eigen::Vector3d& position) {
    std::lock_guard guard(mMutex);
    mPoints.at(pointId).mPosition = position;
}

void Map::updateKeyFramePose(KeyFrameId keyFrameId, const tft::RigidTransform3D& mapToCamera) {
    if (mapToCamera.mSource != mFrame) {
        throw std::invalid_argument("Keyframe pose must use the map coordinate frame");
    }
    std::lock_guard guard(mMutex);
    mKeyFrames.at(keyFrameId).mMapToCamera = std::make_shared<tft::RigidTransform3D>(mapToCamera);
}

void Map::addObservation(KeyFrameId keyFrameId, std::size_t featureIndex, MapPointId pointId) {
    std::lock_guard guard(mMutex);
    auto& keyframe = mKeyFrames.at(keyFrameId);
    auto& point = mPoints.at(pointId);
    const auto& feature = keyframe.mKeyPoints.at(featureIndex);
    const auto& existingPoint = keyframe.mObservedPointByFeature.at(featureIndex);
    const auto existingObservation = point.mObservations.find(keyFrameId);
    if (existingPoint && *existingPoint != pointId) {
        throw std::invalid_argument("Feature is already assigned to another map point");
    }
    if (existingObservation != point.mObservations.end()) {
        if (existingObservation->second != featureIndex) {
            throw std::invalid_argument("Map point is already observed at another feature");
        }
        return;
    }

    for (const auto& [otherId, _] : point.mObservations) {
        auto& other = mKeyFrames.at(otherId);
        ++other.mSharedPointCount[keyFrameId];
        ++keyframe.mSharedPointCount[otherId];
    }
    point.mObservations.emplace(keyFrameId, featureIndex);
    keyframe.mObservedPoints.push_back(pointId);
    keyframe.mObservedPointByFeature[featureIndex] = pointId;
    feature->setObservation(pointId);
}

bool Map::removePoint(MapPointId pointId) {
    std::lock_guard guard(mMutex);
    const auto pointIt = mPoints.find(pointId);
    if (pointIt == mPoints.end()) return false;

    const auto& observations = pointIt->second.mObservations;
    const auto decrement = [](auto& counts, KeyFrameId otherId) {
        const auto it = counts.find(otherId);
        if (it == counts.end()) return;
        if (it->second <= 1) counts.erase(it);
        else --it->second;
    };
    for (auto first = observations.begin(); first != observations.end(); ++first) {
        for (auto second = std::next(first); second != observations.end(); ++second) {
            auto& firstCounts = mKeyFrames.at(first->first).mSharedPointCount;
            auto& secondCounts = mKeyFrames.at(second->first).mSharedPointCount;
            decrement(firstCounts, second->first);
            decrement(secondCounts, first->first);
        }
    }

    for (const auto& [frameId, featureIndex] : observations) {
        auto& frame = mKeyFrames.at(frameId);
        auto& observedPoint = frame.mObservedPointByFeature.at(featureIndex);
        if (observedPoint && *observedPoint == pointId) {
            observedPoint.reset();
            frame.mKeyPoints.at(featureIndex)->clearObservation();
        }
        auto& pointIds = frame.mObservedPoints;
        pointIds.erase(std::remove(pointIds.begin(), pointIds.end(), pointId), pointIds.end());
    }

    mPoints.erase(pointIt);
    return true;
}


bool Map::decideNewKeyframe(const std::uint32_t numInliers, const std::uint32_t framesSinceLastKeyframe) {
    // TODO instead of number of frames use % of tracked points of last key frame
    return numInliers >= 30 && framesSinceLastKeyframe >= 9;
}

void Map::cullMapPoints() {
    constexpr std::uint64_t minVisibilityCount = 8;
    constexpr std::uint64_t minSuccessRateDenominator = 4;
    std::vector<MapPointId> toRemove;
    {
        std::lock_guard guard(mMutex);
        for (const auto& [id, point] : mPoints) {
            if (point.mVisibleInTracking >= minVisibilityCount &&
                point.mNumInlierObservations * minSuccessRateDenominator < point.mVisibleInTracking) {
                toRemove.push_back(id);
            }
        }
    }
    for (const auto pointId : toRemove) removePoint(pointId);
}
