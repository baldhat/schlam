//
// Created by baldhat on 10/11/26.
//

#include "Map.h"

#include "KeyPoint.h"

#include <stdexcept>


Map::Map(const std::string& frameName) : mFrame(frameName) {}

std::vector<MapPointId> Map::addMapPoints(const std::vector<Eigen::Vector3d>& points3d,
        const std::vector<std::shared_ptr<KeyPoint>>& keyPoints) {
    if (points3d.size() != keyPoints.size() ) {
        throw std::runtime_error("[Map::addMapPoints] 3D points and Keypoints need to be the same number!");
    }
    std::vector<MapPointId> pointIds;
    pointIds.reserve(points3d.size());
    for (std::size_t i = 0; i < points3d.size(); ++i) {
        pointIds.push_back(addMapPoint(points3d[i], keyPoints[i]));
    }
    return pointIds;
}
MapPointId Map::addMapPoint(const Eigen::Vector3d& point, const std::shared_ptr<KeyPoint>& keyPoint) {
    auto descriptor = keyPoint->getDescriptor();
    std::lock_guard guard(mMutex);
    const auto pointId = nextMapPointId;
    mPoints.insert({pointId, MapPoint(point, descriptor, pointId)});
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
    auto& point = mPoints.at(pointId);
    const auto& feature = mKeyFrames.at(keyFrameId).mKeyPoints.at(featureIndex);
    point.mObservations.insert_or_assign(keyFrameId, featureIndex);
    feature->setObservation(pointId);
}


bool Map::decideNewKeyframe(const std::uint32_t numInliers, const std::uint32_t framesSinceLastKeyframe) {
    // TODO instead of number of frames use % of tracked points of last key frame
    return numInliers >= 50 && framesSinceLastKeyframe >= 10;
}
