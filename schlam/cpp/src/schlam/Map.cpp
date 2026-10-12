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
    const auto pointId = nextMapPointId;
    mPoints.insert({pointId, MapPoint(point, descriptor, pointId)});
    nextMapPointId++;
    return pointId;
}

KeyFrameId Map::addKeyFrame(const std::shared_ptr<Frame>& frame) {
    std::lock_guard guard(mKeyFramesMutex);
    const auto keyFrameId = nextKeyFrameId;
    mKeyFrames.insert({keyFrameId, KeyFrame(keyFrameId, *frame)});
    nextKeyFrameId++;
    std::cout << "[Map]: Added new keyframe with id " << keyFrameId << std::endl;
    return keyFrameId;
}

std::vector<KeyFrameSnapshot> Map::getKeyFrameSnapshots() const {
    std::lock_guard guard(mKeyFramesMutex);
    std::vector<KeyFrameSnapshot> snapshots;
    snapshots.reserve(mKeyFrames.size());
    for (const auto& [id, keyFrame] : mKeyFrames) {
        snapshots.push_back({id, keyFrame.mIntrinsics, keyFrame.mImageData, keyFrame.mMapToCamera});
    }
    return snapshots;
}

void Map::addObservation(KeyFrameId keyFrameId, std::size_t featureIndex, MapPointId pointId) {
    auto& point = mPoints.at(pointId);
    const auto& feature = mKeyFrames.at(keyFrameId).mKeyPoints.at(featureIndex);
    point.mObservations.insert_or_assign(keyFrameId, featureIndex);
    feature->setObservation(pointId);
}


bool Map::decideNewKeyframe(const std::uint32_t numInliers, const std::uint32_t framesSinceLastKeyframe) {
    return numInliers >= 50 && framesSinceLastKeyframe >= 15;
}
