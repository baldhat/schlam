//
// Created by baldhat on 10/11/26.
//

#ifndef SCHLAM_MAP_H
#define SCHLAM_MAP_H

#include "MapTypes.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <map>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include "KeyPoint.h"
#include "tft/rigid_transform_3d.hpp"

class MapPoint;
struct MapPointMatch;
class Frame;
class KeyFrame;


struct KeyFrameSnapshot {
    KeyFrameId id;
    Eigen::Matrix3d intrinsics;
    cv::Mat imageData;
    tft::RigidTransform3D mapToCamera;
};

struct MapSnapshot {
    std::vector<Eigen::Vector3d> points;
    std::vector<KeyFrameSnapshot> keyFrames;
};

struct Observation {
    Descriptor descriptor;
    Eigen::Vector3d acceptedCameraCenter;
};

struct TrackingSnapshot {
    std::map<MapPointId, MapPoint> mapPoints;
    std::map<MapPointId, Observation> newObservations;
    std::map<MapPointId, std::uint64_t> visibleInTracking;
    std::string frame;
};

class Map {
public:
    Map(const std::string& frameName);
    std::vector<MapPointId> addMapPoints(const std::vector<Eigen::Vector3d>& points3d,
        const std::vector<std::shared_ptr<KeyPoint>>& keyPoints, const KeyFrameId creatorId);
    MapPointId addMapPoint(const Eigen::Vector3d& point, const std::shared_ptr<KeyPoint>& keyPoint, const KeyFrameId creatorId);

    KeyFrameId addKeyFrame(const std::shared_ptr<Frame>& frame,
        const std::vector<MapPointMatch> matches,
        const std::vector<bool> inliers);
    KeyFrameId addKeyFrame(const std::shared_ptr<Frame>& frame);

    MapSnapshot getSnapshot() const;
    TrackingSnapshot getTrackingSnapshot() const;
    void updateMapPointPosition(MapPointId pointId, const Eigen::Vector3d& position);
    void updateKeyFramePose(KeyFrameId keyFrameId, const tft::RigidTransform3D& mapToCamera);
    void applyTrackingSnapshot(const TrackingSnapshot& snapshot);
    void addObservation(KeyFrameId keyFrameId, std::size_t featureIndex, MapPointId pointId);
    bool removePoint(MapPointId pointId);

    bool decideNewKeyframe(const std::uint32_t numInliers, const std::uint32_t framesSinceLastKeyframe, const bool localMappingIdle);
    void cullMapPoints();

    std::string mFrame;
    std::map<MapPointId, MapPoint> mPoints;
    std::map<KeyFrameId, KeyFrame> mKeyFrames;

    MapPointId nextMapPointId{0};
    KeyFrameId nextKeyFrameId{0};

private:
    mutable std::mutex mMutex;
};


#endif //SCHLAM_MAP_H
