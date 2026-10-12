//
// Created by baldhat on 10/11/26.
//

#ifndef SCHLAM_MAP_H
#define SCHLAM_MAP_H
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "KeyFrame.h"
#include "MapPoint.hpp"
#include "MapTypes.h"

class KeyPoint;

struct KeyFrameSnapshot {
    KeyFrameId id;
    Eigen::Matrix3d intrinsics;
    cv::Mat imageData;
    tft::RigidTransform3D mapToCamera;
};

class Map {
public:
    Map(const std::string& frameName);
    std::vector<MapPointId> addMapPoints(const std::vector<Eigen::Vector3d>& points3d,
        const std::vector<std::shared_ptr<KeyPoint>>& keyPoints);
    MapPointId addMapPoint(const Eigen::Vector3d& point, const std::shared_ptr<KeyPoint>& keyPoint);

    KeyFrameId addKeyFrame(const std::shared_ptr<Frame>& frame);
    std::vector<KeyFrameSnapshot> getKeyFrameSnapshots() const;
    void addObservation(KeyFrameId keyFrameId, std::size_t featureIndex, MapPointId pointId);

    bool decideNewKeyframe(const std::uint32_t numInliers, const std::uint32_t framesSinceLastKeyframe);

    std::string mFrame;
    std::unordered_map<MapPointId, MapPoint> mPoints;
    std::unordered_map<KeyFrameId, KeyFrame> mKeyFrames;

    MapPointId nextMapPointId{0};
    KeyFrameId nextKeyFrameId{0};

private:
    mutable std::mutex mKeyFramesMutex;
};


#endif //SCHLAM_MAP_H
