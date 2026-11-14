//
// Created by baldhat on 10/11/26.
//

#ifndef SCHLAM_MAPPOINT_H
#define SCHLAM_MAPPOINT_H

#include <cstddef>
#include <deque>
#include <unordered_map>

#include <Eigen/Core>

#include "MapTypes.h"
#include "config/Config.hpp"
#include "feature_detection/IDescriptor.h"


class MapPoint {
public:
    MapPoint(const Eigen::Vector3d& position, const DescriptorPtr& descriptor, const MapPointId id, const KeyFrameId creatorId,
             schlam::config::MapPoints settings = {});

    void addInlierObservation(const DescriptorPtr& descriptor, const Eigen::Vector3d& cameraCenter);
    void addTriangulationObservation(const DescriptorPtr& descriptor);
    void addViewingObservation(const Eigen::Vector3d& cameraCenter);
    bool isViewDirectionCompatible(const Eigen::Vector3d& cameraCenter) const;

    Eigen::Vector3d mPosition;
    // descriptor with the lowest median Hamming distance to the history
    DescriptorPtr mDescriptor;
    DescriptorPtr mLastInlierDescriptor;
    MapPointId mId;
    std::unordered_map<KeyFrameId, std::size_t> mObservations;
    KeyFrameId mCreatorId;
    std::uint64_t mVisibleInTracking{0};
    // Accepted tracking observations after creation; triangulation is excluded.
    std::uint64_t mNumInlierObservations{0};

private:
    void updateDescriptor(const DescriptorPtr& descriptor);
    Eigen::Vector3d mViewingDirectionSum = Eigen::Vector3d::Zero();
    schlam::config::MapPoints mSettings;
    std::deque<DescriptorPtr> mDescriptorHistory;
};


#endif //SCHLAM_MAPPOINT_H
