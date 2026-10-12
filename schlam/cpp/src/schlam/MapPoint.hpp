//
// Created by baldhat on 10/11/26.
//

#ifndef SCHLAM_MAPPOINT_H
#define SCHLAM_MAPPOINT_H

#include <array>
#include <cstddef>
#include <deque>
#include <unordered_map>

#include <eigen3/Eigen/Core>

#include "MapTypes.h"


class MapPoint {
public:
    using Descriptor = std::array<std::uint8_t, 32>;

    MapPoint(const Eigen::Vector3d& position, const Descriptor& descriptor, const MapPointId id);

    void addInlierObservation(const Descriptor& descriptor);

    Eigen::Vector3d mPosition;
    // descriptor with the lowest median Hamming distance to the history
    Descriptor mDescriptor;
    Descriptor mLastInlierDescriptor;
    MapPointId mId;
    std::unordered_map<KeyFrameId, std::size_t> mObservations;

private:
    static constexpr std::size_t descriptorHistorySize = 10;
    std::deque<Descriptor> mDescriptorHistory;
};


#endif //SCHLAM_MAPPOINT_H
