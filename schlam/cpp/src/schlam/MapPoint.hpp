//
// Created by baldhat on 10/11/26.
//

#ifndef SCHLAM_MAPPOINT_H
#define SCHLAM_MAPPOINT_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>

#include <eigen3/Eigen/Core>


class MapPoint {
public:
    using Descriptor = std::array<std::uint8_t, 32>;

    MapPoint(Eigen::Vector3d position, Descriptor descriptor);

    void addInlierObservation(const Descriptor& descriptor);

    Eigen::Vector3d mPosition;
    // Observed descriptor with the lowest median Hamming distance to the history.
    Descriptor mDescriptor;
    Descriptor mLastInlierDescriptor;

private:
    static constexpr std::size_t descriptorHistorySize = 10;
    // Store descriptor copies rather than retaining frames and their images.
    std::deque<Descriptor> mDescriptorHistory;
};


#endif //SCHLAM_MAPPOINT_H
