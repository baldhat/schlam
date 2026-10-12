//
// Created by baldhat on 10/11/26.
//

#include "MapPoint.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <vector>

namespace {
int hammingDistance(const MapPoint::Descriptor& first, const MapPoint::Descriptor& second) {
    int distance = 0;
    for (std::size_t i = 0; i < first.size(); ++i) {
        distance += std::popcount(static_cast<unsigned int>(first[i] ^ second[i]));
    }
    return distance;
}
}

MapPoint::MapPoint(const Eigen::Vector3d& position, const Descriptor& descriptor, const MapPointId id)
    : mPosition(position), mDescriptor(descriptor), mLastInlierDescriptor(descriptor),
      mDescriptorHistory{descriptor}, mId(id) {
}

void MapPoint::addInlierObservation(const Descriptor& descriptor) {
    mLastInlierDescriptor = descriptor;
    mDescriptorHistory.push_back(descriptor);
    if (mDescriptorHistory.size() > descriptorHistorySize) {
        mDescriptorHistory.pop_front();
    }

    int bestMedian = std::numeric_limits<int>::max();
    std::size_t bestIndex = 0;
    const auto medianIndex = (mDescriptorHistory.size() - 1) / 2;
    std::vector<int> distances(mDescriptorHistory.size());
    for (std::size_t i = 0; i < mDescriptorHistory.size(); ++i) {
        for (std::size_t j = 0; j < mDescriptorHistory.size(); ++j) {
            distances[j] = hammingDistance(mDescriptorHistory[i], mDescriptorHistory[j]);
        }
        std::nth_element(distances.begin(), distances.begin() + medianIndex, distances.end());
        // Keep the earlier observation when medians tie to avoid needless changes.
        if (distances[medianIndex] < bestMedian) {
            bestMedian = distances[medianIndex];
            bestIndex = i;
        }
    }
    mDescriptor = mDescriptorHistory[bestIndex];
}
