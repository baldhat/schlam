//
// Created by baldhat on 10/11/26.
//

#include "MapPoint.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
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

MapPoint::MapPoint(const Eigen::Vector3d& position, const Descriptor& descriptor, const MapPointId id, const KeyFrameId creatorId)
    : mPosition(position), mDescriptor(descriptor), mLastInlierDescriptor(descriptor),
      mDescriptorHistory{descriptor}, mId(id), mCreatorId(creatorId){
}

void MapPoint::addInlierObservation(const Descriptor& descriptor, const Eigen::Vector3d& cameraCenter) {
    mNumInlierObservations++;
    addViewingObservation(cameraCenter);
    updateDescriptor(descriptor);
}

void MapPoint::addTriangulationObservation(const Descriptor& descriptor) {
    updateDescriptor(descriptor);
}

void MapPoint::addViewingObservation(const Eigen::Vector3d& cameraCenter) {
    const Eigen::Vector3d direction = cameraCenter - mPosition;
    const double length = direction.norm();
    if (std::isfinite(length) && length > 1e-9)
        mViewingDirectionSum += direction / length;
}

bool MapPoint::isViewDirectionCompatible(const Eigen::Vector3d& cameraCenter) const {
    const double meanLength = mViewingDirectionSum.norm();
    if (!std::isfinite(meanLength) || meanLength <= 1e-9) return true;

    const Eigen::Vector3d direction = cameraCenter - mPosition;
    const double length = direction.norm();
    if (!std::isfinite(length) || length <= 1e-9) return false;
    constexpr double cosMaxViewingAngle = 0.5; // 60 degrees
    return direction.dot(mViewingDirectionSum) >=
           cosMaxViewingAngle * length * meanLength;
}

void MapPoint::updateDescriptor(const Descriptor& descriptor) {
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
