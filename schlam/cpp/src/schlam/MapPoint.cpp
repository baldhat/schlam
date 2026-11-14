//
// Created by baldhat on 10/11/26.
//

#include "MapPoint.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>


MapPoint::MapPoint(const Eigen::Vector3d& position, const DescriptorPtr& descriptor, const MapPointId id, const KeyFrameId creatorId,
                   schlam::config::MapPoints settings)
    : mPosition(position), mDescriptor(descriptor), mLastInlierDescriptor(descriptor),
      mId(id), mCreatorId(creatorId), mSettings(settings), mDescriptorHistory{descriptor} {
}

void MapPoint::addInlierObservation(const DescriptorPtr& descriptor, const Eigen::Vector3d& cameraCenter) {
    mNumInlierObservations++;
    addViewingObservation(cameraCenter);
    updateDescriptor(descriptor);
}

void MapPoint::addTriangulationObservation(const DescriptorPtr& descriptor) {
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
    const double cosMaxViewingAngle = std::cos(mSettings.maximumViewingAngleDegrees * std::numbers::pi / 180.0);
    return direction.dot(mViewingDirectionSum) >=
           cosMaxViewingAngle * length * meanLength;
}

void MapPoint::updateDescriptor(const DescriptorPtr& descriptor) {
    mLastInlierDescriptor = descriptor;
    mDescriptorHistory.push_back(descriptor);
    if (mDescriptorHistory.size() > mSettings.descriptorHistorySize) {
        mDescriptorHistory.pop_front();
    }

    double bestMedian = std::numeric_limits<double>::max();
    std::size_t bestIndex = 0;
    const auto medianIndex = (mDescriptorHistory.size() - 1) / 2;
    std::vector<double> distances(mDescriptorHistory.size());
    for (std::size_t i = 0; i < mDescriptorHistory.size(); ++i) {
        for (std::size_t j = 0; j < mDescriptorHistory.size(); ++j) {
            distances[j] = mDescriptorHistory[i]->getDistance(mDescriptorHistory[j]);
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
