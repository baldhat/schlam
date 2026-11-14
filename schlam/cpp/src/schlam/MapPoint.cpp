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
#include <stdexcept>


MapPoint::MapPoint(const Eigen::Vector3d& position, const DescriptorPtr& descriptor, const MapPointId id, const KeyFrameId creatorId,
                   const utils::TTimestamp& creationTimestamp,
                   schlam::config::MapPoints settings)
    : mPosition(position), mDescriptor(descriptor), mLastInlierDescriptor(descriptor),
      mId(id), mCreatorId(creatorId), mCreationTimestamp(creationTimestamp),
      mSettings(settings), mDescriptorHistory{descriptor} {
    if (!std::isfinite(settings.observationDistanceFactor) || settings.observationDistanceFactor < 1.0) {
        throw std::invalid_argument("mapPoints.observationDistanceFactor must be finite and at least one");
    }
}

void MapPoint::addInlierObservation(const DescriptorPtr& descriptor, const Eigen::Vector3d& cameraCenter) {
    mNumInlierObservations++;
    addViewingObservation(cameraCenter);
    const double distance = (cameraCenter - mPosition).stableNorm();
    if (std::isfinite(distance) && distance > 1e-9) {
        mObservationDistances.insert(std::lower_bound(mObservationDistances.begin(), mObservationDistances.end(), distance), distance);
    }
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

bool MapPoint::isViewCompatible(const Eigen::Vector3d& cameraCenter) const {
    const Eigen::Vector3d direction = cameraCenter - mPosition;
    const double length = direction.stableNorm();
    if (!std::isfinite(length) || length <= 1e-9) return false;
    if (!mObservationDistances.empty()) {
        const auto middle = mObservationDistances.size() / 2;
        const double median = mObservationDistances.size() % 2 ? mObservationDistances[middle] :
            mObservationDistances[middle - 1] / 2.0 + mObservationDistances[middle] / 2.0;
        const double factor = mSettings.observationDistanceFactor;
        if (length < median / factor || length / factor > median) return false;
    }
    const double meanLength = mViewingDirectionSum.norm();
    if (!std::isfinite(meanLength) || meanLength <= 1e-9) return true;
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
