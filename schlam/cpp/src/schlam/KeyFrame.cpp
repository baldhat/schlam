//
// Created by baldhat on 10/12/26.
//

#include "KeyFrame.h"

#include <typeinfo>

namespace {

std::vector<OrbDescriptor::Words> packOrbDescriptors(const std::vector<std::shared_ptr<KeyPoint>>& points) {
    std::vector<OrbDescriptor::Words> packed;
    packed.reserve(points.size());
    for (const auto& point : points) {
        const auto descriptor = point->getDescriptor();
        if (!descriptor || typeid(*descriptor) != typeid(OrbDescriptor)) return {};
        packed.push_back(static_cast<const OrbDescriptor&>(*descriptor).packedWords());
    }
    return packed;
}

} // namespace


KeyFrame::KeyFrame(
  const KeyFrameId id,
  const Frame& frame)
      : mId(id), mTimestamp(frame.mTimestamp), mIntrinsics(frame.mIntrinsics), mKeypointTree(frame.mKeypointTree)
      , mKeyPoints(mKeypointTree->getFeatures()), mMapToCamera(frame.mMapToCamera), mImageData(frame.mImage)
      , mOrbDescriptors(packOrbDescriptors(mKeyPoints)) {
    mObservedPointByFeature.resize(mKeyPoints.size());
}


std::vector<std::shared_ptr<KeyPoint>> KeyFrame::getUnmatchedFeatures() const {
    std::vector<std::shared_ptr<KeyPoint>> result;
    for (auto& point : mKeyPoints) {
        if (!point->getMatched()) {
            result.push_back(point);
        }
    }
    return result;
}
