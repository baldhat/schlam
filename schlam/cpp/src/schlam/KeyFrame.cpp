//
// Created by baldhat on 10/12/26.
//

#include "KeyFrame.h"


KeyFrame::KeyFrame(
  const KeyFrameId id,
  const Frame& frame)
      : mId(id), mIntrinsics(frame.mIntrinsics), mKeyPoints(frame.mKeypointTree->getFeatures())
      , mMapToCamera(frame.mMapToCamera), mTimestamp(frame.mTimestamp), mKeypointTree(frame.mKeypointTree)
      , mImageData(frame.mImage){

}


std::vector<std::shared_ptr<KeyPoint>> KeyFrame::getUnmatchedFeatures() {
    std::vector<std::shared_ptr<KeyPoint>> result;
    for (auto& point : mKeyPoints) {
        if (!point->getMatched()) {
            result.push_back(point);
        }
    }
    return result;
}
