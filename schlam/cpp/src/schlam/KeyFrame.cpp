//
// Created by baldhat on 10/12/26.
//

#include "KeyFrame.h"


KeyFrame::KeyFrame(
  const KeyFrameId id,
  const Frame& frame)
      : mId(id), mIntrinsics(frame.mIntrinsics), mKeyPoints(frame.mKeypointTree->getFeatures())
      , mMapToCamera(*frame.mMapToCamera), mTimestamp(frame.mTimestamp), mKeypointTree(frame.mKeypointTree)
      , mImageData(frame.mImage){

}