//
// Created by baldhat on 2/26/26.
//

#include "Frame.h"

#include <stdexcept>


Frame::Frame(const std::shared_ptr<ImageData> aImageData)
    : mTimestamp(aImageData->mTimestamp)
    , mImage(aImageData->mImage)
    , mIntrinsics(aImageData->mIntrinsics) {
}

const tft::RigidTransform3D& Frame::getMapToCamera() const {
    if (!mMapToCamera) {
        throw std::logic_error("Frame has no accepted map-to-camera pose");
    }
    return *mMapToCamera;
}
