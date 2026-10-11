//
// Created by baldhat on 2/26/26.
//

#include "Frame.h"


Frame::Frame(const std::shared_ptr<ImageData> aImageData)
    : mImage(aImageData->mImage)
    , mIntrinsics(aImageData->mIntrinsics)
    , mTimestamp(aImageData->mTimestamp)    {

}
