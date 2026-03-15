//
// Created by baldhat on 2/26/26.
//

#include "Frame.h"

#include "data/image_data.hpp"

Frame::Frame(const std::shared_ptr<ImageData> aImageData)
    : mImage(aImageData->mImage)
    , mIntrinsics(aImageData->mIntrinsics){

}
