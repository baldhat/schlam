
#include "image_data.hpp"

ImageData::ImageData(const utils::TTimestamp &aTimestamp,
                    cv::Mat &aImage, const Eigen::Matrix3f &aIntrinsics,
                    const std::string &aCF)
    : mTimestamp(aTimestamp)
    , mImage(aImage)
    , mIntrinsics(aIntrinsics)
    , mCoordinateFrame(aCF) {
        
    
}