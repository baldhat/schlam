
#pragma once

#include "../utils.hpp"

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Core>

class ImageData {
public:
  ImageData(const utils::TTimestamp &aTimestamp, cv::Mat &aImage,
            const Eigen::Matrix3f &aIntrinsics, const std::string &aCF);

  const utils::TTimestamp mTimestamp;
  cv::Mat mImage;
  const Eigen::Matrix3f mIntrinsics;
  const std::string mCoordinateFrame;
};