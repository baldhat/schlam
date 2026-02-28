//
// Created by baldhat on 2/7/26.
//

#ifndef SCHLAM_UTILS_H
#define SCHLAM_UTILS_H

#include "KeyPoint.h"

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Eigen>

struct Point {
    std::int32_t x;
    std::int32_t y;
};

inline std::uint32_t clip(const std::uint32_t aValue, const std::uint32_t aMin, const std::uint32_t aMax) {
    return std::min(std::max(aValue, aMin), aMax);
}


std::vector<cv::Mat> buildPyramid(const cv::Mat &aImage, const std::uint8_t aNLevels, const double aScale);

std::vector<Point> getPointsInRadius(int aRadius);

std::vector<KeyPoint> removeAtImageBorder(const std::vector<KeyPoint> &aKps, const std::uint32_t aImageWidth, const std::uint32_t aImageHeight,
                         const std::uint16_t aBorderSize);

std::vector<Eigen::Vector3f> toEigen(const std::vector<KeyPoint> &aKeypoints);

std::vector<Eigen::Vector3f> toNormalized(const std::vector<Eigen::Vector3f>& aPoints , const Eigen::Matrix3f &aInvIntrinsics);

template <typename T>
std::vector<T> filterByInlierMask(const std::vector<T>& input, const std::vector<bool>& mask) {
    std::vector<T> result;
    assert(input.size() == mask.size());
    for (int i = 0; i < input.size(); ++i) {
        if (mask[i]) {
            result.push_back(input[i]);
        }
    }
    return result;
}

#endif //SCHLAM_UTILS_H
