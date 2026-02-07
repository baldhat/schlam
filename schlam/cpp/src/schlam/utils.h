//
// Created by baldhat on 2/7/26.
//

#ifndef SCHLAM_UTILS_H
#define SCHLAM_UTILS_H

#include <opencv2/opencv.hpp>

#include "KeyPoint.h"

struct Point {
    std::int32_t x;
    std::int32_t y;
};

inline std::uint32_t clip(const std::uint32_t aValue, const std::uint32_t aMin, const std::uint32_t aMax) {
    return std::min(std::max(aValue, aMin), aMax);
}


std::vector<cv::Mat> buildPyramid(const cv::Mat &aImage, const std::uint8_t aNLevels, const double aScale);

std::vector<Point> getPointsInRadius(int aRadius);

void removeAtImageBorder(std::vector<KeyPoint> &aKps, const std::uint32_t aImageWidth, const std::uint32_t aImageHeight,
                         const std::uint16_t aBorderSize);

#endif //SCHLAM_UTILS_H