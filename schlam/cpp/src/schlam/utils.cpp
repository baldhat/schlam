//
// Created by baldhat on 2/7/26.
//

#include "utils.h"

#include "KeyPoint.h"

std::vector<cv::Mat> buildPyramid(const cv::Mat &aImage, const std::uint8_t aNLevels, const double aScale) {
    std::vector<cv::Mat> pyramid{aImage};
    for (int i = 0; i < aNLevels; i++) {
        auto lastImage = pyramid.back();
        int newWidth = std::round(lastImage.cols * aScale);
        int newHeight = std::round(lastImage.rows * aScale);
        cv::Mat resizedImage;
        cv::resize(lastImage, resizedImage, cv::Size(newWidth, newHeight));
        pyramid.push_back(resizedImage);
    }
    return pyramid;
}

std::vector<Point> getPointsInRadius(int aRadius) {
    std::vector<Point> points;
    // Calculate radius squared once to avoid repeated sqrt() or multiplication
    int radiusSquared = aRadius * aRadius;

    // Iterate through the bounding box of the circle
    for (int x = -aRadius; x <= aRadius; ++x) {
        for (int y = -aRadius; y <= aRadius; ++y) {
            // Check if the point (x, y) is within the circle
            if (x * x + y * y <= radiusSquared) {
                points.push_back({x, y});
            }
        }
    }

    return points;
}


void removeAtImageBorder(std::vector<KeyPoint> &aKps, const std::uint32_t aImageWidth, const std::uint32_t aImageHeight,
                         const std::uint16_t aBorderSize) {
    for (std::uint32_t i = 0; i < aKps.size(); i++) {
        std::uint32_t x{aKps[i].getImgX()}, y{aKps[i].getImgY()};
        if (x < aBorderSize || y < aBorderSize || x >= (aImageWidth - aBorderSize) || y >= (aImageHeight - aBorderSize)) {
            aKps.erase(aKps.begin() + i);
        }
    }
}
