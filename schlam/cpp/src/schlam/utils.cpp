//
// Created by baldhat on 2/7/26.
//

#include "utils.h"

#include "KeyPoint.h"

#include <eigen3/Eigen/Eigen>

std::vector<cv::Mat> buildPyramid(const cv::Mat &aImage,
                                  const std::uint8_t aNLevels,
                                  const double aScale) {
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
std::vector<KeyPoint> removeAtImageBorder(const std::vector<KeyPoint>& aKps,
                         const std::uint32_t aImageWidth,
                         const std::uint32_t aImageHeight,
                         const std::uint16_t aBorderSize) {
  std::vector<KeyPoint> filtered;
  for (std::uint32_t i = 0; i < aKps.size(); i++) {
    std::uint32_t x{aKps[i].getImgX()}, y{aKps[i].getImgY()};
    if (x >= aBorderSize && y >= aBorderSize &&
        x < (aImageWidth - aBorderSize) && y < (aImageHeight - aBorderSize)) {
      filtered.push_back(aKps[i]);
    }
  }
  return filtered;
}

std::vector<Eigen::Vector3f> toEigen(const std::vector<KeyPoint> &aKeypoints) {
  std::vector<Eigen::Vector3f> points;
  for (const auto &keyPoint : aKeypoints) {
    points.emplace_back(static_cast<float>(keyPoint.getImgX()),
                  static_cast<float>(keyPoint.getImgY()), 1);
  }
  return points;

}

std::vector<Eigen::Vector3f> toNormalized(const std::vector<Eigen::Vector3f>& aPoints , const Eigen::Matrix3f &aInvIntrinsics) {
  std::vector<Eigen::Vector3f> points;
  for (const auto &point: aPoints) {
    points.emplace_back(aInvIntrinsics * point);
  }
  return points;
}

Eigen::Matrix3f computeNormalizationMatrix(const std::vector<Eigen::Vector3f>& aPoints) {
  Eigen::Vector2f centroid(0, 0);
  for (const auto& p : aPoints) {
    centroid += p.head<2>();
  }
  centroid /= static_cast<float>(aPoints.size());

  float meanDist = 0;
  for (const auto& p : aPoints) {
    meanDist += (p.head<2>() - centroid).norm();
  }
  meanDist /= static_cast<float>(aPoints.size());

  float scale = std::sqrt(2.0f) / meanDist;

  Eigen::Matrix3f T = Eigen::Matrix3f::Identity();
  T(0, 0) = scale;
  T(1, 1) = scale;
  T(0, 2) = -scale * centroid.x();
  T(1, 2) = -scale * centroid.y();

  return T;
}
