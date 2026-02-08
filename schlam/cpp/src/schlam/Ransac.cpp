//
// Created by baldhat on 2/8/26.
//

#include "Ransac.h"
#include "KeyPoint.h"
#include "utils.h"

#include <opencv2/calib3d/calib3d.hpp>

#include <eigen3/Eigen/src/Core/Matrix.h>

#include <vector>


std::tuple<Eigen::Matrix3d, double> calculateEssentialMatrix(const std::vector<KeyPoint> aKeypoints1,
                                         const std::vector<KeyPoint> aKeypoints2, const Eigen::Matrix3d aIntrinsics) {
    auto points1 = toOpenCV(aKeypoints1);
    auto points2 = toOpenCV(aKeypoints2);
    cv::Mat mask;
    auto essentialMat = cv::findEssentialMat(points1, points2, aIntrinsics, cv::RANSAC, 0.999, 1.0, mask);
    //score =
}

Eigen::Matrix3d calculateHomographyMatrix(const std::vector<KeyPoint> aKeypoints1,
                                         const std::vector<KeyPoint> aKeypoints2, const Eigen::Matrix3d aIntrinsics) {

}


