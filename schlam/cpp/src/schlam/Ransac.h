//
// Created by baldhat on 2/8/26.
//

#ifndef SCHLAM_RANSAC_H
#define SCHLAM_RANSAC_H

std::tuple<Eigen::Matrix3d, double> calculateEssentialMatrix(const std::vector<KeyPoint> aKeypoints1,
                                         const std::vector<KeyPoint> aKeypoints2, const Eigen::Matrix3d aIntrinsics);
std::tuple<Eigen::Matrix3d, double> calculateHomographyMatrix(const std::vector<KeyPoint> aKeypoints1,
                                         const std::vector<KeyPoint> aKeypoints2, const Eigen::Matrix3d aIntrinsics);

#endif //SCHLAM_RANSAC_H

