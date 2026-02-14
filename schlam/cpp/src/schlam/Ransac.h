//
// Created by baldhat on 2/8/26.
//

#ifndef SCHLAM_RANSAC_H
#define SCHLAM_RANSAC_H

#include "KeyPoint.h"

#include <eigen3/Eigen/Eigen>

#include <opencv2/core.hpp>

#include <vector>

void findEssential(
    const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoint,
    const std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2>> &aCandidates,
    std::vector<bool> &aInliers, double &aScore);

Eigen::Matrix3f
computeEssential(std::array<Eigen::Matrix<float, 8, 3>, 2> &aPoints);

std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2>>
selectCandidates(const std::vector<Eigen::Vector3f> &aPoints1,
                 const std::vector<Eigen::Vector3f> &aPoints2,
                 std::uint32_t aNumCandidates);

std::vector<std::uint32_t> getSparseSubset(std::uint32_t N, std::uint32_t T);

double
checkEssential(const Eigen::Matrix3f &aEssentialMatrix,
               const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
               std::vector<bool> &aInliers, const double aSigma);

double reprojectionScore(const Eigen::Matrix3f& aEssentialMatrix, const Eigen::Vector3f& aP1, const Eigen::Vector3f& aP2, double aInvSigmaSquare);

#endif // SCHLAM_RANSAC_H
