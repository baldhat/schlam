//
// Created by baldhat on 2/8/26.
//

#ifndef SCHLAM_RANSAC_H
#define SCHLAM_RANSAC_H

#include "KeyPoint.h"

#include <eigen3/Eigen/Eigen>

#include <opencv2/core.hpp>

#include <optional>
#include <vector>

std::optional<std::tuple<Eigen::Matrix3f, Eigen::Vector3f, std::vector<Eigen::Vector3f>>> reconstructInitial(const std::vector<KeyPoint> aKeypoints1,
                 const std::vector<KeyPoint> aKeypoints2,
                 const Eigen::Matrix3f aIntrinsics);

// -----------------------------------------------------------------------
// ------------------------ Essential Matrix -----------------------------
// -----------------------------------------------------------------------
void findEssential(
    const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoint,
    const std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2>> &aCandidates,
    std::vector<bool> &aInliers, double &aScore,
    Eigen::Matrix3f &aEssentialMatrix, double aSigma);

Eigen::Matrix3f
computeEssential(std::array<Eigen::Matrix<float, 8, 3>, 2> &aPoints);

double
checkEssential(const Eigen::Matrix3f &aEssentialMatrix,
               const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
               std::vector<bool> &aInliers, const double aSigma);

double calculateSymmetricErrorEssential(const Eigen::Vector3f &aLine,
                                        const Eigen::Vector3f &aPoint,
                                        double aInvSigmaSquare);

std::optional<std::tuple<Eigen::Matrix3f, Eigen::Vector3f, std::vector<Eigen::Vector3f>>> recoverPoseFromEssential(
  const Eigen::Matrix3f aEssential,
  const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
  const std::vector<bool> &aInliers);

std::array<Eigen::Vector3f, 2> triangulate(const Eigen::Vector3f& aP1, const Eigen::Vector3f& aP2, const Eigen::Matrix4f& aTransform);
// ----------------------------------------------------------------------
// --------------------------- Homography -------------------------------
// ----------------------------------------------------------------------

void findHomography(
    const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoint,
    const std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2>> &aCandidates,
    std::vector<bool> &aInliers, double &aScore,
    Eigen::Matrix3f &aHomographyMatrix, double aSigma);

Eigen::Matrix3f
computeHomography(std::array<Eigen::Matrix<float, 8, 3>, 2> &aPoints);

double
checkHomography(const Eigen::Matrix3f &aHomography,
                const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
                std::vector<bool> &aInliers, const double aSigma);

double calculateErrorHomography(const Eigen::Matrix3f &aHomography,
                                         const Eigen::Vector3f &aP1,
                                         const Eigen::Vector3f &aP2,
                                         const double aInvSigmaSquare);

std::optional<std::tuple<Eigen::Matrix3f, Eigen::Vector3f, std::vector<Eigen::Vector3f>>> recoverPoseFromHomography(
  const Eigen::Matrix3f aHomography,
  const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
  const std::vector<bool> &aInliers);

// ----------
// -------------------------- Common -------------------------------------
// -----------------------------------------------------------------------
std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2>>
selectCandidates(const std::vector<Eigen::Vector3f> &aPoints1,
                 const std::vector<Eigen::Vector3f> &aPoints2,
                 std::uint32_t aNumCandidates);

std::vector<std::uint32_t> getSparseSubset(std::uint32_t N, std::uint32_t T);

#endif // SCHLAM_RANSAC_H
