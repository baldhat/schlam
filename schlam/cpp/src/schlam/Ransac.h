//
// Created by baldhat on 2/8/26.
//

#ifndef SCHLAM_RANSAC_H
#define SCHLAM_RANSAC_H

#include "KeyPoint.h"

#include <eigen3/Eigen/Eigen>

#include <opencv2/core.hpp>

#include <optional>
#include <memory>
#include <vector>

std::optional<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<Eigen::Vector3d>, std::vector<bool> > >
reconstructInitial(
    const std::vector<std::shared_ptr<KeyPoint>>& aKeypoints1,
    const std::vector<std::shared_ptr<KeyPoint>>& aKeypoints2,
    const Eigen::Matrix3d &aIntrinsics);

// -----------------------------------------------------------------------
// ------------------------ Essential Matrix -----------------------------
// -----------------------------------------------------------------------
void findEssential(
    const std::array<std::vector<Eigen::Vector3d>, 2> &aAllPoint,
    const std::vector<std::array<Eigen::Matrix<double, 8, 3>, 2> > &aCandidates,
    std::vector<bool> &aInliers, double &aScore,
    Eigen::Matrix3d &aEssentialMatrix, double aSigma);

Eigen::Matrix3d
computeEssential(std::array<Eigen::Matrix<double, 8, 3>, 2> &aPoints);

double
checkEssential(const Eigen::Matrix3d &aEssentialMatrix,
               const std::array<std::vector<Eigen::Vector3d>, 2> &aAllPoints,
               std::vector<bool> &aInliers, const double aSigma);

double calculateSymmetricErrorEssential(const Eigen::Vector3d &aLine,
                                        const Eigen::Vector3d &aPoint,
                                        double aInvSigmaSquare);

std::optional<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<Eigen::Vector3d>, std::vector<bool> > >
recoverPoseFromEssential(
    const Eigen::Matrix3d aEssential,
    const std::array<std::vector<Eigen::Vector3d>, 2> &aAllPoints,
    const std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>& aKeyPoints,
    const std::vector<bool> &aInliers, const Eigen::Matrix3d& aIntrinsics);

std::array<Eigen::Vector3d, 2> triangulate(const Eigen::Vector3d &aP1, const Eigen::Vector3d &aP2,
                                           const Eigen::Matrix4d &aTransform);

// ----------------------------------------------------------------------
// --------------------------- Homography -------------------------------
// ----------------------------------------------------------------------

void findHomography(
    const std::array<std::vector<Eigen::Vector3d>, 2> &aAllPoint,
    const std::vector<std::array<Eigen::Matrix<double, 8, 3>, 2> > &aCandidates,
    std::vector<bool> &aInliers, double &aScore,
    Eigen::Matrix3d &aHomographyMatrix, double aSigma);

Eigen::Matrix3d
computeHomography(std::array<Eigen::Matrix<double, 8, 3>, 2> &aPoints);

double
checkHomography(const Eigen::Matrix3d &aHomography,
                const std::array<std::vector<Eigen::Vector3d>, 2> &aAllPoints,
                std::vector<bool> &aInliers, const double aSigma);

double calculateErrorHomography(const Eigen::Matrix3d &aHomography,
                                const Eigen::Vector3d &aP1,
                                const Eigen::Vector3d &aP2,
                                const double aInvSigmaSquare);

std::optional<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<Eigen::Vector3d>, std::vector<bool> > >
recoverPoseFromHomography(
    const Eigen::Matrix3d& aHomography,
    const std::array<std::vector<Eigen::Vector3d>, 2> &aAllPoints,
    const std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>& aKeyPoints,
    const std::vector<bool> &aInliers, const Eigen::Matrix3d& aIntrinsics);

// ----------
// -------------------------- Common -------------------------------------
// -----------------------------------------------------------------------
std::vector<std::array<Eigen::Matrix<double, 8, 3>, 2> >
selectCandidates(const std::vector<Eigen::Vector3d> &aPoints1,
                 const std::vector<Eigen::Vector3d> &aPoints2,
                 std::uint32_t aNumCandidates);

std::vector<std::uint32_t> getSparseSubset(std::uint32_t N, std::uint32_t T);

Eigen::Vector3d euclidean(const Eigen::Vector4f &aPT);

std::pair<bool, double> isValid(const Eigen::Vector3d &p1_3D, const Eigen::Vector3d &p2_3D, const Eigen::Matrix3d &R,
             const Eigen::Vector3d &t, const Eigen::Matrix3d &intrinsics,
             const std::shared_ptr<KeyPoint>& aKP1, const std::shared_ptr<KeyPoint>& aKP2);

#endif // SCHLAM_RANSAC_H
