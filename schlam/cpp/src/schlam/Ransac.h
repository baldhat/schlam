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

std::optional<std::tuple<Eigen::Matrix3f, Eigen::Vector3f, std::vector<Eigen::Vector3f>, std::vector<bool> > >
reconstructInitial(
    const std::vector<KeyPoint*> &aKeypoints1,
    const std::vector<KeyPoint*> &aKeypoints2,
    const Eigen::Matrix3f &aIntrinsics);

// -----------------------------------------------------------------------
// ------------------------ Essential Matrix -----------------------------
// -----------------------------------------------------------------------
void findEssential(
    const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoint,
    const std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2> > &aCandidates,
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

std::optional<std::tuple<Eigen::Matrix3f, Eigen::Vector3f, std::vector<Eigen::Vector3f>, std::vector<bool> > >
recoverPoseFromEssential(
    const Eigen::Matrix3f aEssential,
    const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
    const std::array<std::vector<KeyPoint*>, 2> &aKeyPoints,
    const std::vector<bool> &aInliers, const Eigen::Matrix3f& aIntrinsics);

std::array<Eigen::Vector3f, 2> triangulate(const Eigen::Vector3f &aP1, const Eigen::Vector3f &aP2,
                                           const Eigen::Matrix4f &aTransform);

// ----------------------------------------------------------------------
// --------------------------- Homography -------------------------------
// ----------------------------------------------------------------------

void findHomography(
    const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoint,
    const std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2> > &aCandidates,
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

std::optional<std::tuple<Eigen::Matrix3f, Eigen::Vector3f, std::vector<Eigen::Vector3f>, std::vector<bool> > >
recoverPoseFromHomography(
    const Eigen::Matrix3f& aHomography,
    const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
    const std::array<std::vector<KeyPoint*>, 2> &aKeyPoints,
    const std::vector<bool> &aInliers, const Eigen::Matrix3f& aIntrinsics);

// ----------
// -------------------------- Common -------------------------------------
// -----------------------------------------------------------------------
std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2> >
selectCandidates(const std::vector<Eigen::Vector3f> &aPoints1,
                 const std::vector<Eigen::Vector3f> &aPoints2,
                 std::uint32_t aNumCandidates);

std::vector<std::uint32_t> getSparseSubset(std::uint32_t N, std::uint32_t T);

Eigen::Vector3f euclidean(const Eigen::Vector4f &aPT);

std::pair<bool, float> isValid(const Eigen::Vector3f &p1_3D, const Eigen::Vector3f &p2_3D, const Eigen::Matrix3f &R,
             const Eigen::Vector3f &t, const Eigen::Matrix3f &intrinsics, const KeyPoint* aKP1, const KeyPoint* aKP2);

#endif // SCHLAM_RANSAC_H
