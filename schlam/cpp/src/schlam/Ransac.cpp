//
// Created by baldhat on 2/8/26.
//

#include "Ransac.h"
#include "utils.h"

#include <opencv2/calib3d/calib3d.hpp>

#include <random>
#include <set>

void reconstruct(const std::vector<KeyPoint> aKeypoints1,
                 const std::vector<KeyPoint> aKeypoints2,
                 const Eigen::Matrix3d aIntrinsics) {
  assert(aKeypoints1.size() == aKeypoints2.size());

  Eigen::Matrix3f invIntrinsics = aIntrinsics.inverse().cast<float>();

  auto points1 = toNormalized(aKeypoints1, invIntrinsics);
  auto points2 = toNormalized(aKeypoints2, invIntrinsics);

  auto candidates = selectCandidates(points1, points2, 500);

  std::vector<bool> inliersHomography, inliersEssential;
  double scoreHomography{0}, scoreEssential{0};

  // TODO Move to thread once it works
  findEssential({points1, points2}, candidates, inliersEssential,
                scoreEssential);
}

void findEssential(
    const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
    const std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2>> &aCandidates,
    std::vector<bool> &aInliers, double &aScore, Eigen::Matrix3f& aEssentialMatrix) {
  std::size_t numCandidates{aAllPoints[0].size()};
  std::vector<bool> inliers(numCandidates, false);
  aScore = 0;

  for (std::uint32_t iter = 0; iter < 500; iter++) {
    auto candidates = aCandidates[iter];
    Eigen::Matrix3f essentialMatrix = computeEssential(candidates);
    double score = checkEssential(essentialMatrix, aAllPoints, inliers, 0.1); // TODO: What is the sigma value supposed to be?
    if (score > aScore) {
      aEssentialMatrix = essentialMatrix;
      aInliers = inliers;
      aScore = score;
    }
  }
}

Eigen::Matrix3f
computeEssential(std::array<Eigen::Matrix<float, 8, 3>, 2> &aPoints) {
  Eigen::Matrix<float, 8, 9> res;
  for (int j = 0; j < 3; ++j) {
    res.block<8, 3>(0, j * 3) =
        aPoints[0].array().colwise() * aPoints[1].col(j).array();
  }
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(res, Eigen::ComputeFullV);
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> essential(
      svd.matrixV().col(8).data());
  return essential;
}

std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2>>
selectCandidates(const std::vector<Eigen::Vector3f> &aPoints1,
                 const std::vector<Eigen::Vector3f> &aPoints2,
                 std::uint32_t aNumCandidates) {

  std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2>> candidates;
  for (std::uint32_t i = 0; i < aNumCandidates; i++) {
    std::array<Eigen::Matrix<float, 8, 3>, 2> pair;

    auto indices = getSparseSubset(8, aNumCandidates);
    pair[0] = Eigen::Matrix<float, 8, 3>();
    pair[1] = Eigen::Matrix<float, 8, 3>();
    for (std::uint8_t j = 0; j < 8; j++) {
      pair[0] << aPoints1[indices[j]];
      pair[1] << aPoints2[indices[j]];
    }
    candidates.push_back(pair);
  }
  return candidates;
}

std::vector<std::uint32_t> getSparseSubset(std::uint32_t N, std::uint32_t T) {
  std::set<int> unique_indices;
  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_int_distribution<int> dist(0, N - 1);

  while (unique_indices.size() < T) {
    unique_indices.insert(dist(g));
  }

  return {unique_indices.begin(), unique_indices.end()};
}

double
checkEssential(const Eigen::Matrix3f &aEssentialMatrix,
               const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
               std::vector<bool> &aInliers, const double aSigma) {
  double score{0};

  // orbslam constants
  constexpr double th = 3.841;
  constexpr double thScore = 5.991;
  const double invSigmaSquare = 1.0 / (aSigma*aSigma);

  for (std::uint32_t i = 0; i < aAllPoints[0].size(); i++) {
    bool inlier{true};

    auto rightReprojectScore = reprojectionScore(aEssentialMatrix, aAllPoints[0][i], aAllPoints[1][i], invSigmaSquare);
    if (rightReprojectScore > th) {
      inlier = false;
    } else {
      score = thScore - rightReprojectScore;
    }

    auto leftReprojectScore = reprojectionScore(aEssentialMatrix, aAllPoints[1][i], aAllPoints[0][i], invSigmaSquare);
    if (leftReprojectScore > th) {
      inlier = false;
    } else {
      score = thScore - leftReprojectScore;
    }

    aInliers[i] = inlier;
  }
  return score;
}

double reprojectionScore(const Eigen::Matrix3f& aEssentialMatrix, const Eigen::Vector3f& aP1, const Eigen::Vector3f& aP2, double aInvSigmaSquare) {
  Eigen::Vector3f vec = aP1.transpose() * aEssentialMatrix;
  double residual = vec.transpose() * aP2;
  double chiSquared2 = (residual*residual) / (vec(0)*vec(0)+vec(1)*vec(1)) * aInvSigmaSquare;
  return chiSquared2;
}
