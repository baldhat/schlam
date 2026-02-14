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

  auto points1 = toNormalized(toEigen(aKeypoints1), invIntrinsics);
  auto points2 = toNormalized(toEigen(aKeypoints2), invIntrinsics);

  auto candidates = selectCandidates(points1, points2, 200);

  std::vector<bool> inliersHomography, inliersEssential;
  double scoreHomography{0}, scoreEssential{0};

  Eigen::Matrix3f essential;
  double sigma = 1.0 / aIntrinsics(0, 0); 

  // TODO Move to thread once it works
  findEssential({points1, points2}, candidates, inliersEssential,
                scoreEssential, essential, sigma);
}

void findEssential(
    const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
    const std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2>> &aCandidates,
    std::vector<bool> &aInliers, double &aScore,
    Eigen::Matrix3f &aEssentialMatrix, double aSigma) {
  std::size_t numCandidates{aAllPoints[0].size()};
  std::vector<bool> inliers(numCandidates, false);
  aScore = 0;

  for (std::uint32_t iter = 0; iter < 200; iter++) {
    auto candidates = aCandidates[iter];
    Eigen::Matrix3f essentialMatrix = computeEssential(candidates);
    double score = checkEssential(essentialMatrix, aAllPoints, inliers, aSigma);
    if (score > aScore) {
      aEssentialMatrix = essentialMatrix;
      aInliers = inliers;
      aScore = score;
    }
  }
  std::cout << "Best essential score: " << aScore << std::endl;
  int count = 0;
  for (const auto &val : aInliers) {
    if (val)
      count++;
  }
  std::cout << "Num inliers: " << count << std::endl;
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
  candidates.reserve(aNumCandidates);

  for (std::uint32_t i = 0; i < aNumCandidates; ++i) {
    std::array<Eigen::Matrix<float, 8, 3>, 2> pair;

    auto indices =
        getSparseSubset(8, static_cast<std::uint32_t>(aPoints1.size()));

    for (std::uint8_t j = 0; j < 8; ++j) {
      pair[0].row(j) = aPoints1[indices[j]];
      pair[1].row(j) = aPoints2[indices[j]];
    }
    candidates.push_back(pair);
  }
  return candidates;
}

std::vector<std::uint32_t> getSparseSubset(std::uint32_t aCount,
                                           std::uint32_t aMaxIndex) {
  if (aCount > aMaxIndex)
    return {}; // Fehlerbehandlung

  std::set<std::uint32_t> unique_indices;
  static std::random_device rd;
  static std::mt19937 g(rd());
  std::uniform_int_distribution<std::uint32_t> dist(0, aMaxIndex - 1);

  while (unique_indices.size() < aCount) {
    unique_indices.insert(dist(g));
  }

  return {unique_indices.begin(), unique_indices.end()};
}

double
checkEssential(const Eigen::Matrix3f &aEssentialMatrix,
               const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
               std::vector<bool> &aInliers, const double aSigma) {
    double score{0};
    const size_t nPoints = aAllPoints[0].size();
    
    aInliers.assign(nPoints, false);

    constexpr double threshold = 3.841;
    constexpr double essentialScore = 5.991;
    const double invSigmaSquare = 1.0 / (aSigma * aSigma);

    const Eigen::Matrix3f essentialMatrixT = aEssentialMatrix.transpose();

    for (size_t i = 0; i < nPoints; i++) {
        const auto& p1 = aAllPoints[0][i]; // Punkt in Bild 1
        const auto& p2 = aAllPoints[1][i]; // Punkt in Bild 2

        Eigen::Vector3f line2 = aEssentialMatrix * p1;
        double error2 = calculateSymmetricError(line2, p2, invSigmaSquare);

        Eigen::Vector3f line1 = essentialMatrixT * p2;
        double error1 = calculateSymmetricError(line1, p1, invSigmaSquare);

        if (error1 <= threshold && error2 <= threshold) {
            aInliers[i] = true;
            score += (essentialScore - error1) + (essentialScore - error2);
        } else {
            aInliers[i] = false;
        }
    }

    return score;
}


double calculateSymmetricError(const Eigen::Vector3f& aLine, 
                               const Eigen::Vector3f& aPoint, 
                               double aInvSigmaSquare) {
    // Punkt-Linie-Abstand: (x' * L)^2 / (a^2 + b^2)
    // aLine = (a, b, c), wobei ax + by + c = 0
    double residual = aLine.dot(aPoint);
    double denominator = aLine.head<2>().squaredNorm();
    
    // Division durch Null verhindern, falls die Linie ungültig ist
    if (denominator < 1e-9) return 0.0;

    return (residual * residual / denominator) * aInvSigmaSquare;
}
