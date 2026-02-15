//
// Created by baldhat on 2/8/26.
//

#include "Ransac.h"
#include "utils.h"

#include <opencv2/calib3d/calib3d.hpp>

#include <random>
#include <set>

void reconstructInitial(const std::vector<KeyPoint> aKeypoints1,
                        const std::vector<KeyPoint> aKeypoints2,
                        const Eigen::Matrix3d aIntrinsics) {
  assert(aKeypoints1.size() == aKeypoints2.size());

  Eigen::Matrix3f invIntrinsics = aIntrinsics.inverse().cast<float>();

  // TODO: make this an actual normalization, not just inverseK
  auto points1 = toNormalized(toEigen(aKeypoints1), invIntrinsics);
  auto points2 = toNormalized(toEigen(aKeypoints2), invIntrinsics);

  auto candidates = selectCandidates(points1, points2, 200);

  std::vector<bool> inliersHomography, inliersEssential;
  double scoreHomography{0}, scoreEssential{0};

  Eigen::Matrix3f essential;
  Eigen::Matrix3f homography;
  double sigma = 1.0 / aIntrinsics(0, 0);

  // TODO Move to thread once it works
  findEssential({points1, points2}, candidates, inliersEssential,
                scoreEssential, essential, sigma);

  findHomography({points1, points2}, candidates, inliersHomography,
                 scoreHomography, homography, sigma);

  auto rh = scoreHomography / (scoreHomography + scoreEssential);
  std::cout << (rh > 0.45 ? "Choosing Homography" : "Choosing Essential")
            << std::endl;
}

// ------------------------------------------------------------
// ---------------------- Homography --------------------------
// ------------------------------------------------------------

void findHomography(
    const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
    const std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2>> &aCandidates,
    std::vector<bool> &aInliers, double &aScore,
    Eigen::Matrix3f &aHomographyMatrix, double aSigma) {
  std::size_t numCandidates{aAllPoints[0].size()};
  std::vector<bool> inliers(numCandidates, false);
  aScore = 0;

  for (std::uint32_t iter = 0; iter < 200; iter++) {
    auto candidates = aCandidates[iter];
    Eigen::Matrix3f homography = computeHomography(candidates);
    double score = checkHomography(homography, aAllPoints, inliers, aSigma);
    if (score > aScore) {
      aHomographyMatrix = homography;
      aInliers = inliers;
      aScore = score;
    }
  }
}

Eigen::Matrix3f
computeHomography(std::array<Eigen::Matrix<float, 8, 3>, 2> &aPoints) {
  Eigen::Matrix<float, 8, 9> res;
  for (std::uint32_t i = 0; i < 4; i++) {
    Eigen::Vector<float, 9> row1;
    Eigen::Vector<float, 9> row2;
    Eigen::Vector3f wx = aPoints[0](i, 2) * aPoints[1].row(i);
    Eigen::Vector3f yx = aPoints[0](i, 1) * aPoints[1].row(i);
    Eigen::Vector3f xx = aPoints[0](i, 0) * aPoints[1].row(i);

    row1 << 0, 0, 0, -wx, yx;
    row2 << wx, 0, 0, 0, -xx;
    res.row(i * 2) = row1;
    res.row(i * 2 + 1) = row2;
  }
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(res, Eigen::ComputeFullV);
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> homography(
      svd.matrixV().col(8).data());
  return homography;
}

double
checkHomography(const Eigen::Matrix3f &aHomography,
                const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
                std::vector<bool> &aInliers, const double aSigma) {
  double score{0};
  const size_t nPoints = aAllPoints[0].size();

  aInliers.assign(nPoints, false);
  auto homographyInv = aHomography.inverse();

  constexpr double threshold = 5.991;
  constexpr double homographyScore = 5.991;
  const double invSigmaSquare = 1.0 / (aSigma * aSigma);

  for (size_t i = 0; i < nPoints; i++) {
    const auto &p1 = aAllPoints[0][i];
    const auto &p2 = aAllPoints[1][i];

    auto error1 =
        calculateErrorHomography(homographyInv, p1, p2, invSigmaSquare);
    auto error2 = calculateErrorHomography(aHomography, p2, p1, invSigmaSquare);

    if (error1 <= threshold && error2 <= threshold) {
      aInliers[i] = true;
      score += (homographyScore - error1) + (homographyScore - error2);
    } else {
      aInliers[i] = false;
    }
  }

  return score;
}

double calculateErrorHomography(const Eigen::Matrix3f &aHomography,
                                const Eigen::Vector3f &aP1,
                                const Eigen::Vector3f &aP2,
                                const double aInvSigmaSquare) {
  const float w2in1inv = 1.0 / aP2.dot(aHomography.row(2));
  const float u2in1 = aP2.dot(aHomography.row(0)) * w2in1inv;
  const float v2in1 = aP2.dot(aHomography.row(1)) * w2in1inv;
  const float squareDist1 = (aP1.x() - u2in1) * (aP1.x() - u2in1) +
                            (aP1.y() - v2in1) * (aP1.y() - v2in1);
  const float chiSquare1 = squareDist1 * aInvSigmaSquare;
  return chiSquare1;
}

// -----------------------------------------------------------------------
// ------------------------ Essential Matrix -----------------------------
// -----------------------------------------------------------------------

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
    const auto &p1 = aAllPoints[0][i];
    const auto &p2 = aAllPoints[1][i];

    Eigen::Vector3f line2 = aEssentialMatrix * p1;
    double error2 = calculateSymmetricErrorEssential(line2, p2, invSigmaSquare);

    Eigen::Vector3f line1 = essentialMatrixT * p2;
    double error1 = calculateSymmetricErrorEssential(line1, p1, invSigmaSquare);

    if (error1 <= threshold && error2 <= threshold) {
      aInliers[i] = true;
      score += (essentialScore - error1) + (essentialScore - error2);
    } else {
      aInliers[i] = false;
    }
  }

  return score;
}

double calculateSymmetricErrorEssential(const Eigen::Vector3f &aLine,
                                        const Eigen::Vector3f &aPoint,
                                        double aInvSigmaSquare) {
  // Punkt-Linie-Abstand: (x' * L)^2 / (a^2 + b^2)
  // aLine = (a, b, c), wobei ax + by + c = 0
  double residual = aLine.dot(aPoint);
  double denominator = aLine.head<2>().squaredNorm();

  // Division durch Null verhindern, falls die Linie ungültig ist
  if (denominator < 1e-9)
    return 0.0;

  return (residual * residual / denominator) * aInvSigmaSquare;
}

// ---------------------------------------------------------------------
// --------------------------- Common ----------------------------------
// ---------------------------------------------------------------------
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
