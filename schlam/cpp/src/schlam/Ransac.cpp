//
// Created by baldhat on 2/8/26.
//

#include "Ransac.h"
#include "random.hpp"
#include "utils.h"

#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <optional>
#include <random>
#include <set>
#include <opencv2/core/eigen.hpp>

std::optional<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<Eigen::Vector3d>, std::vector<bool> > >
reconstructInitial(
    const std::vector<std::shared_ptr<KeyPoint>>& aKeypoints1,
    const std::vector<std::shared_ptr<KeyPoint>>& aKeypoints2,
    const Eigen::Matrix3d &aIntrinsics) {
    assert(aKeypoints1.size() == aKeypoints2.size());
    const int numCandidates = 500;

    if (aKeypoints1.size() < 8 || aKeypoints2.size() < 8) {
        std::cout << "WARN: Not enough matches found!" << std::endl;
        return std::nullopt;
    }

    Eigen::Matrix3d invIntrinsics = aIntrinsics.inverse().cast<double>();

    // TODO: Check using hartley normalization, not just inverseK
    auto points1 = toNormalized(toEigen(aKeypoints1), invIntrinsics);
    auto points2 = toNormalized(toEigen(aKeypoints2), invIntrinsics);

    auto candidates = selectCandidates(points1, points2, numCandidates);

    std::vector<bool> inliersHomography, inliersEssential;
    double scoreHomography{0}, scoreEssential{0};

    Eigen::Matrix3d essential;
    Eigen::Matrix3d homography;
    double sigma = 1.0 / aIntrinsics(0, 0);

    findEssential({points1, points2}, candidates, inliersEssential,
                  scoreEssential, essential, sigma);

    findHomography({points1, points2}, candidates, inliersHomography,
                   scoreHomography, homography, sigma);

    const double totalScore = scoreHomography + scoreEssential;
    if (totalScore <= 0.0 || !std::isfinite(totalScore))
        return std::nullopt;

    auto rh = scoreHomography / totalScore;
    std::cout << (rh > 0.45 ? "Choosing Homography" : "Choosing Essential")
            << std::endl;
    if (rh > 0.45) {
        return recoverPoseFromHomography(homography, {points1, points2}, {aKeypoints1, aKeypoints2}, inliersHomography,
                                         aIntrinsics);
    } else {
        auto output = recoverPoseFromEssential(essential, {points1, points2}, {aKeypoints1, aKeypoints2},
                                               inliersEssential, aIntrinsics);
        if (output.has_value()) {
            // Returns the transformation that expresses C1 in coordinate frame C2
            auto [R, t, pts, inliers] = output.value();
        }
        return output;
    }
}

// ------------------------------------------------------------
// ---------------------- Homography --------------------------
// ------------------------------------------------------------
std::optional<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<Eigen::Vector3d>, std::vector<bool> > >
recoverPoseFromHomography(
    const Eigen::Matrix3d &aHomography,
    const std::array<std::vector<Eigen::Vector3d>, 2> &aAllPoints,
    const std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>& aKeyPoints,
    const std::vector<bool> &aInliers, const Eigen::Matrix3d &aIntrinsics) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(aHomography, Eigen::ComputeFullV);
    auto normalizedHomography = (1.0 / svd.singularValues()[1]) * aHomography;
    auto normalizedSVD = normalizedHomography.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d U = normalizedSVD.matrixU();
    const Eigen::Matrix3d V = normalizedSVD.matrixV();
    const double rotationSign = U.determinant() * V.determinant() < 0 ? -1.0f : 1.0f;
    auto lambdas = normalizedSVD.singularValues();
    auto stretch = sqrt(
        (lambdas[0] * lambdas[0] - lambdas[1] * lambdas[1]) / (lambdas[0] * lambdas[0] - lambdas[2] * lambdas[2]));
    auto compression = sqrt(
        (lambdas[1] * lambdas[1] - lambdas[2] * lambdas[2]) / (lambdas[0] * lambdas[0] - lambdas[2] * lambdas[2]));
    //auto V = normalizedSVD.matrixV();
    //auto norm = sqrt(stretch * stretch + compression * compression);
    //stretch /= norm;
    //compression /= norm;
    std::vector<std::tuple<Eigen::Matrix3d, Eigen::Vector3d> > solutions;
    auto combinations = std::vector<std::array<int, 2> >{
        {1, 1},
        {1, -1},
        {-1, 1},
        {-1, -1}
    };
    // if (std::abs(lambdas[0] - lambdas[1]) > 0.000001) {
    //     combinations.emplace_back(
    //
    // }


    for (const auto &combination: combinations) {
        Eigen::Vector3d x = {combination[0] * stretch, 0, combination[1] * compression};
        auto sin = ((lambdas[0] - lambdas[2]) * x[0] * x[2]);
        auto cos = lambdas[0] * x[2] * x[2] + lambdas[2] * x[0] * x[0];
        Eigen::Matrix3d R{
            {cos, 0, -sin},
            {0, 1, 0},
            {sin, 0, cos}
        };
        Eigen::Vector3d t{
            (lambdas[0] - lambdas[2]) * x[0],
            0,
            -(lambdas[0] - lambdas[2]) * x[2]
        };
        solutions.push_back({rotationSign * U * R * V.transpose(), U * t});
    }
    for (const auto &combination: combinations) {
        Eigen::Vector3d x = {combination[0] * stretch, 0, combination[1] * compression};
        auto sin = ((lambdas[0] + lambdas[2]) * x[0] * x[2]);
        auto cos = lambdas[2] * x[0] * x[0] - lambdas[0] * x[2] * x[2];
        Eigen::Matrix3d R{
            {cos, 0, sin},
            {0, -1, 0},
            {sin, 0, -cos}
        };
        Eigen::Vector3d t{
            (lambdas[0] + lambdas[2]) * x[0],
            0,
            (lambdas[0] + lambdas[2]) * x[2]
        };
        solutions.push_back({rotationSign * U * R * V.transpose(), U * t});
    }

    Eigen::Matrix3d bestRot;
    Eigen::Vector3d bestTrans;
    std::vector<Eigen::Vector3d> reconstructedPts;
    reconstructedPts.reserve(aInliers.size());
    std::uint32_t bestNumPositive{0};
    double bestParallax = 0;
    std::uint32_t secondBstNumPositive{0};
    std::vector<bool> triangulated;
    int numInliers = 0;
    for (const auto &inlier: aInliers) {
        numInliers += inlier ? 1 : 0;
    }


    for (std::uint32_t i = 0; i < solutions.size(); ++i) {
        std::uint32_t numValid{0};
        std::vector<double> currentParallaxes;
        Eigen::Matrix4d transformMat = Eigen::Matrix4d::Identity();
        transformMat.block<3, 3>(0, 0) = std::get<0>(solutions[i]);
        transformMat.block<3, 1>(0, 3) = std::get<1>(solutions[i]);
        std::vector<Eigen::Vector3d> pts;
        std::vector<bool> currentTriangulated;
        pts.reserve(aInliers.size());

        for (std::uint32_t j = 0; j < aAllPoints[0].size(); ++j) {
            auto p1{aAllPoints[0][j]}, p2{aAllPoints[1][j]};
            const auto [p1_3D, p2_3D] = triangulate(p1, p2, transformMat);
            auto kp1 = aKeyPoints[0][j];
            auto kp2 = aKeyPoints[1][j];
            auto [valid, parallax] = isValid(p1_3D, p2_3D, std::get<0>(solutions[i]), std::get<1>(solutions[i]),
                                             aIntrinsics, kp1, kp2);
            pts.push_back(p1_3D);
            if (aInliers[j] && valid) {
                numValid++;
                currentTriangulated.push_back(parallax < 0.9999);
                currentParallaxes.push_back(parallax);
            } else {
                currentTriangulated.push_back(false);
            }
        }
        double parallax = 0;
        if (numValid > 0) {
            std::sort(currentParallaxes.begin(), currentParallaxes.end());
            size_t idx = std::min(50, static_cast<int>(currentParallaxes.size() - 1));
            parallax = std::acos(std::clamp(currentParallaxes[idx], -1.0, 1.0)) * 180.0 / CV_PI;
        }

        if (numValid > bestNumPositive) {
            secondBstNumPositive = bestNumPositive;
            bestNumPositive = numValid;
            bestRot = std::get<0>(solutions[i]);
            bestTrans = std::get<1>(solutions[i]);
            bestParallax = parallax;
            reconstructedPts = pts;
            triangulated = currentTriangulated;
        } else if (numValid > secondBstNumPositive) {
            secondBstNumPositive = numValid;
        }
    }

    if (secondBstNumPositive < 0.75 * bestNumPositive && bestNumPositive > 0.9 * numInliers && bestParallax > 1.0) {
        return std::optional<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<Eigen::Vector3d>, std::vector<
            bool> > >({
            bestRot, bestTrans, reconstructedPts, triangulated
        });
    }
    std::cout << "Best solution is not that much better" << std::endl;
    return std::nullopt;
}

void findHomography(
    const std::array<std::vector<Eigen::Vector3d>, 2> &aAllPoints,
    const std::vector<std::array<Eigen::Matrix<double, 8, 3>, 2> > &aCandidates,
    std::vector<bool> &aInliers, double &aScore,
    Eigen::Matrix3d &aHomographyMatrix, double aSigma) {
    std::size_t numCandidates{aAllPoints[0].size()};
    std::vector<bool> inliers(numCandidates, false);
    aScore = 0;
    aInliers.assign(numCandidates, false);
    aHomographyMatrix.setZero();

    for (std::uint32_t iter = 0; iter < aCandidates.size(); iter++) {
        auto candidates = aCandidates[iter];
        Eigen::Matrix3d homography = computeHomography(candidates);
        double score = checkHomography(homography, aAllPoints, inliers, aSigma);
        if (score > aScore) {
            aHomographyMatrix = homography;
            aInliers = inliers;
            aScore = score;
        }
    }
}

Eigen::Matrix3d
computeHomography(std::array<Eigen::Matrix<double, 8, 3>, 2> &aPoints) {
    Eigen::Matrix<double, 8, 9> res;
    res.setZero();
    for (std::uint32_t i = 0; i < 4; i++) {
        Eigen::Vector<double, 9> row1;
        Eigen::Vector<double, 9> row2;
        Eigen::Vector3d wx = aPoints[0](i, 2) * aPoints[1].row(i);
        Eigen::Vector3d yx = aPoints[0](i, 1) * aPoints[1].row(i);
        Eigen::Vector3d xx = aPoints[0](i, 0) * aPoints[1].row(i);

        row1 << 0, 0, 0, -wx, yx;
        row2 << wx, 0, 0, 0, -xx;
        res.row(i * 2) = row1;
        res.row(i * 2 + 1) = row2;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(res, Eigen::ComputeFullV);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> homography(
        svd.matrixV().col(8).data());
    return homography;
}

double
checkHomography(const Eigen::Matrix3d &aHomography,
                const std::array<std::vector<Eigen::Vector3d>, 2> &aAllPoints,
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

double calculateErrorHomography(const Eigen::Matrix3d &aHomography,
                                const Eigen::Vector3d &aP1,
                                const Eigen::Vector3d &aP2,
                                const double aInvSigmaSquare) {
    const double w2in1inv = 1.0 / aP2.dot(aHomography.row(2));
    const double u2in1 = aP2.dot(aHomography.row(0)) * w2in1inv;
    const double v2in1 = aP2.dot(aHomography.row(1)) * w2in1inv;
    const double squareDist1 = (aP1.x() - u2in1) * (aP1.x() - u2in1) +
                              (aP1.y() - v2in1) * (aP1.y() - v2in1);
    const double chiSquare1 = squareDist1 * aInvSigmaSquare;
    return chiSquare1;
}

// -----------------------------------------------------------------------
// ------------------------ Essential Matrix -----------------------------
// -----------------------------------------------------------------------

void findEssential(
    const std::array<std::vector<Eigen::Vector3d>, 2> &aAllPoints,
    const std::vector<std::array<Eigen::Matrix<double, 8, 3>, 2> > &aCandidates,
    std::vector<bool> &aInliers, double &aScore,
    Eigen::Matrix3d &aEssentialMatrix, double aSigma) {
    std::size_t numCandidates{aAllPoints[0].size()};
    std::vector<bool> inliers(numCandidates, false);
    aScore = 0;
    aInliers.assign(numCandidates, false);
    aEssentialMatrix.setZero();

#if SCHLAM_USE_OPENCV_RECONSTRUCTION
    if (numCandidates < 5 || aAllPoints[1].size() != numCandidates || aCandidates.empty() ||
        !std::isfinite(aSigma) || aSigma <= 0) {
        return;
    }
    std::array<std::vector<cv::Point2d>, 2> points;
    for (std::size_t camera = 0; camera < points.size(); ++camera) {
        points[camera].reserve(numCandidates);
        for (const auto& point : aAllPoints[camera]) {
            points[camera].emplace_back(point.x(), point.y());
        }
    }
    // Coordinates are already normalized by K^-1, so use unit focal length.
    // Match the existing symmetric epipolar test's normalized distance threshold.
    const cv::Mat models = cv::findEssentialMat(points[0], points[1], 1.0, cv::Point2d(0, 0),
        cv::RANSAC, 0.99, std::sqrt(3.841) * aSigma, static_cast<int>(aCandidates.size()));
    if (models.empty() || models.cols != 3 || models.rows % 3 != 0) {
        return;
    }
    for (int row = 0; row < models.rows; row += 3) {
        Eigen::Matrix3d essentialMatrix;
        cv::cv2eigen(models.rowRange(row, row + 3), essentialMatrix);
        // Preserve the score and mask convention used to compare against homography.
        const double score = checkEssential(essentialMatrix, aAllPoints, inliers, aSigma);
        if (score > aScore) {
            aEssentialMatrix = essentialMatrix;
            aInliers = inliers;
            aScore = score;
        }
    }
#else
    for (std::uint32_t iter = 0; iter < aCandidates.size(); iter++) {
        auto candidates = aCandidates[iter];
        Eigen::Matrix3d essentialMatrix = computeEssential(candidates);
        double score = checkEssential(essentialMatrix, aAllPoints, inliers, aSigma);
        if (score > aScore) {
            aEssentialMatrix = essentialMatrix;
            aInliers = inliers;
            aScore = score;
        }
    }
#endif
}

Eigen::Matrix3d
computeEssential(std::array<Eigen::Matrix<double, 8, 3>, 2> &aPoints) {
    Eigen::Matrix<double, 8, 9> res;
    for (int j = 0; j < 3; ++j) {
        res.block<8, 3>(0, j * 3) =
                aPoints[0].array().colwise() * aPoints[1].col(j).array();
    }
    auto svd = res.jacobiSvd(Eigen::ComputeFullV);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> essential(
        svd.matrixV().col(8).data());

    auto svd_e = essential.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d s = svd_e.singularValues();

    double avg_s = (s(0) + s(1)) / 2.0f;
    Eigen::DiagonalMatrix<double, 3> S_new(avg_s, avg_s, 0.0f);

    return svd_e.matrixU() * S_new * svd_e.matrixV().transpose();
}

double
checkEssential(const Eigen::Matrix3d &aEssentialMatrix,
               const std::array<std::vector<Eigen::Vector3d>, 2> &aAllPoints,
               std::vector<bool> &aInliers, const double aSigma) {
    double score{0};
    const size_t nPoints = aAllPoints[0].size();

    aInliers.assign(nPoints, false);

    constexpr double threshold = 3.841;
    constexpr double essentialScore = 5.991;
    const double invSigmaSquare = 1.0 / (aSigma * aSigma);

    const Eigen::Matrix3d essentialMatrixT = aEssentialMatrix.transpose();

    for (size_t i = 0; i < nPoints; i++) {
        const auto &p1 = aAllPoints[0][i];
        const auto &p2 = aAllPoints[1][i];

        Eigen::Vector3d line2 = aEssentialMatrix * p1;
        double error2 = calculateSymmetricErrorEssential(line2, p2, invSigmaSquare);

        Eigen::Vector3d line1 = essentialMatrixT * p2;
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

double calculateSymmetricErrorEssential(const Eigen::Vector3d &aLine,
                                        const Eigen::Vector3d &aPoint,
                                        double aInvSigmaSquare) {
    // Punkt-Linie-Abstand: (x' * L)^2 / (a^2 + b^2)
    // aLine = (a, b, c), wobei ax + by + c = 0
    double residual = aLine.dot(aPoint);
    double denominator = aLine.head<2>().squaredNorm();

    // Division durch Null verhindern, falls die Linie ungültig ist
    if (denominator < 1e-9) {
        return std::numeric_limits<double>::max();
    }

    return (residual * residual / denominator) * aInvSigmaSquare;
}

std::optional<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<Eigen::Vector3d>, std::vector<bool> > >
recoverPoseFromEssential(
    const Eigen::Matrix3d aEssential,
    const std::array<std::vector<Eigen::Vector3d>, 2> &aAllPoints,
    const std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>& aKeyPoints,
    const std::vector<bool> &aInliers, const Eigen::Matrix3d &aIntrinsics) {
    auto svd = aEssential.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Vector3d t = U.col(2);

    Eigen::Matrix3d W;
    W << 0, -1, 0,
            1, 0, 0,
            0, 0, 1;

    if (U.determinant() < 0) {
        U.col(2) = -U.col(2);
    }
    if (V.determinant() < 0) {
        V.col(2) = -V.col(2);
    }

    Eigen::Matrix3d R1 = U * W * V.transpose();
    Eigen::Matrix3d R2 = U * W.transpose() * V.transpose();

    std::array<Eigen::Matrix3d, 4> Rs = {R1, R1, R2, R2};
    std::array<Eigen::Vector3d, 4> ts = {t, -t, t, -t};

    Eigen::Matrix3d bestRot;
    Eigen::Vector3d bestTrans;
    std::vector<Eigen::Vector3d> reconstructedPts;
    std::vector<bool> triangulated;
    reconstructedPts.reserve(aInliers.size());

    std::uint32_t bestNumPositive = 0;
    double bestParallax = 0;
    std::uint32_t secondBstNumPositive = 0;
    int numInliers = 0;
    for (const auto &inlier: aInliers) {
        numInliers += inlier ? 1 : 0;
    }

    for (std::uint32_t i = 0; i < 4; ++i) {
        std::uint32_t numValid{0};
        Eigen::Matrix4d transformMat = Eigen::Matrix4d::Identity();
        transformMat.block<3, 3>(0, 0) = Rs[i];
        transformMat.block<3, 1>(0, 3) = ts[i]; //-Rs[i] * ts[i];
        std::vector<Eigen::Vector3d> pts;
        std::vector<bool> currentTriangulated;
        pts.reserve(aInliers.size());
        std::vector<double> currentParallaxes;

        for (std::uint32_t j = 0; j < aAllPoints[0].size(); ++j) {
            auto p1{aAllPoints[0][j]}, p2{aAllPoints[1][j]};
            const auto [p1_3D, p2_3D] = triangulate(p1, p2, transformMat);
            pts.push_back(p1_3D);
            auto kp1 = aKeyPoints[0][j];
            auto kp2 = aKeyPoints[1][j];
            auto [valid, parallax] = isValid(p1_3D, p2_3D, Rs[i], ts[i], aIntrinsics, kp1, kp2);
            if (aInliers[j] && valid) {
                currentParallaxes.push_back(parallax);
                numValid++;
                currentTriangulated.push_back(parallax < 0.99998);
            } else {
                currentTriangulated.push_back(false);
            }
        }
        double parallax = 0;
        if (numValid > 0) {
            std::sort(currentParallaxes.begin(), currentParallaxes.end());

            size_t idx = std::min(50, int(currentParallaxes.size() - 1));
            parallax = std::acos(std::clamp(currentParallaxes[idx], -1.0, 1.0)) * 180.0 / CV_PI;
        }
        if (numValid > bestNumPositive) {
            secondBstNumPositive = bestNumPositive;
            bestNumPositive = numValid;
            bestParallax = parallax;
            bestRot = Rs[i];
            bestTrans = ts[i];
            reconstructedPts = pts;
            triangulated = currentTriangulated;
        } else if (numValid > secondBstNumPositive) {
            secondBstNumPositive = numValid;
        }
    }

    std::cout << "Best num positive: " << bestNumPositive << std::endl;
    std::cout << "Second Best num positive: " << secondBstNumPositive << std::endl;
    std::cout << "Num inliers: " << numInliers << std::endl;
    std::cout << "Best parallax: " << bestParallax << std::endl;

    if ((secondBstNumPositive < (0.75 * bestNumPositive)) && (bestNumPositive > (0.9 * numInliers)) && bestParallax > 1.0) {
        // std::cout << "Best solution is valid" << std::endl;
        return std::optional<std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<Eigen::Vector3d>, std::vector<
            bool> > >({
            bestRot, bestTrans, reconstructedPts, triangulated
        });
    }
    std::cout << "Solution not clear enough, not returning pose" << std::endl;
    return std::nullopt;
}

std::pair<bool, double> isValid(const Eigen::Vector3d &p1_3D, const Eigen::Vector3d &p2_3D, const Eigen::Matrix3d &R,
                               const Eigen::Vector3d &t, const Eigen::Matrix3d &intrinsics,
                               const std::shared_ptr<KeyPoint>& aKP1, const std::shared_ptr<KeyPoint>& aKP2) {
    bool finite = std::isfinite(p1_3D.x()) && std::isfinite(p1_3D.y()) &&
                  std::isfinite(p1_3D.z());
    if (!finite) return {false, 0};

    Eigen::Vector3d O2 = -R.transpose() * t;

    Eigen::Vector3d normal1 = p1_3D;
    double dist1 = normal1.norm();

    Eigen::Vector3d normal2 = p1_3D - O2;
    double dist2 = normal2.norm();

    double cosParallax = normal1.dot(normal2) / (dist1 * dist2);
    if (!std::isfinite(cosParallax)) return {false, 0};

    constexpr double cosParallaxThreshold{0.9999}; // 0.99998
    // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
    if (p1_3D(2) <= 0 && cosParallax < cosParallaxThreshold)
        return {false, cosParallax};

    // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
    Eigen::Vector3d p3dC2 = R * p1_3D + t;
    //assert(p3dC2 == p2_3D);

    if (p3dC2(2) <= 0 && cosParallax < cosParallaxThreshold)
        return {false, cosParallax};

    // Check reprojection error in first image
    double im1x, im1y;
    double invZ1 = 1.0 / p1_3D(2);

    const double fx = intrinsics(0, 0);
    const double fy = intrinsics(1, 1);
    const double cx = intrinsics(0, 2);
    const double cy = intrinsics(1, 2);
    im1x = fx * p1_3D(0) * invZ1 + cx;
    im1y = fy * p1_3D(1) * invZ1 + cy;

    double squareError1 = (im1x - aKP1->getImgX()) * (im1x - aKP1->getImgX()) + (im1y - aKP1->getImgY()) * (
                             im1y - aKP1->getImgY());

    if (!std::isfinite(squareError1) || squareError1 > 4)
        return {false, cosParallax};

    // Check reprojection error in second image
    double im2x, im2y;
    double invZ2 = 1.0 / p3dC2(2);
    im2x = fx * p3dC2(0) * invZ2 + cx;
    im2y = fy * p3dC2(1) * invZ2 + cy;

    double squareError2 = (im2x - aKP2->getImgX()) * (im2x - aKP2->getImgX()) + (im2y - aKP2->getImgY()) * (
                             im2y - aKP2->getImgY());

    if (!std::isfinite(squareError2) || squareError2 > 4)
        return {false, cosParallax};

    // Low-parallax points support the pose, but are excluded from the triangulated mask.
    return {true, cosParallax};
}

std::array<Eigen::Vector3d, 2> triangulate(const Eigen::Vector3d &aP1, const Eigen::Vector3d &aP2,
                                           const Eigen::Matrix4d &aTransform) {
    Eigen::Matrix<double, 3, 4> P1;
    P1.setZero();
    P1.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 3, 4> P2;
    P2.setZero();
    P2.block<3, 3>(0, 0) = aTransform.block<3, 3>(0, 0);
    P2.block<3, 1>(0, 3) = aTransform.block<3, 1>(0, 3);

#if SCHLAM_USE_OPENCV_RECONSTRUCTION
    cv::Mat projection1, projection2;
    cv::eigen2cv(P1, projection1);
    cv::eigen2cv(P2, projection2);
    const cv::Mat point1 = (cv::Mat_<double>(2, 1) << aP1.x(), aP1.y());
    const cv::Mat point2 = (cv::Mat_<double>(2, 1) << aP2.x(), aP2.y());
    cv::Mat homogeneous;
    cv::triangulatePoints(projection1, projection2, point1, point2, homogeneous);
    Eigen::Vector4d pt1_3d;
    cv::cv2eigen(homogeneous, pt1_3d);
#else
    Eigen::Matrix4d A;
    A.row(0) = aP1.x() * P1.row(2) - P1.row(0);
    A.row(1) = aP1.y() * P1.row(2) - P1.row(1);
    A.row(2) = aP2.x() * P2.row(2) - P2.row(0);
    A.row(3) = aP2.y() * P2.row(2) - P2.row(1);

    auto svd = A.jacobiSvd(Eigen::ComputeFullV);
    auto pt1_3d = svd.matrixV().col(3);
#endif
    auto pt2_3d = aTransform * pt1_3d;

    Eigen::Vector3d pt1 = pt1_3d.head(3) / pt1_3d(3);
    Eigen::Vector3d pt2 = pt2_3d.head(3) / pt2_3d(3);

    return {pt1, pt2};
}

// ---------------------------------------------------------------------
// --------------------------- Common ----------------------------------
// ---------------------------------------------------------------------
std::vector<std::array<Eigen::Matrix<double, 8, 3>, 2> >
selectCandidates(const std::vector<Eigen::Vector3d> &aPoints1,
                 const std::vector<Eigen::Vector3d> &aPoints2,
                 std::uint32_t aNumCandidates) {
    std::vector<std::array<Eigen::Matrix<double, 8, 3>, 2> > candidates;
    candidates.reserve(aNumCandidates);

    for (std::uint32_t i = 0; i < aNumCandidates; ++i) {
        std::array<Eigen::Matrix<double, 8, 3>, 2> pair;

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
    auto& engine = utils::randomEngine();
    std::uniform_int_distribution<std::uint32_t> dist(0, aMaxIndex - 1);

    while (unique_indices.size() < aCount) {
        unique_indices.insert(dist(engine));
    }

    return {unique_indices.begin(), unique_indices.end()};
}
