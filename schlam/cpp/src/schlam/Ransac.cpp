//
// Created by baldhat on 2/8/26.
//

#include "Ransac.h"
#include "utils.h"

#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Dense>

#include <optional>
#include <random>
#include <set>
#include <opencv2/core/eigen.hpp>

std::optional<std::tuple<Eigen::Matrix3f, Eigen::Vector3f, std::vector<Eigen::Vector3f> > > reconstructInitial(
    const std::vector<KeyPoint> aKeypoints1,
    const std::vector<KeyPoint> aKeypoints2,
    const Eigen::Matrix3f aIntrinsics) {
    assert(aKeypoints1.size() == aKeypoints2.size());

    if (aKeypoints1.size() < 8 || aKeypoints2.size() < 8) {
        std::cout << "WARN: Not enough matches found!" << std::endl;
        return std::nullopt;
    }

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
    if (rh > 0.45) {
        // cv::Mat ho;
        // cv::eigen2cv(homography, ho);
        //cv::Mat
        // cv::decomposeHomographyMat()
        return recoverPoseFromHomography(homography, {points1, points2}, inliersHomography);
    } else {
        cv::Mat R1;
        cv::Mat R2;
        cv::Vec3f t;
        cv::Mat ess;
        cv::eigen2cv(essential, ess);
        cv::decomposeEssentialMat(ess, R1, R2, t);
        std::cout << "OpenCV: " << R1 << " or " << R2 << std::endl;
        auto output = recoverPoseFromEssential(essential, {points1, points2}, inliersEssential);
        if (output.has_value()) {
            auto [R, t, pts] = output.value();
            std::cout << "Ours: " << R << std::endl;
        }
        return output;
    }
}

// ------------------------------------------------------------
// ---------------------- Homography --------------------------
// ------------------------------------------------------------
std::optional<std::tuple<Eigen::Matrix3f, Eigen::Vector3f, std::vector<Eigen::Vector3f> > > recoverPoseFromHomography(
    const Eigen::Matrix3f aHomography,
    const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
    const std::vector<bool> &aInliers) {
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(aHomography, Eigen::ComputeFullV);
    auto normalizedHomography = (1.0 / svd.singularValues()[1]) * aHomography;
    auto normalizedSVD = normalizedHomography.jacobiSvd(Eigen::ComputeEigenvectors | Eigen::ComputeFullV);
    auto lambdas = normalizedSVD.singularValues();
    auto stretch = sqrt(
        (lambdas[0] * lambdas[0] - lambdas[1] * lambdas[1]) / (lambdas[0] * lambdas[0] - lambdas[2] * lambdas[2]));
    auto compression = sqrt(
        (lambdas[1] * lambdas[1] - lambdas[2] * lambdas[2]) / (lambdas[0] * lambdas[0] - lambdas[2] * lambdas[2]));
    //auto V = normalizedSVD.matrixV();
    //auto norm = sqrt(stretch * stretch + compression * compression);
    //stretch /= norm;
    //compression /= norm;
    std::vector<std::tuple<Eigen::Matrix3f, Eigen::Vector3f> > solutions;
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
        Eigen::Vector3f x = {combination[0] * stretch, 0, combination[1] * compression};
        auto sin = ((lambdas[0] - lambdas[2]) * x[0] * x[2]);
        auto cos = lambdas[0] * x[2] * x[2] + lambdas[2] * x[0] * x[0];
        Eigen::Matrix3f R{
            {cos, 0, -sin},
            {0, 1, 0},
            {sin, 0, cos}
        };
        Eigen::Vector3f t{
            (lambdas[0] - lambdas[2]) * x[0],
            0,
            -(lambdas[0] - lambdas[2]) * x[2]
        };
        solutions.push_back({R, t});
    }
    for (const auto &combination: combinations) {
        Eigen::Vector3f x = {combination[0] * stretch, 0, combination[1] * compression};
        auto sin = ((lambdas[0] + lambdas[2]) * x[0] * x[2]);
        auto cos = lambdas[0] * x[2] * x[2] + lambdas[1] * x[0] * x[0];
        Eigen::Matrix3f R{
            {cos, 0, sin},
            {0, -1, 0},
            {-sin, 0, -cos}
        };
        Eigen::Vector3f t{
            (lambdas[0] + lambdas[2]) * x[0],
            0,
            (lambdas[0] + lambdas[2]) * x[2]
        };
        solutions.push_back({R, t});
    }

    Eigen::Matrix3f bestRot;
    Eigen::Vector3f bestTrans;
    std::vector<Eigen::Vector3f> reconstructedPts;
    reconstructedPts.reserve(aInliers.size());
    std::uint32_t bestNumPositive{0};
    std::uint32_t secondBstNumPositive{0};
    int numInliers = 0;
    for (const auto &inlier: aInliers) {
        numInliers += inlier ? 1 : 0;
    }


    for (std::uint32_t i = 0; i < solutions.size(); ++i) {
        std::uint32_t numPositive{0};
        Eigen::Matrix4f transformMat = Eigen::Matrix4f::Identity();
        transformMat.block<3, 3>(0, 0) = std::get<0>(solutions[i]);
        transformMat.block<3, 1>(0, 3) = -std::get<0>(solutions[i]) * std::get<1>(solutions[i]);
        std::vector<Eigen::Vector3f> pts;
        pts.reserve(aInliers.size());


        for (std::uint32_t j = 0; j < aAllPoints[0].size(); ++j) {
            if (!aInliers[j]) continue;
            auto p1{aAllPoints[0][j]}, p2{aAllPoints[1][j]};
            const auto [p1_3D, p2_3D] = triangulate(p1, p2, transformMat);
            pts.push_back(p1_3D);
            if (p1_3D.z() > 0 && p2_3D.z() > 0) {
                numPositive++;
            }
        }
        std::cout << "Num positive: " << numPositive << std::endl;
        if (numPositive > bestNumPositive) {
            secondBstNumPositive = bestNumPositive;
            bestNumPositive = numPositive;
            bestRot = std::get<0>(solutions[i]);
            bestTrans = std::get<1>(solutions[i]); //-Rs[i]*ts[i]; // TODO: Why different to what is used above?
            reconstructedPts = pts;
        } else if (numPositive > secondBstNumPositive) {
            secondBstNumPositive = numPositive;
        }
    }

    std::cout << "Best num positive: " << bestNumPositive << std::endl;
    std::cout << "Second Best num positive: " << secondBstNumPositive << std::endl;
    std::cout << "Num inliers: " << numInliers << std::endl;
    if (secondBstNumPositive < 0.75 * bestNumPositive && bestNumPositive > 0.9 * numInliers) {
        std::cout << "Best solution is valid" << std::endl;
        return std::optional<std::tuple<Eigen::Matrix3f, Eigen::Vector3f, std::vector<Eigen::Vector3f> > >({
            bestRot, bestTrans, reconstructedPts
        });
    }
    std::cout << "Best solution is not that much better" << std::endl;
    return std::nullopt;
}

void findHomography(
    const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
    const std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2> > &aCandidates,
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
    const std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2> > &aCandidates,
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
    auto svd = res.jacobiSvd(Eigen::ComputeFullV);
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> essential(
        svd.matrixV().col(8).data());

    auto svd_e = essential.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3f s = svd_e.singularValues();

    float avg_s = (s(0) + s(1)) / 2.0f;
    Eigen::DiagonalMatrix<float, 3> S_new(avg_s, avg_s, 0.0f);

    return svd_e.matrixU() * S_new * svd_e.matrixV().transpose();
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
    if (denominator < 1e-9) {
        return 0.0;
    }

    return (residual * residual / denominator) * aInvSigmaSquare;
}

std::optional<std::tuple<Eigen::Matrix3f, Eigen::Vector3f, std::vector<Eigen::Vector3f> > > recoverPoseFromEssential(
    const Eigen::Matrix3f aEssential,
    const std::array<std::vector<Eigen::Vector3f>, 2> &aAllPoints,
    const std::vector<bool> &aInliers) {
    auto essentialSVD = aEssential.jacobiSvd(Eigen::ComputeFullV);
    auto t = essentialSVD.matrixV().col(2);
    Eigen::Matrix3f skewMat;
    skewMat.block<1, 3>(0, 0) = Eigen::Vector3f{0, -t[2], t[1]};
    skewMat.block<1, 3>(1, 0) = Eigen::Vector3f{t[2], 0, -t[0]};
    skewMat.block<1, 3>(2, 0) = Eigen::Vector3f{-t[1], t[0], 0};
    auto svd = (aEssential * skewMat).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto ur = svd.matrixU();
    auto vr = svd.matrixV();
    auto R1 = ur * vr.transpose();
    auto R1_ = R1 * R1.determinant();
    ur.col(2) = -ur.col(2);
    auto R2 = ur * vr.transpose();
    auto R2_ = R2 * R2.determinant();

    std::array<Eigen::Vector3f, 4> ts = {t, t, -t, -t};
    std::array<Eigen::Matrix3f, 4> Rs = {R1_, R2_, R1_, R2_};

    Eigen::Matrix3f bestRot;
    Eigen::Vector3f bestTrans;
    std::vector<Eigen::Vector3f> reconstructedPts;
    reconstructedPts.reserve(aInliers.size());
    std::uint32_t bestNumPositive{0};
    std::uint32_t secondBstNumPositive{0};
    int numInliers = 0;
    for (const auto &inlier: aInliers) {
        numInliers += inlier ? 1 : 0;
    }

    for (std::uint32_t i = 0; i < 4; ++i) {
        std::uint32_t numPositive{0};
        Eigen::Matrix4f transformMat = Eigen::Matrix4f::Identity();
        transformMat.block<3, 3>(0, 0) = Rs[i];
        transformMat.block<3, 1>(0, 3) = -Rs[i] * ts[i];
        std::vector<Eigen::Vector3f> pts;
        pts.reserve(aInliers.size());

        for (std::uint32_t j = 0; j < aAllPoints[0].size(); ++j) {
            if (!aInliers[j]) continue;
            auto p1{aAllPoints[0][j]}, p2{aAllPoints[1][j]};
            const auto [p1_3D, p2_3D] = triangulate(p1, p2, transformMat);
            pts.push_back(p1_3D);
            if (p1_3D.z() > 0 && p2_3D.z() > 0) {
                numPositive++;
            }
        }
        if (numPositive > bestNumPositive) {
            secondBstNumPositive = bestNumPositive;
            bestNumPositive = numPositive;
            bestRot = Rs[i];
            bestTrans = ts[i]; //-Rs[i]*ts[i]; // TODO: Why different to what is used above?
            reconstructedPts = pts;
        } else if (numPositive > secondBstNumPositive) {
            secondBstNumPositive = numPositive;
        }
    }

    std::cout << "Best num positive: " << bestNumPositive << std::endl;
    std::cout << "Second Best num positive: " << secondBstNumPositive << std::endl;
    std::cout << "Num inliers: " << numInliers << std::endl;
    if (secondBstNumPositive < 0.75 * bestNumPositive && bestNumPositive > 0.9 * numInliers) {
        std::cout << "Best solution is valid" << std::endl;
        return std::optional<std::tuple<Eigen::Matrix3f, Eigen::Vector3f, std::vector<Eigen::Vector3f> > >({
            bestRot, bestTrans, reconstructedPts
        });
    } else {
        std::cout << "Best solution is not that much better" << std::endl;
        return std::nullopt;
    }
}

std::array<Eigen::Vector3f, 2> triangulate(const Eigen::Vector3f &aP1, const Eigen::Vector3f &aP2,
                                           const Eigen::Matrix4f &aTransform) {
    Eigen::Matrix<float,3,4> P1;
    P1.setZero();
    P1.block<3,3>(0,0) = Eigen::Matrix3f::Identity();

    Eigen::Matrix<float,3,4> P2;
    P2.setZero();
    P2.block<3,3>(0,0) = aTransform.block<3,3>(0,0);
    P2.block<3,1>(0,3) = aTransform.block<3,1>(0,3);

    Eigen::Matrix4f A;
    A.row(0) = aP1.x() * P1.row(2) - P1.row(0);
    A.row(1) = aP1.y() * P1.row(2) - P1.row(1);
    A.row(2) = aP2.x() * P2.row(2) - P2.row(0);
    A.row(3) = aP2.y() * P2.row(2) - P2.row(1);

    auto svd = A.jacobiSvd(Eigen::ComputeFullV);
    auto pt1_3d = svd.matrixV().col(3);
    auto pt2_3d = aTransform * pt1_3d;

    Eigen::Vector3f pt1 = pt1_3d.head(3)/pt1_3d(3);
    Eigen::Vector3f pt2 = pt2_3d.head(3)/pt2_3d(3);

    return {pt1, pt2};
}

// ---------------------------------------------------------------------
// --------------------------- Common ----------------------------------
// ---------------------------------------------------------------------
std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2> >
selectCandidates(const std::vector<Eigen::Vector3f> &aPoints1,
                 const std::vector<Eigen::Vector3f> &aPoints2,
                 std::uint32_t aNumCandidates) {
    std::vector<std::array<Eigen::Matrix<float, 8, 3>, 2> > candidates;
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
