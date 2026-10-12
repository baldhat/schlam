//
// Created by baldhat on 2/26/26.
//

#include "Optimizer.h"

#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/parameter.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "linear_solver_csparse.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"

G2O_USE_OPTIMIZATION_LIBRARY(eigen);
G2O_USE_OPTIMIZATION_LIBRARY(dense);

namespace {
// The reference pose is fixed at identity, so ||t|| is the camera baseline.
// One scalar constraint removes scale freedom without fixing its direction.
class EdgeBaselineLength : public g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void computeError() override {
        const auto* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        _error[0] = pose->estimate().translation().norm() / _measurement - 1.0;
    }

    void linearizeOplus() override {
        const auto* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const auto& translation = pose->estimate().translation();
        const double length = translation.norm();
        _jacobianOplusXi.setZero();
        if (length > 0.0) {
            // g2o applies a left SE(3) update: rotation preserves ||t||.
            _jacobianOplusXi.rightCols<3>() = translation.transpose() / (length * _measurement);
        }
    }

    bool read(std::istream& stream) override {
        stream >> _measurement >> information()(0, 0);
        return static_cast<bool>(stream);
    }

    bool write(std::ostream& stream) const override {
        stream << measurement() << ' ' << information()(0, 0);
        return static_cast<bool>(stream);
    }
};
}


Optimizer::Optimizer() {

}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, std::vector<Eigen::Vector3d> >
Optimizer::optimizePoseAndPoints(const std::vector<std::vector<std::shared_ptr<KeyPoint>>>& aKeyPoints,
                    const Eigen::Matrix3d &aRotation,
                    const Eigen::Vector3d &aTranslation,
                    const std::vector<Eigen::Vector3d> &aPoints,
                    const Eigen::Matrix3d &aIntrinsics) {
    const double initialBaseline = aTranslation.norm();
    if (!std::isfinite(initialBaseline) || initialBaseline <= 0.0) {
        throw std::invalid_argument("Joint pose and point optimization requires a finite, nonzero baseline");
    }

    g2o::SparseOptimizer optimizer;
    g2o::OptimizationAlgorithmProperty solverProp;
    //optimizer.setVerbose(true);
    optimizer.setAlgorithm(
        g2o::OptimizationAlgorithmFactory::instance()->construct("lm_fix6_3",
                                                                 solverProp)
    );

    std::vector<std::tuple<Eigen::Matrix3d, Eigen::Vector3d> > cams = {
        {Eigen::Matrix3d::Identity(3, 3), Eigen::Vector3d::Zero()},
        {aRotation, aTranslation}
    };
    int vertex_id{0};
    // Add camera poses
    for (const auto &cam: cams) {
        auto [R, t] = cam;
        g2o::SE3Quat pose(R.cast<double>(), t.cast<double>());
        g2o::VertexSE3Expmap *vertex = new g2o::VertexSE3Expmap();
        vertex->setId(vertex_id);
        if (vertex_id == 0) {
            vertex->setFixed(true);
        }
        vertex->setEstimate(pose);
        optimizer.addVertex(vertex);
        vertex_id++;
    }

    auto* baselineEdge = new EdgeBaselineLength;
    baselineEdge->setVertex(0, optimizer.vertex(1));
    baselineEdge->setMeasurement(initialBaseline);
    baselineEdge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    optimizer.addEdge(baselineEdge);

    // Add 3D points
    for (const auto &point: aPoints) {
        g2o::VertexPointXYZ *v_point = new g2o::VertexPointXYZ();
        v_point->setId(vertex_id++);
        v_point->setEstimate(point.cast<double>());
        v_point->setMarginalized(true); // Required for BA Schur complement trick
        optimizer.addVertex(v_point);
    }

    // Create edges between poses and points
    for (int i = 0; i < cams.size(); i++) {
        for (int j = 0; j < aKeyPoints[0].size(); j++) {
            g2o::EdgeSE3ProjectXYZ *edge = new g2o::EdgeSE3ProjectXYZ();
            edge->setVertex(0, optimizer.vertex(cams.size() + j));
            edge->setVertex(1, optimizer.vertex(i));
            edge->fx = aIntrinsics(0, 0);
            edge->fy = aIntrinsics(1, 1);
            edge->cx = aIntrinsics(0, 2);
            edge->cy = aIntrinsics(1, 2);
            edge->setMeasurement(Eigen::Vector2d(aKeyPoints[i][j]->getImgX(), aKeyPoints[i][j]->getImgY()));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);

            optimizer.addEdge(edge);
        }
    }

    // OPTIMIZE
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(50);

    g2o::VertexSE3Expmap* v_pose = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1));
    g2o::SE3Quat optimized_pose = v_pose->estimate();
    Eigen::Matrix3d R = optimized_pose.rotation().toRotationMatrix().cast<double>();
    Eigen::Vector3d t = optimized_pose.translation().cast<double>();


    std::vector<Eigen::Vector3d> optimized_points;
    for (int j = 0; j < aPoints.size(); ++j) {
        g2o::VertexPointXYZ* v_point = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(cams.size() + j));
        Eigen::Vector3d optimized_point = v_point->estimate();
        optimized_points.push_back(optimized_point);
    }

    return {R, t, optimized_points};
}


std::tuple<Eigen::Matrix3d, Eigen::Vector3d>
Optimizer::optimizePose(const std::vector<std::shared_ptr<KeyPoint>>& aKeyPoints,
                    const Eigen::Matrix3d &aRotation,
                    const Eigen::Vector3d &aTranslation,
                    const std::vector<Eigen::Vector3d> &aPoints,
                    const Eigen::Matrix3d &aIntrinsics) {
    const auto result = optimizePoseWithInliers(aKeyPoints, aRotation, aTranslation, aPoints, aIntrinsics);
    return {result.rotation, result.translation};
}

PoseOptimizationResult
Optimizer::optimizePoseWithInliers(const std::vector<std::shared_ptr<KeyPoint>>& aKeyPoints,
                                 const Eigen::Matrix3d& aRotation,
                                 const Eigen::Vector3d& aTranslation,
                                 const std::vector<Eigen::Vector3d>& aPoints,
                                 const Eigen::Matrix3d& aIntrinsics) {
    if (aKeyPoints.size() != aPoints.size()) {
        throw std::invalid_argument("Pose optimization requires one observation per map point");
    }
    for (const auto& keyPoint : aKeyPoints) {
        if (!keyPoint) {
            throw std::invalid_argument("Pose optimization observations must not be null");
        }
    }
    if (aPoints.empty()) {
        return {aRotation, aTranslation, {}};
    }

    g2o::SparseOptimizer optimizer;
    //optimizer.setVerbose(true);
    g2o::OptimizationAlgorithmProperty solverProp;
    auto* algorithm = g2o::OptimizationAlgorithmFactory::instance()->construct("lm_dense6_3", solverProp);

    optimizer.setAlgorithm(algorithm);

    auto* pose = new g2o::VertexSE3Expmap;
    pose->setId(0);
    pose->setFixed(false);
    pose->setEstimate(g2o::SE3Quat(aRotation, aTranslation));
    optimizer.addVertex(pose);

    // Match the ORB detector's default 1.2x pyramid, with one-pixel sigma at level 0.
    constexpr double pyramidScaleFactor = 1.2;
    constexpr double monocularChi2Threshold = 5.991;
    const double huberDelta = std::sqrt(monocularChi2Threshold);

    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> projectionEdges;
    projectionEdges.reserve(aPoints.size());
    for (std::size_t j = 0; j < aPoints.size(); ++j) {
        auto* edge = new g2o::EdgeSE3ProjectXYZOnlyPose;
        edge->setVertex(0, pose);
        edge->Xw = aPoints[j];
        edge->fx = aIntrinsics(0, 0);
        edge->fy = aIntrinsics(1, 1);
        edge->cx = aIntrinsics(0, 2);
        edge->cy = aIntrinsics(1, 2);
        edge->setMeasurement(Eigen::Vector2d(aKeyPoints[j]->getImgX(), aKeyPoints[j]->getImgY()));
        const double inverseVariance = std::pow(pyramidScaleFactor, -2.0 * aKeyPoints[j]->getLevel());
        edge->setInformation(Eigen::Matrix2d::Identity() * inverseVariance);
        auto* robustKernel = new g2o::RobustKernelHuber;
        robustKernel->setDelta(huberDelta);
        edge->setRobustKernel(robustKernel);
        optimizer.addEdge(edge);
        projectionEdges.push_back(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(50);

    // Squared reprojection error normalized by the observation's octave variance.
    const auto isInlier = [&](g2o::EdgeSE3ProjectXYZOnlyPose* edge) {
        edge->computeError();
        const double chi2 = edge->chi2();
        return edge->isDepthPositive() && std::isfinite(chi2) && chi2 <= monocularChi2Threshold;
    };
    std::vector<bool> inliers(projectionEdges.size(), false);
    std::size_t inlierCount = 0;
    for (std::size_t j = 0; j < projectionEdges.size(); ++j) {
        auto* edge = projectionEdges[j];
        inliers[j] = isInlier(edge);
        edge->setLevel(inliers[j] ? 0 : 1);
        if (inliers[j]) {
            ++inlierCount;
        }
    }

    if (inlierCount >= 3 && optimizer.initializeOptimization(0)) {
        std::cout << "Optimizing again with " << (projectionEdges.size() - inlierCount)
                  << " outliers removed" << std::endl;
        for (std::size_t j = 0; j < projectionEdges.size(); ++j) {
            if (inliers[j]) projectionEdges[j]->setRobustKernel(nullptr);
        }
        optimizer.optimize(50);
    }

    // Residuals change during refinement; only final, previously accepted
    // observations may update the map's descriptors.
    for (std::size_t j = 0; j < projectionEdges.size(); ++j) {
        inliers[j] = inliers[j] && isInlier(projectionEdges[j]);
    }
    const auto& optimizedPose = pose->estimate();
    const Eigen::Matrix3d rotation = optimizedPose.rotation().toRotationMatrix();
    const Eigen::Vector3d translation = optimizedPose.translation();
    if (!rotation.allFinite() || !translation.allFinite()) {
        return {aRotation, aTranslation, std::vector<bool>(aPoints.size(), false)};
    }
    return {rotation, translation, inliers};
}
