//
// Created by baldhat on 2/26/26.
//

#include "Optimizer.h"

#include <algorithm>
#include <cmath>
#include <istream>
#include <ostream>
#include <set>
#include <stdexcept>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/parameter.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "CovisibilityHelpers.h"
#include "Map.h"
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
    optimizer.setVerbose(false);
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

std::tuple<Eigen::Matrix3d, Eigen::Vector3d>
Optimizer::optimizePose(const std::vector<std::shared_ptr<KeyPoint>>& keyPoints,
                        const Eigen::Matrix3d& rotation,
                        const Eigen::Vector3d& translation,
                        const std::vector<Eigen::Vector3d>& points,
                        const Eigen::Matrix3d& intrinsics) {
    auto result = optimizePoseWithInliers(keyPoints, rotation, translation, points, intrinsics);
    return {result.rotation, result.translation};
}

bool Optimizer::optimizeFullMap(Map* map, bool removeOutlierPoints) {
    if (!map || map->mKeyFrames.size() < 2 || map->mPoints.empty()) return false;

    constexpr KeyFrameId referenceKeyFrameId = 0;
    constexpr KeyFrameId initialKeyFrameId = 1;
    const auto& referencePose = *map->mKeyFrames.at(referenceKeyFrameId).mMapToCamera;
    const double initialBaseline = map->mKeyFrames.at(initialKeyFrameId).mMapToCamera->mTranslation.norm();
    if (!referencePose.matrix().isIdentity(1e-10)) {
        throw std::invalid_argument("Full-map optimization requires the identity reference pose");
    }
    if (!std::isfinite(initialBaseline) || initialBaseline <= 0.0) {
        throw std::invalid_argument("Full-map optimization requires a finite, nonzero initialization baseline");
    }

    g2o::SparseOptimizer optimizer;
    g2o::OptimizationAlgorithmProperty solverProp;
    //optimizer.setVerbose(true);
    optimizer.setAlgorithm(
        g2o::OptimizationAlgorithmFactory::instance()->construct("lm_fix6_3",
                                                                 solverProp)
    );


    int vertex_id{0};
    // Add camera poses
    std::unordered_map<KeyFrameId, int> frameToVertexId;
    for (const auto &[frameId, keyframe]: map->mKeyFrames) {
        auto transform = keyframe.mMapToCamera;
        auto R = transform->mRotation;
        auto t = transform->mTranslation;
        g2o::SE3Quat pose(R.cast<double>(), t.cast<double>());
        g2o::VertexSE3Expmap *vertex = new g2o::VertexSE3Expmap();
        vertex->setId(vertex_id);
        if (frameId == referenceKeyFrameId) {
            vertex->setFixed(true);
        }
        vertex->setEstimate(pose);
        optimizer.addVertex(vertex);
        frameToVertexId.insert({frameId, vertex_id});
        vertex_id++;
    }

    auto* baselineEdge = new EdgeBaselineLength;
    baselineEdge->setVertex(0, optimizer.vertex(frameToVertexId.at(initialKeyFrameId)));
    baselineEdge->setMeasurement(initialBaseline);
    baselineEdge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    optimizer.addEdge(baselineEdge);


    // Add 3D points
    std::unordered_map<MapPointId, int> pointToVertexId;
    for (const auto &[pointId, point]: map->mPoints) {
        g2o::VertexPointXYZ *v_point = new g2o::VertexPointXYZ();
        v_point->setId(vertex_id);
        v_point->setEstimate(point.mPosition.cast<double>());
        v_point->setMarginalized(true); // Required for BA Schur complement trick
        optimizer.addVertex(v_point);
        pointToVertexId.insert({pointId, vertex_id});
        vertex_id++;
    }

    struct ProjectionObservation {
        MapPointId pointId;
        g2o::EdgeSE3ProjectXYZ* edge;
        double inverseVariance;
    };
    std::vector<ProjectionObservation> projectionObservations;

    // Create edges between poses and points
    for (const auto& [frameId, frame] : map->mKeyFrames) {
        for (const auto& [pointId, point] : map->mPoints) {
            if (!point.mObservations.contains(frameId)) continue;

            g2o::EdgeSE3ProjectXYZ *edge = new g2o::EdgeSE3ProjectXYZ();
            edge->setVertex(0, optimizer.vertex(pointToVertexId.at(pointId)));
            edge->setVertex(1, optimizer.vertex(frameToVertexId.at(frameId)));
            edge->fx = frame.mIntrinsics(0, 0);
            edge->fy = frame.mIntrinsics(1, 1);
            edge->cx = frame.mIntrinsics(0, 2);
            edge->cy = frame.mIntrinsics(1, 2);
            auto keypoint = frame.mKeyPoints.at(point.mObservations.at(frameId));
            edge->setMeasurement(Eigen::Vector2d(keypoint->getImgX(), keypoint->getImgY()));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);

            optimizer.addEdge(edge);
            if (removeOutlierPoints) {
                constexpr double pyramidScaleFactor = 1.2;
                const double inverseVariance = std::pow(pyramidScaleFactor, -2.0 * keypoint->getLevel());
                projectionObservations.push_back({pointId, edge, inverseVariance});
            }
        }
    }

    // OPTIMIZE
    if (!optimizer.initializeOptimization()) return false;
    //optimizer.setVerbose(true);
    if (optimizer.optimize(50) <= 0) return false;

    // Validate before publishing to the map
    for (const auto& [frameId, vertexId] : frameToVertexId) {
        const auto* pose = static_cast<const g2o::VertexSE3Expmap*>(optimizer.vertex(vertexId));
        if (!pose->estimate().rotation().coeffs().allFinite() ||
            !pose->estimate().translation().allFinite()) return false;
    }
    for (const auto& [pointId, vertexId] : pointToVertexId) {
        const auto* point = static_cast<const g2o::VertexPointXYZ*>(optimizer.vertex(vertexId));
        if (!point->estimate().allFinite()) return false;
    }

    for (auto& [frameId, frame] : map->mKeyFrames) {
        g2o::VertexSE3Expmap* v_pose = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(frameToVertexId.at(frameId)));
        g2o::SE3Quat optimized_pose = v_pose->estimate();
        Eigen::Matrix3d R = optimized_pose.rotation().toRotationMatrix().cast<double>();
        Eigen::Vector3d t = optimized_pose.translation().cast<double>();
        map->updateKeyFramePose(frameId, tft::RigidTransform3D(frame.mMapToCamera->mSource, frame.mMapToCamera->mTarget, R, t));
    }

    for (auto& [pointId, point] : map->mPoints) {
        g2o::VertexPointXYZ* v_point = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(pointToVertexId.at(pointId)));
        Eigen::Vector3d optimized_point = v_point->estimate();
        map->updateMapPointPosition(pointId, optimized_point);
    }

    if (removeOutlierPoints) {
        constexpr double monocularChi2Threshold = 5.991;
        std::set<MapPointId> outlierPointIds;
        std::unordered_map<MapPointId, std::size_t> inlierObservationCounts;
        for (const auto& observation : projectionObservations) {
            observation.edge->computeError();
            const double normalizedChi2 = observation.edge->chi2() * observation.inverseVariance;
            if (observation.edge->isDepthPositive() && std::isfinite(normalizedChi2) &&
                normalizedChi2 <= monocularChi2Threshold)
                ++inlierObservationCounts[observation.pointId];
        }
        for (const auto& [pointId, point] : map->mPoints) {
            const auto inlierCount = inlierObservationCounts[pointId];
            if (inlierCount < 2 || inlierCount * 5 < point.mObservations.size() * 4)
                outlierPointIds.insert(pointId);
        }
        for (const auto pointId : outlierPointIds) map->removePoint(pointId);
        std::cout << "Removed " << outlierPointIds.size() << " outlier map points" << std::endl;
    }
    return true;
}


bool Optimizer::optimizeCovisibilityGraph(Map* map, KeyFrame& keyframe, std::size_t maxNumVariable, std::size_t maxNumFixed) {
    if (!map || maxNumVariable == 0 || maxNumFixed < 2 ||
        !map->mKeyFrames.contains(keyframe.mId)) return false;

    auto neighbors = getCovisibilityNeighbors(map, keyframe.mId, maxNumVariable - 1, 15);

    std::set<KeyFrameId> localFrames{keyframe.mId};
    std::set<MapPointId> candidatePoints(keyframe.mObservedPoints.begin(), keyframe.mObservedPoints.end());

    for (const auto& [neighborId, count] : neighbors) {
        localFrames.insert(neighborId);
        const auto& neighbor = map->mKeyFrames.at(neighborId);
        candidatePoints.insert(neighbor.mObservedPoints.begin(), neighbor.mObservedPoints.end());
    }

    // A boundary frame's score is the number of candidate points it observes.
    std::unordered_map<KeyFrameId, std::uint32_t> boundaryVotes;
    for (const auto pointId : candidatePoints) {
        for (const auto& [frameId, featureIndex] : map->mPoints.at(pointId).mObservations) {
            if (!localFrames.contains(frameId)) ++boundaryVotes[frameId];
        }
    }
    std::vector<Neighbor> boundaryCandidates(boundaryVotes.begin(), boundaryVotes.end());
    const auto stronger = [](const Neighbor& a, const Neighbor& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    };
    const auto fixedKeep = std::min(boundaryCandidates.size(), maxNumFixed);
    std::partial_sort(boundaryCandidates.begin(), boundaryCandidates.begin() + fixedKeep,
                      boundaryCandidates.end(), stronger);
    boundaryCandidates.resize(fixedKeep);

    std::set<KeyFrameId> boundaryFrames;
    std::set<KeyFrameId> fixedFrames;
    for (const auto& [frameId, votes] : boundaryCandidates) {
        boundaryFrames.insert(frameId);
        fixedFrames.insert(frameId);
    }

    // The map coordinate system is defined by keyframe 0. Keep it fixed even
    // when it is selected as a local neighbor rather than a boundary frame.
    constexpr KeyFrameId referenceKeyFrameId = 0;
    if (localFrames.contains(referenceKeyFrameId)) fixedFrames.insert(referenceKeyFrameId);

    const auto cameraCenter = [&](KeyFrameId id) {
        const auto& pose = *map->mKeyFrames.at(id).mMapToCamera;
        return (-pose.mRotation.transpose() * pose.mTranslation).eval();
    };
    const auto hasScaleAnchor = [&](const std::set<KeyFrameId>& frames) {
        for (auto first = frames.begin(); first != frames.end(); ++first) {
            for (auto second = std::next(first); second != frames.end(); ++second) {
                if ((cameraCenter(*first) - cameraCenter(*second)).squaredNorm() > 1e-12)
                    return true;
            }
        }
        return false;
    };
    // Fix older local poses when the bounded boundary has too few anchors.
    for (const auto frameId : localFrames) {
        if (hasScaleAnchor(fixedFrames) || fixedFrames.size() >= maxNumFixed) break;
        if (frameId != keyframe.mId) fixedFrames.insert(frameId);
    }
    if (!hasScaleAnchor(fixedFrames)) return false;

    std::set<MapPointId> optimizedPoints;
    std::set<KeyFrameId> activeFixedFrames;
    for (const auto pointId : candidatePoints) {
        const auto& point = map->mPoints.at(pointId);
        std::size_t includedObservations = 0;
        bool hasVariableObservation = false;
        for (const auto& [frameId, featureIndex] : point.mObservations) {
            if (localFrames.contains(frameId) || boundaryFrames.contains(frameId)) {
                ++includedObservations;
                if (localFrames.contains(frameId) && !fixedFrames.contains(frameId))
                    hasVariableObservation = true;
            }
        }
        if (includedObservations < 2 || !hasVariableObservation) continue;
        optimizedPoints.insert(pointId);
        for (const auto& [frameId, featureIndex] : point.mObservations) {
            if (fixedFrames.contains(frameId)) activeFixedFrames.insert(frameId);
        }
    }
    if (optimizedPoints.empty() || !hasScaleAnchor(activeFixedFrames)) return false;

    g2o::SparseOptimizer optimizer;
    g2o::OptimizationAlgorithmProperty solverProp;
    optimizer.setAlgorithm(
        g2o::OptimizationAlgorithmFactory::instance()->construct("lm_fix6_3",
                                                                 solverProp)
    );

    int vertex_id{0};
    // Create point vertices
    std::unordered_map<MapPointId, int> pointToVertexId;
    for (const auto &pointId: optimizedPoints) {
        g2o::VertexPointXYZ *v_point = new g2o::VertexPointXYZ();
        v_point->setId(vertex_id);
        const auto& point = map->mPoints.at(pointId);
        v_point->setEstimate(point.mPosition.cast<double>());
        v_point->setMarginalized(true);
        optimizer.addVertex(v_point);
        pointToVertexId.insert({pointId, vertex_id});
        vertex_id++;
    }

    std::unordered_map<KeyFrameId, int> frameToVertexId;
    // Add local poses, including any poses fixed to anchor the local problem.
    for (const auto &frameId: localFrames) {
        auto& frame = map->mKeyFrames.at(frameId);
        auto transform = frame.mMapToCamera;
        auto R = transform->mRotation;
        auto t = transform->mTranslation;
        g2o::SE3Quat pose(R.cast<double>(), t.cast<double>());
        g2o::VertexSE3Expmap *vertex = new g2o::VertexSE3Expmap();
        vertex->setId(vertex_id);
        vertex->setEstimate(pose);
        vertex->setFixed(fixedFrames.contains(frameId));
        optimizer.addVertex(vertex);
        frameToVertexId.insert({frameId, vertex_id});
        vertex_id++;
    }
    // add fixed frames
    for (const auto &frameId: boundaryFrames) {
        auto& frame = map->mKeyFrames.at(frameId);
        auto transform = frame.mMapToCamera;
        auto R = transform->mRotation;
        auto t = transform->mTranslation;
        g2o::SE3Quat pose(R.cast<double>(), t.cast<double>());
        g2o::VertexSE3Expmap *vertex = new g2o::VertexSE3Expmap();
        vertex->setId(vertex_id);
        vertex->setEstimate(pose);
        vertex->setFixed(true);
        optimizer.addVertex(vertex);
        frameToVertexId.insert({frameId, vertex_id});
        vertex_id++;
    }


    // Create edges between poses and points
    for (const auto& pointId : optimizedPoints) {
        const auto& point = map->mPoints.at(pointId);
        for (const auto& [frameId, _] : point.mObservations) {
            if (boundaryFrames.contains(frameId) || localFrames.contains(frameId)) {
                const auto& frame = map->mKeyFrames.at(frameId);
                g2o::EdgeSE3ProjectXYZ *edge = new g2o::EdgeSE3ProjectXYZ();
                edge->setVertex(0, optimizer.vertex(pointToVertexId.at(pointId)));
                edge->setVertex(1, optimizer.vertex(frameToVertexId.at(frameId)));
                edge->fx = frame.mIntrinsics(0, 0);
                edge->fy = frame.mIntrinsics(1, 1);
                edge->cx = frame.mIntrinsics(0, 2);
                edge->cy = frame.mIntrinsics(1, 2);
                auto keypoint = frame.mKeyPoints.at(point.mObservations.at(frameId));
                edge->setMeasurement(Eigen::Vector2d(keypoint->getImgX(), keypoint->getImgY()));
                edge->setInformation(Eigen::Matrix2d::Identity());
                edge->setRobustKernel(new g2o::RobustKernelHuber);

                optimizer.addEdge(edge);
            }
        }
    }

    std::cout << "Optimizing Covisibility graph with: " << optimizedPoints.size() << " points, "
              << localFrames.size() - (fixedFrames.size() - boundaryFrames.size())
              << " variable frames and " << fixedFrames.size() << " fixed frames" << std::endl;

    // OPTIMIZE
    if (!optimizer.initializeOptimization()) return false;
    optimizer.setVerbose(false);
    optimizer.setForceStopFlag(&stopCovisibility_);
    optimizer.optimize(50);
    stopCovisibility_ = false;

    // Validate before publishing to the map
    for (const auto& [frameId, vertexId] : frameToVertexId) {
        const auto* pose = static_cast<const g2o::VertexSE3Expmap*>(optimizer.vertex(vertexId));
        if (!pose->estimate().rotation().coeffs().allFinite() ||
            !pose->estimate().translation().allFinite()) return false;
    }
    for (const auto& [pointId, vertexId] : pointToVertexId) {
        const auto* point = static_cast<const g2o::VertexPointXYZ*>(optimizer.vertex(vertexId));
        if (!point->estimate().allFinite()) return false;
    }

    for (const auto frameId : localFrames) {
        if (fixedFrames.contains(frameId)) continue;
        auto& frame = map->mKeyFrames.at(frameId);
        g2o::VertexSE3Expmap* v_pose = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(frameToVertexId.at(frameId)));
        g2o::SE3Quat optimized_pose = v_pose->estimate();
        Eigen::Matrix3d R = optimized_pose.rotation().toRotationMatrix().cast<double>();
        Eigen::Vector3d t = optimized_pose.translation().cast<double>();
        map->updateKeyFramePose(frameId, tft::RigidTransform3D(frame.mMapToCamera->mSource, frame.mMapToCamera->mTarget, R, t));
    }

    for (auto& pointId : optimizedPoints) {
        g2o::VertexPointXYZ* v_point = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(pointToVertexId.at(pointId)));
        Eigen::Vector3d optimized_point = v_point->estimate();
        map->updateMapPointPosition(pointId, optimized_point);
    }
    return true;

}

void Optimizer::stopCovisibility() {
    stopCovisibility_ = true;
}
