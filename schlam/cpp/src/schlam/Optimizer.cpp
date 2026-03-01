//
// Created by baldhat on 2/26/26.
//

#include "Optimizer.h"

#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/parameter.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "linear_solver_csparse.h"
#include "parameter_cameraparameters.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

G2O_USE_OPTIMIZATION_LIBRARY(eigen);
G2O_USE_OPTIMIZATION_LIBRARY(dense);


Optimizer::Optimizer() {

}

std::tuple<Eigen::Matrix3f, Eigen::Vector3f, std::vector<Eigen::Vector3f> >
Optimizer::optimize(const std::vector<std::vector<KeyPoint> > &aKeyPoints,
                    const Eigen::Matrix3f &aRotation,
                    const Eigen::Vector3f &aTranslation,
                    const std::vector<Eigen::Vector3f> &aPoints,
                    const Eigen::Matrix3f &aIntrinsics) {
    g2o::SparseOptimizer optimizer;
    g2o::OptimizationAlgorithmProperty solverProp;
    optimizer.setVerbose(true);
    optimizer.setAlgorithm(
        g2o::OptimizationAlgorithmFactory::instance()->construct("lm_fix6_3",
                                                                 solverProp)
    );

    g2o::CameraParameters *cam_params =
            new g2o::CameraParameters(
                aIntrinsics(0, 0),
                Eigen::Vector2d(aIntrinsics(0, 2), aIntrinsics(1, 2)), 0.f);
    cam_params->setId(0);

    optimizer.addParameter(cam_params);

    std::vector<std::tuple<Eigen::Matrix3f, Eigen::Vector3f> > cams = {
        {Eigen::Matrix3f::Identity(3, 3), Eigen::Vector3f::Zero()},
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
            g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();

            // Connect to the Point (Vertex 0 in this edge type) and Pose (Vertex 1)
            edge->setVertex(0, optimizer.vertex(cams.size() + j));
            edge->setVertex(1, optimizer.vertex(i));

            // Set the 2D pixel measurement
            edge->setMeasurement(Eigen::Vector2d(aKeyPoints[i][j].getImgX(), aKeyPoints[i][j].getImgY()));

            // Set Information Matrix (Inverse Covariance).
            // Identity means we trust x and y equally and errors are uncorrelated.
            edge->setInformation(Eigen::Matrix2d::Identity());

            // Assign the camera parameters
            edge->setParameterId(0, 0);

            // Optional: Add a robust kernel (like Huber) to handle outliers
            edge->setRobustKernel(new g2o::RobustKernelHuber);

            optimizer.addEdge(edge);
        }
    }

    // OPTIMIZE
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(10);

    g2o::VertexSE3Expmap* v_pose = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1));
    g2o::SE3Quat optimized_pose = v_pose->estimate();
    Eigen::Matrix3f R = optimized_pose.rotation().toRotationMatrix().cast<float>();
    Eigen::Vector3f t = optimized_pose.translation().cast<float>();


    std::vector<Eigen::Vector3f> optimized_points;
    for (int j = 0; j < aPoints.size(); ++j) {
        g2o::VertexPointXYZ* v_point = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(cams.size() + j));
        Eigen::Vector3d optimized_point = v_point->estimate();
        optimized_points.push_back(optimized_point.cast<float>());
    }

    return {R, t, optimized_points};
}
