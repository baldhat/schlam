//
// Created by baldhat on 2/26/26.
//

#include "Optimizer.h"

#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/parameter.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "parameter_cameraparameters.h"
#include "g2o/core/robust_kernel_impl.h"


Optimizer::Optimizer() {
    mOptimizer.setVerbose(false);

    g2o::OptimizationAlgorithmProperty solverProperty;
    mOptimizer.setAlgorithm(
        g2o::OptimizationAlgorithmFactory::instance()->construct(mSolverName,
                                                                 mSolverProperty));
}

std::tuple<std::vector<std::vector<KeyPoint> >, Eigen::Matrix3f, Eigen::Vector3f, std::vector<
    Eigen::Vector3f> >
Optimizer::optimize(const std::vector<std::vector<KeyPoint> > &aKeyPoints,
                    const Eigen::Matrix3f &aRotation,
                    const Eigen::Vector3f &aTranslation,
                    const std::vector<Eigen::Vector3f> &aPoints,
                    const Eigen::Matrix3f &aIntrinsics) {
    g2o::CameraParameters *cam_params =
            new g2o::CameraParameters(
                aIntrinsics(0, 0),
                Eigen::Vector2d(aIntrinsics(0, 2), aIntrinsics(1, 2)), 0.f);
    cam_params->setId(0);

    mOptimizer.addParameter(cam_params);

    std::vector<std::tuple<Eigen::Matrix3f, Eigen::Vector3f> > cams = {
        {Eigen::Matrix3f::Identity(3, 3), Eigen::Vector3f::Zero()},
        {aRotation, aTranslation}
    };
    int vertex_id{0};
    for (const auto &cam: cams) {
        g2o::VertexSE3Expmap *vertex = new g2o::VertexSE3Expmap();
        vertex->setId(vertex_id);
        auto [R, t] = cam;
        vertex->setEstimate(g2o::SE3Quat(R.cast<double>(), t.cast<double>()));

        if (vertex_id == 0) {
            vertex->setFixed(true);
        }

        vertex_id++;
    }

    for (const auto &point: aPoints) {
        g2o::VertexPointXYZ *v_point = new g2o::VertexPointXYZ();
        v_point->setId(vertex_id++);
        v_point->setEstimate(point.cast<double>());
        v_point->setMarginalized(true); // Required for BA Schur complement trick
        mOptimizer.addVertex(v_point);
    }

    for (int i = 0; i < cams.size(); i++) {
        for (int j = 0; j < aKeyPoints.size(); j++) {
            g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();

            // Connect to the Point (Vertex 0 in this edge type) and Pose (Vertex 1)
            edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(mOptimizer.vertex(i)));
            edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(mOptimizer.vertex(cams.size() + j)));

            // Set the 2D pixel measurement
            edge->setMeasurement(Eigen::Vector2d(aKeyPoints[i][j].getImgX(), aKeyPoints[i][j].getImgY()));

            // Set Information Matrix (Inverse Covariance).
            // Identity means we trust x and y equally and errors are uncorrelated.
            edge->setInformation(Eigen::Matrix2d::Identity());

            // Assign the camera parameters
            edge->setParameterId(0, 0);

            // Optional: Add a robust kernel (like Huber) to handle outliers
            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            edge->setRobustKernel(rk);

            mOptimizer.addEdge(edge);
        }
    }

    // OPTIMIZE
    mOptimizer.initializeOptimization();
    mOptimizer.setVerbose(true);
    mOptimizer.optimize(10);

    for (int i = 0; i < cams.size(); ++i) {
        g2o::VertexSE3Expmap* v_pose = static_cast<g2o::VertexSE3Expmap*>(mOptimizer.vertex(i));
        g2o::SE3Quat optimized_pose = v_pose->estimate();
        std::cout << std::endl;
    }

    for (int j = 0; j < aPoints.size(); ++j) {
        g2o::VertexPointXYZ* v_point = static_cast<g2o::VertexPointXYZ*>(mOptimizer.vertex(cams.size() + j));
        Eigen::Vector3d optimized_point = v_point->estimate();
        std::cout << std::endl;
    }
}
