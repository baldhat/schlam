#pragma once

#include "../tft/rigid_transform_3d.hpp"
#include "../tft/transformer.hpp"

#include <pangolin/pangolin.h>
#include <eigen3/Eigen/Core>
#include <vector>

class Plotter {
public:
    Plotter(std::shared_ptr<tft::Transformer> apTransformer);
    ~Plotter() = default;

    // Update point cloud
    void updatePointCloud(const std::vector<Eigen::Vector3d>& points);
    void addTransform(const tft::RigidTransform3D& points);

    // Start visualization (runs in main thread)
    void run();

    void setup();

private:
    std::shared_ptr<tft::Transformer> mpTransformer;
    std::vector<Eigen::Vector3d> mCloud;
    std::vector<tft::RigidTransform3D> mTransforms;

    // Helper drawing functions
    void DrawGrid(int size, float step);

    void plotTransform(const tft::RigidTransform3D& transform);

    pangolin::OpenGlMatrix GetPangolinModelMatrix(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
};
