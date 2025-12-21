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
    void addTransform(const std::shared_ptr<tft::RigidTransform3D> transform);

    // Start visualization (runs in main thread)
    void run();

    void setup();

private:
    std::shared_ptr<tft::Transformer> mpTransformer;
    std::vector<Eigen::Vector3d> mCloud;

    // Helper drawing functions
    void DrawGrid(int size, float step);

    void plotTransform(const std::shared_ptr<tft::RigidTransform3D> transform, 
                       const double radius, 
                       const double length,
                       const bool showFrameNames
    );

    pangolin::OpenGlMatrix GetPangolinModelMatrix(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

    void drawCylinder(float radius, float length, int slices = 16);
    void drawAxes(const double radius, const double length);
};
