#pragma once

#include <pangolin/pangolin.h>
#include <eigen3/Eigen/Core>
#include <vector>

class Plotter {
public:
    Plotter();
    ~Plotter() = default;

    // Update point cloud
    void updatePointCloud(const std::vector<Eigen::Vector3f>& points);

    // Start visualization (runs in main thread)
    void run();

    void setup();

private:
    std::vector<Eigen::Vector3f> mCloud;

    // Helper drawing functions
    void DrawGrid(int size, float step);
    void DrawPoints(const std::vector<Eigen::Vector3f>& points);
};
