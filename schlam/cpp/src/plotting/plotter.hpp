#pragma once

// local
#include "../tft/rigid_transform_3d.hpp"
#include "../tft/transformer.hpp"
#include "../data/image_data.hpp"

// pangolin
#include <pangolin/pangolin.h>

// eigen
#include <eigen3/Eigen/Core>

// STL
#include <vector>

class KeyPoint;

class Plotter {
public:
    Plotter(std::shared_ptr<tft::Transformer> apTransformer);

    ~Plotter() = default;

    // Update point cloud
    void updatePointCloud(const std::vector<Eigen::Vector3f> &points, const std::string aCF);

    void addTransform(const std::shared_ptr<tft::RigidTransform3D> transform);

    void addFrustum(const std::shared_ptr<ImageData> aImageData);

    void plotFeatures(const cv::Mat &aImage, const std::vector<KeyPoint> &aFeatures);

    void plotMatches(const cv::Mat &aImage1, const cv::Mat &aImage2, const std::vector<KeyPoint> &aFeatures1,
                     const std::vector<KeyPoint> &aFeatures2, const std::vector<std::array<std::uint32_t, 2>> aMatches);

    // Start visualization (runs in main thread)
    void run();

    void setup();

private:
    std::shared_ptr<tft::Transformer> mpTransformer;
    std::vector<Eigen::Vector3f> mCloud;
    std::vector<std::shared_ptr<tft::RigidTransform3D> > mTransforms;
    std::vector<std::shared_ptr<ImageData> > mFrustums;
    mutable std::unique_ptr<pangolin::GlTexture> m3DImageTexture;

    // Feature Image
    mutable std::unique_ptr<pangolin::GlTexture> mFeatureImageTexture;
    cv::Mat mFeatureImage;
    std::atomic_bool mFeatureImageChanged{false};

    // Matcher Image
    mutable std::unique_ptr<pangolin::GlTexture> mMatcherImageTexture;
    cv::Mat mMatcherImage;
    std::atomic_bool mMatcherImageChanged{false};


    // Helper drawing functions
    void DrawGrid(int size, float step);

    void plotTransform(const std::shared_ptr<tft::RigidTransform3D> transform,
                       const double radius,
                       const double length,
                       const bool showFrameNames
    );

    void plotFrustum(std::shared_ptr<ImageData> aImageData, double alpha) const;

    void setFeatureTexture(const cv::Mat &aImage);

    void setMatcherTexture(const cv::Mat &aImage);

    void showFeatures();

    void showMatches();

    pangolin::OpenGlMatrix GetPangolinModelMatrix(const Eigen::Matrix3f &R, const Eigen::Vector3f &t) const;

    void drawCylinder(float radius, float length, int slices = 16);

    void drawAxes(const double radius, const double length);
};
