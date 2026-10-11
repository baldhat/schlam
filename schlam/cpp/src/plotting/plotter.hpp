#pragma once

// local
#include "../tft/rigid_transform_3d.hpp"
#include "../tft/transformer.hpp"
#include "../data/image_data.hpp"
#include "schlam/Frame.h"

// pangolin
#include <pangolin/pangolin.h>

// eigen
#include <eigen3/Eigen/Core>

// STL
#include <vector>
#include <memory>
#include <mutex>
#include <string>
#include <optional>


class KeyPoint;

class Plotter {
public:
    Plotter(std::shared_ptr<tft::Transformer> apTransformer, const Eigen::Matrix3d& aCameraIntrinsics);

    ~Plotter() = default;

    bool shouldPause();

    // Update point cloud
    void updatePointCloud(const std::vector<Eigen::Vector3d> &points, const std::string aCF);

    void addTransform(const std::shared_ptr<tft::RigidTransform3D> transform);

    void updateFrustum(const std::shared_ptr<ImageData> aImageData);

    void updateTimestamp(const utils::TTimestamp& aTimestamp);

    void plotFeatures(const cv::Mat &aImage, const std::vector<std::shared_ptr<KeyPoint>>& aFeatures);

    void plotMatches(const cv::Mat& oldFrame, const cv::Mat& newFrame,
                     const std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>& aMatches);

    // Start visualization (runs in main thread)
    void run();

    void setup();

private:
    std::atomic_bool mPause{false};
    std::mutex mPauseMutex;

    Eigen::Matrix3d mCameraIntrinsics;

    std::shared_ptr<tft::Transformer> mpTransformer;
    std::vector<Eigen::Vector3d> mCloud;
    std::vector<std::shared_ptr<tft::RigidTransform3D> > mTransforms;

    std::shared_ptr<tft::RigidTransform3D> mFrustumPose;
    std::shared_ptr<ImageData> mFrustum;
    std::mutex mFrustumMutex;

    std::string mTimestampText;
    std::optional<utils::TTimestamp> mSequenceStartTimestamp;
    std::mutex mTimestampMutex;


    mutable std::unique_ptr<pangolin::GlTexture> m3DImageTexture;
    std::atomic_bool m3DImageChanged{false};

    // Feature Image
    mutable std::unique_ptr<pangolin::GlTexture> mFeatureImageTexture;
    cv::Mat mFeatureImage;
    std::atomic_bool mFeatureImageChanged{false};

    // Matcher Image
    mutable std::unique_ptr<pangolin::GlTexture> mMatcherImageTexture;
    cv::Mat mMatcherImage;
    std::atomic_bool mMatcherImageChanged{false};


    // Helper drawing functions
    void DrawGrid(int size, double step);

    void plotTransform(const std::shared_ptr<tft::RigidTransform3D> transform,
                       const double radius,
                       const double length,
                       const bool showFrameNames
    );

    void plotFrustum(std::shared_ptr<ImageData> aImageData, std::shared_ptr<tft::RigidTransform3D> aTransform,
                     double alpha);

    void setFeatureTexture(const cv::Mat &aImage);

    void setMatcherTexture(const cv::Mat &aImage);

    void showFeatures();

    void showMatches();

    void showTimestamp();

    pangolin::OpenGlMatrix GetPangolinModelMatrix(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) const;

    void drawCylinder(double radius, double length, int slices = 16);

    void drawAxes(const double radius, const double length);
};
