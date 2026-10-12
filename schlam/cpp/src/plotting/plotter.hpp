#pragma once

// local
#include "../tft/rigid_transform_3d.hpp"
#include "../tft/transformer.hpp"
#include "../data/image_data.hpp"
#include "schlam/Frame.h"
#include "schlam/MapTypes.h"

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
#include <unordered_map>


class KeyPoint;
class Map;

class Plotter {
public:
    Plotter(std::shared_ptr<tft::Transformer> apTransformer, const Eigen::Matrix3d& aCameraIntrinsics);

    ~Plotter() = default;

    bool shouldPause();

    // The caller owns the map and must keep it alive while visualization runs.
    void setMap(const Map* map);

    void addTransform(const std::shared_ptr<tft::RigidTransform3D> transform);

    // The predicted pose maps the camera into world coordinates.
    void updateFrustum(const std::shared_ptr<ImageData> aImageData,
                       const std::shared_ptr<tft::RigidTransform3D> aPredictedPose = nullptr);

    void updateTimestamp(const utils::TTimestamp& aTimestamp);

    // Camera positions are expressed in world coordinates.
    void resetCameraPaths(const Eigen::Vector3d& aReferencePosition);

    void addGroundTruthCameraPosition(const Eigen::Vector3d& aPosition);

    void addPredictedCameraPosition(const Eigen::Vector3d& aPosition);

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
    const Map* mpMap{nullptr};
    std::shared_ptr<tft::RigidTransform3D> mMapToWorld;
    std::uint64_t mMapRevision{0};
    std::mutex mMapMutex;
    std::uint64_t mKeyFrameTextureMapRevision{0};
    std::unordered_map<KeyFrameId, std::unique_ptr<pangolin::GlTexture>> mKeyFrameTextures;
    std::vector<std::shared_ptr<tft::RigidTransform3D> > mTransforms;

    std::shared_ptr<tft::RigidTransform3D> mFrustumPose;
    std::shared_ptr<tft::RigidTransform3D> mPredictedFrustumPose;
    std::shared_ptr<ImageData> mFrustum;
    // Camera paths and frustums must be rendered from the same pose state.
    std::mutex mCameraVisualizationMutex;

    std::string mTimestampText;
    std::optional<utils::TTimestamp> mSequenceStartTimestamp;
    std::mutex mTimestampMutex;

    std::vector<Eigen::Vector3d> mGroundTruthCameraPath;
    std::vector<Eigen::Vector3d> mPredictedCameraPath;


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

    void plotFrustum(const cv::Mat& image, const Eigen::Matrix3d& intrinsics,
                     const std::shared_ptr<tft::RigidTransform3D>& transform,
                     pangolin::GlTexture& texture, double alpha);

    void drawCameras(bool showGroundTruthPath, bool showPredictedPath,
                     bool showCam0Frustum, bool showPredictedFrustum, double alpha);

    // Called with mCameraVisualizationMutex held by drawCameras.
    void drawFrustums(bool aShowCam0, bool aShowPredicted, double aAlpha);

    void drawMap(bool showKeyFrameFrustums, bool showFrameNames, double alpha);

    void setFeatureTexture(const cv::Mat &aImage);

    void setMatcherTexture(const cv::Mat &aImage);

    void showFeatures();

    void showMatches();

    void showTimestamp();

    void drawCameraPaths(bool aShowGroundTruth, bool aShowPredicted);

    pangolin::OpenGlMatrix GetPangolinModelMatrix(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) const;

    void drawCylinder(double radius, double length, int slices = 16);

    void drawAxes(const double radius, const double length);
};
