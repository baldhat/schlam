//
// Created by jonathan_michel on 10/24/26.
//

#ifndef SCHLAM_PLOTTERINTERFACE_H
#define SCHLAM_PLOTTERINTERFACE_H

#include "schlam/Map.h"

class IPlotter {
public:
    virtual bool shouldPause() = 0;
    virtual void setMap(const Map* map) = 0;
    virtual void addTransform(const std::shared_ptr<tft::RigidTransform3D> transform) = 0;
    // The predicted pose maps the camera into world coordinates.
    virtual void updateFrustum(const std::shared_ptr<ImageData> aImageData,
                       const std::shared_ptr<tft::RigidTransform3D> aPredictedPose = nullptr) = 0;
    virtual void updateTimestamp(const utils::TTimestamp& aTimestamp) = 0;
    virtual void recordFrameTime(std::chrono::steady_clock::duration elapsed) = 0;
    // Camera positions are expressed in world coordinates.
    virtual void resetCameraPaths(const Eigen::Vector3d& aReferencePosition) = 0;
    virtual void addGroundTruthCameraPosition(const Eigen::Vector3d& aPosition) = 0;
    virtual void addPredictedCameraPosition(const Eigen::Vector3d& aPosition) = 0;
    virtual void plotFeatures(const cv::Mat &aImage, const std::vector<std::shared_ptr<KeyPoint>>& aFeatures) = 0;
    virtual void plotMatches(const cv::Mat& oldFrame, const cv::Mat& newFrame,
                     const std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>& aMatches) = 0;
    // Start visualization (runs in main thread)
    virtual void run() = 0;
};

#endif //SCHLAM_PLOTTERINTERFACE_H
