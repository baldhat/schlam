//
// Created by jonathan_michel on 10/24/26.
//

#ifndef SCHLAM_DUMMYPLOTTER_H
#define SCHLAM_DUMMYPLOTTER_H
#include "IPlotter.hpp"


class DummyPlotter : public IPlotter {
public:
    bool shouldPause() override;
    void setMap(const Map* map) override;
    void addTransform(const std::shared_ptr<tft::RigidTransform3D> transform) override;
    void updateFrustum(const std::shared_ptr<ImageData> aImageData,
                       const std::shared_ptr<tft::RigidTransform3D> aPredictedPose = nullptr) override;
    void updateTimestamp(const utils::TTimestamp& aTimestamp) override;
    void recordFrameTime(std::chrono::steady_clock::duration elapsed) override;
    void resetCameraPaths(const Eigen::Vector3d& aReferencePosition) override;
    void addGroundTruthCameraPosition(const Eigen::Vector3d& aPosition) override;
    void addPredictedCameraPosition(const Eigen::Vector3d& aPosition) override;
    void plotFeatures(const cv::Mat &aImage, const std::vector<std::shared_ptr<KeyPoint>>& aFeatures) override;
    void plotMatches(const cv::Mat& oldFrame, const cv::Mat& newFrame,
                     const std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>& aMatches) override;
    void run() override;
};


#endif //SCHLAM_DUMMYPLOTTER_H
