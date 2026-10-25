//
// Created by jonathan_michel on 10/24/26.
//

#include "DummyPlotter.h"

#include <thread>

bool DummyPlotter::shouldPause() {return false;}
void DummyPlotter::setMap(const Map* map) {}
void DummyPlotter::addTransform(const std::shared_ptr<tft::RigidTransform3D> transform) {}
void DummyPlotter::updateFrustum(const std::shared_ptr<ImageData> aImageData,
                   const std::shared_ptr<tft::RigidTransform3D> aPredictedPose) {}
void DummyPlotter::updateTimestamp(const utils::TTimestamp& aTimestamp) {}
void DummyPlotter::recordFrameTime(std::chrono::steady_clock::duration elapsed) {}
void DummyPlotter::resetCameraPaths(const Eigen::Vector3d& aReferencePosition) {}
void DummyPlotter::addGroundTruthCameraPosition(const Eigen::Vector3d& aPosition) {}
void DummyPlotter::addPredictedCameraPosition(const Eigen::Vector3d& aPosition) {}
void DummyPlotter::plotFeatures(const cv::Mat &aImage, const std::vector<std::shared_ptr<KeyPoint>>& aFeatures) {}
void DummyPlotter::plotMatches(const cv::Mat& oldFrame, const cv::Mat& newFrame,
                 const std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>& aMatches) {}
void DummyPlotter::run() { while (true) {std::this_thread::sleep_for(std::chrono::milliseconds(100));}}