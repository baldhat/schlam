#pragma once

#include "data/DatasetInterface.hpp"

#include <filesystem>
#include <string>

class FFMPEGDataset final : public DatasetInterface {
public:
    FFMPEGDataset(const std::filesystem::path& calibrationFile, const std::string& source);
    ~FFMPEGDataset();

    double getImageAspectRatio() override;
    bool hasIMU() override { return false; }
    bool hasGT() override { return false; }
    std::shared_ptr<ImageData> getNextImageData() override;
    std::shared_ptr<IMUData> getNextIMUData() override { return nullptr; }
    std::shared_ptr<GTData> getNextGTData() override { return nullptr; }
    bool empty() override { return !stream_; }
    Eigen::Matrix3d getCameraIntrinsics() override { return intrinsics_; }
    float percentageRead() override { return 0.0f; }

private:
    struct Stream;
    std::unique_ptr<Stream> stream_;
    Eigen::Matrix3d intrinsics_;
    cv::Size imageSize_;
    cv::Mat undistortMap1_;
    cv::Mat undistortMap2_;
};
