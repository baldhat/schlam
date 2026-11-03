//
// Created by jonathan_michel on 10/24/26.
//

#include "KittiDataset.hpp"

#include "tft/rigid_transform_3d.hpp"
#include "data/data_helper.hpp"

#include <Eigen/Dense>

#include <csv.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

#include <yaml-cpp/yaml.h>

// STL
#include <filesystem>
#include <thread>
#include <chrono>
#include <fstream>
#include <stdexcept>


KittiDataset::KittiDataset(std::shared_ptr<tft::Transformer> aTransformer, const std::string& aSequence)
    : mImageQueue(3), mIMUQueue(2000), mGTQueue(2000), mSequence(aSequence) {
    const char *raw_val = std::getenv("KITTI_PATH");
    mDatasetPath = (raw_val != nullptr) ? raw_val : "";

    loadCamera0Config();
    aTransformer->registerTransform(std::make_shared<tft::RigidTransform3D>(
         mGTCF,mCamera0CF, mCamera0Rotation, mCamera0Translation));

    loadGTConfig();
    mImageLoader = std::thread(&KittiDataset::loadImageData, this);
    mGTLoader = std::thread(&KittiDataset::loadGTData, this);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

KittiDataset::~KittiDataset() {
    mStop = true;
    if (mImageLoader.joinable())
        mImageLoader.join();
    if (mGTLoader.joinable())
        mGTLoader.join();
}

double KittiDataset::getImageAspectRatio() {
    return 1241 / 376.0;
}

bool KittiDataset::hasIMU() {
    return false;
}

bool KittiDataset::hasGT() {
    return true;
}

bool KittiDataset::empty() {
    return (mImageQueue.empty() && mAllImagesRead) || (mIMUQueue.empty() && mAllGTRead) ||
               (mGTQueue.empty() && mAllGTRead);
}

float KittiDataset::percentageRead() {
    return static_cast<float>(100.0 * mNumImagesRead / mNumImages);
}

void KittiDataset::loadCamera0Config() {
    mCamera0Path = mDatasetPath / "data_odometry_calib/dataset/sequences" / mSequence ;
    auto calibFile = mCamera0Path / "calib.txt";
    std::ifstream file(calibFile);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + calibFile.string());
    }

    auto matrices = helpers::readLinesAsMatrices<3,4>(file);
    /*
     * PO
     * P1
     * P2
     * P3
     * Tr
     */
    mCamera0Intrinsics = matrices[0].block<3,3>(0, 0);
    //mCamera0Rotation = matrices[4].block<3,3>(0, 0);
    //mCamera0Translation = matrices[4].block<3, 1>(0, 3);
    mCamera0Rotation = Eigen::Matrix3d::Identity();
    mCamera0Translation = Eigen::Vector3d::Identity();
}

Eigen::Matrix3d KittiDataset::getCameraIntrinsics() {
    return mCamera0Intrinsics;
}


void KittiDataset::loadGTConfig() {
    auto poseFileName = mDatasetPath / "data_odometry_poses/dataset/poses" / (mSequence + ".txt");
    auto timeFileName = mDatasetPath / "data_odometry_calib/dataset/sequences" / mSequence /  + "times.txt";

    std::ifstream poseFile(poseFileName);
    if (!poseFile.is_open()) {
        throw std::runtime_error("Cannot open file: " + poseFileName.string());
    }
    mPoses = helpers::readLinesAsMatrices<3, 4>(poseFile, false);

    std::ifstream timeFile(timeFileName);
    if (!timeFile.is_open()) {
        throw std::runtime_error("Cannot open file: " + timeFileName.string());
    }
    std::string line;
    while (std::getline(timeFile, line)) {
        std::istringstream stream(line);
        std::string timeSeconds;
        stream >> timeSeconds;
        mTimePoints.emplace_back(helpers::stringSecondsToTimePoint(timeSeconds));
    }
}

void KittiDataset::loadImageData() {
    std::vector<std::filesystem::path> filesInDirectory;
    auto imagesPath = mDatasetPath / "data_odometry_gray/dataset/sequences" / mSequence / "image_0";
    auto imageFilepaths = std::filesystem::directory_iterator(imagesPath);
    std::copy(imageFilepaths, std::filesystem::directory_iterator(), std::back_inserter(filesInDirectory));
    std::sort(filesInDirectory.begin(), filesInDirectory.end());
    mNumImages = filesInDirectory.size();

    std::size_t index = 0;
    for (auto &filepath: filesInDirectory) {
        const std::string imageFile = filepath.string();
        utils::TTimestamp timestamp = mTimePoints.at(index);

        // already rectified
        cv::Mat image = cv::imread(imageFile, cv::IMREAD_GRAYSCALE);
        auto imageData = std::make_shared<ImageData>(
            timestamp, image, mCamera0Intrinsics, mCamera0CF);

        while (!mStop && mImageQueue.try_push_back(imageData) != boost::queue_op_status::success);
        index++;
    }
    mAllImagesRead = true;
}

void KittiDataset::loadGTData() {
    assert(mPoses.size() == mTimePoints.size());
    for (std::size_t i = 0; i < mPoses.size(); i++) {
        auto pose = mPoses.at(i);
        auto position = pose.block<3, 1>(0, 3);
        auto rotation = pose.block<3, 3>(0,0);
        auto velocity = Eigen::Vector3d(std::nanf(""), std::nanf(""), std::nanf(""));
        auto gtData = std::make_shared<GTData>(mTimePoints.at(i), position, rotation, velocity, mGTCF);

        while (!mStop && mGTQueue.try_push_back(gtData) != boost::queue_op_status::success);
    }
    mAllGTRead = true;
}

std::shared_ptr<ImageData> KittiDataset::getNextImageData() {
    auto imageData = mImageQueue.pull_front();
    mNumImagesRead++;
    return imageData;
}

std::shared_ptr<IMUData> KittiDataset::getNextIMUData() {
    return nullptr;
}

std::shared_ptr<GTData> KittiDataset::getNextGTData() {
    return mGTQueue.pull_front();
}

std::chrono::system_clock::time_point
stringNanoToTimePoint(const std::string &nano_str) {
    long long nanos = std::stoll(nano_str);
    std::chrono::nanoseconds duration(nanos);
    return std::chrono::system_clock::time_point(
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            duration));
}

std::chrono::system_clock::time_point
longNanoToTimePoint(const std::uint64_t &aNanos) {
    std::chrono::nanoseconds duration(aNanos);
    return std::chrono::system_clock::time_point(
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            duration));
}