#include "mav_dataloader.hpp"
#include "src/tft/rigid_transform_3d.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <csv.h>
#include <opencv2/imgcodecs.hpp>

#include <thread>
#include <yaml-cpp/yaml.h>

// STL
#include <filesystem>
#include <iostream>

MAVDataloader::MAVDataloader(const std::filesystem::path &aDatasetPath,
                             std::shared_ptr<tft::Transformer> aTransformer)
    : mDatasetPath(aDatasetPath), mImageQueue(3), mIMUQueue(20) {
    loadCamera0Config();
    aTransformer->registerTransform(std::make_shared<tft::RigidTransform3D>(
        mCamera0CF, mIMUCF, mCamera0Rotation, mCamera0Translation));
    aTransformer->registerTransform(std::make_shared<tft::RigidTransform3D>(
        mGTCamera0CF, mGTCF, mCamera0Rotation, mCamera0Translation));

    loadIMUConfig();
    aTransformer->registerTransform(std::make_shared<tft::RigidTransform3D>(
        mIMUCF, mCamera0CF, mCamera0Rotation, mCamera0Translation));

    mImageLoader = std::thread(std::bind(&MAVDataloader::loadImageData, this));
    mIMULoader = std::thread(std::bind(&MAVDataloader::loadIMUData, this));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

MAVDataloader::~MAVDataloader() {
    if (mImageLoader.joinable())
        mImageLoader.join();
    if (mIMULoader.joinable())
        mIMULoader.join();
}

bool MAVDataloader::empty() { return mImageQueue.empty() || mIMUQueue.empty(); }

void MAVDataloader::loadCamera0Config() {
    mCamera0Path = mDatasetPath / "cam0";
    YAML::Node camera0Config = YAML::LoadFile(mCamera0Path / "sensor.yaml");
    float fu = camera0Config["intrinsics"][0].as<float>(0.0);
    float fv = camera0Config["intrinsics"][1].as<float>(0.0);
    float cu = camera0Config["intrinsics"][2].as<float>(0.0);
    float cv = camera0Config["intrinsics"][3].as<float>(0.0);
    mCamera0Intrinsics << fu, 0, cu, 0, fv, cv, 0, 0, 1;

    auto ec = camera0Config["T_BS"]["data"].as<std::vector<double> >();
    mCamera0Rotation << ec[0], ec[1], ec[2], ec[4], ec[5], ec[6], ec[8], ec[9],
            ec[10];
    mCamera0Translation = Eigen::Vector3f(ec[3], ec[7], ec[11]);
}

void MAVDataloader::loadIMUConfig() {
    YAML::Node imuConfig = YAML::LoadFile(mDatasetPath / "imu0" / "sensor.yaml");
    double mGyroscopeNoiseDensity =
            imuConfig["gyroscope_noise_density"].as<double>();
    double mGyroscopeRandomWalk = imuConfig["gyroscope_random_walk"].as<double>();
    double mAccelerometerNoiseDensity =
            imuConfig["accelerometer_noise_density"].as<double>();
    double mAccelerometerRandomWalk =
            imuConfig["accelerometer_random_walk"].as<double>();

    auto imuDataPath = (mDatasetPath / "imu0" / "data.csv").string();
    mIMUCSVReader = std::make_shared<io::CSVReader<7> >(imuDataPath);
    mIMUCSVReader->read_header(io::ignore_extra_column, "#timestamp [ns]",
                               "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]",
                               "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]",
                               "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]");

    auto gtDataPath =
            (mDatasetPath / "state_groundtruth_estimate0" / "data.csv").string();
    mGTCSVReader = std::make_shared<io::CSVReader<11> >(gtDataPath);
    mGTCSVReader->read_header(
        io::ignore_extra_column, "#timestamp", "p_RS_R_x [m]", "p_RS_R_y [m]",
        "p_RS_R_z [m]", "q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []",
        "v_RS_R_x [m s^-1]", "v_RS_R_y [m s^-1]", "v_RS_R_z [m s^-1]"
        // "b_w_RS_S_x [rad s^-1]", "b_w_RS_S_y [rad s^-1]", "b_w_RS_S_z [rad
        // s^-1]", "b_a_RS_S_x [m s^-2]", "b_a_RS_S_y [m s^-2]", "b_a_RS_S_z [m
        // s^-2]"
    );
}

void MAVDataloader::loadImageData() {
    auto imagePath = mCamera0Path / "data";
    std::vector<std::filesystem::path> filesInDirectory;
    auto imageFilepaths = std::filesystem::directory_iterator(imagePath);
    std::copy(imageFilepaths, std::filesystem::directory_iterator(), std::back_inserter(filesInDirectory));
    std::sort(filesInDirectory.begin(), filesInDirectory.end());

    for (auto &filepath: filesInDirectory) {
        const std::string imageFile = filepath.string();
        utils::TTimestamp timestamp =
                stringNanoToTimePoint(filepath.filename().stem());

        cv::Mat image = cv::imread(imageFile, cv::IMREAD_GRAYSCALE);
        auto imageData = std::make_shared<ImageData>(
            timestamp, image, mCamera0Intrinsics, mCamera0CF);
        mImageQueue.push_back(imageData);
    }
}

void MAVDataloader::loadIMUData() {
    std::uint64_t ts, gt_ts;
    float ax, ay, az, wx, wy, wz;
    float gt_px, gt_py, gt_pz, gt_qw, gt_qx, gt_qy, gt_qz, gt_vx, gt_vy, gt_vz;
    while (mIMUCSVReader->read_row(ts, wx, wy, wz, ax, ay, az)) {
        Eigen::Vector3f acceleration(ax, ay, az);
        Eigen::Vector3f angularVelocity(wx, wy, wz);

        mGTCSVReader->read_row(gt_ts, gt_px, gt_py, gt_pz, gt_qw, gt_qx, gt_qy,
                               gt_qz, gt_vx, gt_vy, gt_vz);
        Eigen::Vector3f gtPos(gt_px, gt_py, gt_pz);
        Eigen::Vector3f gtVel(gt_vx, gt_vy, gt_vz);
        Eigen::Quaternionf gtQuat(gt_qw, gt_qx, gt_qy, gt_qz);

        auto data = std::make_shared<std::pair<IMUData, GTData> >(std::pair(
            IMUData(longNanoToTimePoint(ts), acceleration, angularVelocity, mIMUCF,
                    mGyroscopeNoiseDensity, mGyroscopeRandomWalk,
                    mAccelerometerNoiseDensity, mAccelerometerRandomWalk),
            GTData(longNanoToTimePoint(gt_ts), gtPos, gtQuat.toRotationMatrix(),
                   gtVel, "world")));

        mIMUQueue.push_back(data);
    }
}

std::shared_ptr<ImageData> MAVDataloader::getNextImageData() {
    return mImageQueue.pull_front();
}

std::shared_ptr<std::pair<IMUData, GTData> > MAVDataloader::getNextIMUData() {
    return mIMUQueue.pull_front();
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
