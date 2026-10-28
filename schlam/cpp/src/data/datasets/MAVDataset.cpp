
#include "MAVDataset.hpp"
#include "tft/rigid_transform_3d.hpp"
#include "data/data_helper.hpp"

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



MAVDataset::MAVDataset(std::shared_ptr<tft::Transformer> aTransformer, const std::string& sequence)
    : mSequence(sequence), mImageQueue(3), mIMUQueue(2000), mGTQueue(2000) {
    const char *raw_val = std::getenv("MAV_PATH");
    mDatasetPath = std::filesystem::path((raw_val != nullptr) ? raw_val : "") / mSequence / "mav0";

    loadCamera0Config();
    aTransformer->registerTransform(std::make_shared<tft::RigidTransform3D>(
        mCamera0CF, mGTCF, mCamera0Rotation, mCamera0Translation));

    loadIMUConfig();
    mImageLoader = std::thread(&MAVDataset::loadImageData, this);
    mIMULoader = std::thread(&MAVDataset::loadIMUData, this);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

MAVDataset::~MAVDataset() {
    mStop = true;
    if (mImageLoader.joinable())
        mImageLoader.join();
    if (mIMULoader.joinable())
        mIMULoader.join();
}

double MAVDataset::getImageAspectRatio() {
    return 1920 / 1080.0;
}

bool MAVDataset::hasIMU() {
    return true;
}

bool MAVDataset::hasGT() {
    return true;
}

bool MAVDataset::empty() {
    return (mImageQueue.empty() && mAllImagesRead) || (mIMUQueue.empty() && mAllIMURead) ||
               (mGTQueue.empty() && mAllIMURead);
}

float MAVDataset::percentageRead() {
    return static_cast<float>(100.0 * mNumImagesRead / mNumImages);
}

void MAVDataset::loadCamera0Config() {
    mCamera0Path = mDatasetPath / "cam0";
    YAML::Node camera0Config = YAML::LoadFile(mCamera0Path / "sensor.yaml");
    double fu = camera0Config["intrinsics"][0].as<double>(0.0);
    double fv = camera0Config["intrinsics"][1].as<double>(0.0);
    double cu = camera0Config["intrinsics"][2].as<double>(0.0);
    double cv = camera0Config["intrinsics"][3].as<double>(0.0);
    mCamera0Intrinsics << fu, 0, cu, 0, fv, cv, 0, 0, 1;

    const auto coefficients = camera0Config["distortion_coefficients"].as<std::vector<double> >();
    const auto distortionModel = camera0Config["distortion_model"].as<std::string>("rad-tan");
    const auto cameraModel = camera0Config["camera_model"].as<std::string>("pinhole");
    mCamera0DistortionCoefficients = cv::Mat(coefficients).clone();

    auto ec = camera0Config["T_BS"]["data"].as<std::vector<double> >();
    mCamera0Rotation << ec[0], ec[1], ec[2], ec[4], ec[5], ec[6], ec[8], ec[9],
            ec[10];
    mCamera0Translation = Eigen::Vector3d(ec[3], ec[7], ec[11]);
}

Eigen::Matrix3d MAVDataset::getCameraIntrinsics() {
    return mCamera0Intrinsics;
}


void MAVDataset::loadIMUConfig() {
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

void MAVDataset::loadImageData() {
    auto imagePath = mCamera0Path / "data";
    std::vector<std::filesystem::path> filesInDirectory;
    auto imageFilepaths = std::filesystem::directory_iterator(imagePath);
    std::copy(imageFilepaths, std::filesystem::directory_iterator(), std::back_inserter(filesInDirectory));
    std::sort(filesInDirectory.begin(), filesInDirectory.end());

    mNumImages = filesInDirectory.size();

    cv::Mat cameraMatrix;
    cv::eigen2cv(mCamera0Intrinsics, cameraMatrix);
    cv::Mat map1, map2;
    cv::Size mapSize;

    for (auto &filepath: filesInDirectory) {
        const std::string imageFile = filepath.string();
        utils::TTimestamp timestamp =
                helpers::stringNanoToTimePoint(filepath.filename().stem());

        cv::Mat image = cv::imread(imageFile, cv::IMREAD_GRAYSCALE);
        if (!image.empty() && !mCamera0DistortionCoefficients.empty()) {
            if (map1.empty() || mapSize != image.size()) {
                mapSize = image.size();
                // Keep the original intrinsics for the corrected pinhole image.
                cv::initUndistortRectifyMap(cameraMatrix, mCamera0DistortionCoefficients,
                                            cv::Mat(), cameraMatrix, mapSize, CV_16SC2, map1, map2);
            }
            cv::Mat correctedImage;
            cv::remap(image, correctedImage, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            image = std::move(correctedImage);
        }
        auto imageData = std::make_shared<ImageData>(
            timestamp, image, mCamera0Intrinsics, mCamera0CF);
        while (!mStop && mImageQueue.try_push_back(imageData) != boost::queue_op_status::success);
    }
    mAllImagesRead = true;
}

void MAVDataset::loadIMUData() {
    std::uint64_t ts, gt_ts;
    double ax, ay, az, wx, wy, wz;
    double gt_px, gt_py, gt_pz, gt_qw, gt_qx, gt_qy, gt_qz, gt_vx, gt_vy, gt_vz;
    while (mIMUCSVReader->read_row(ts, wx, wy, wz, ax, ay, az)) {
        Eigen::Vector3d acceleration(ax, ay, az);
        Eigen::Vector3d angularVelocity(wx, wy, wz);

        mGTCSVReader->read_row(gt_ts, gt_px, gt_py, gt_pz, gt_qw, gt_qx, gt_qy,
                               gt_qz, gt_vx, gt_vy, gt_vz);
        Eigen::Vector3d gtPos(gt_px, gt_py, gt_pz);
        Eigen::Vector3d gtVel(gt_vx, gt_vy, gt_vz);
        Eigen::Quaterniond gtQuat(gt_qw, gt_qx, gt_qy, gt_qz);

        auto imuData = std::make_shared<IMUData>(
            helpers::longNanoToTimePoint(ts), acceleration, angularVelocity, mGTCF,
            mGyroscopeNoiseDensity, mGyroscopeRandomWalk,
            mAccelerometerNoiseDensity, mAccelerometerRandomWalk);
        auto gtData = std::make_shared<GTData>(helpers::longNanoToTimePoint(gt_ts), gtPos, gtQuat.toRotationMatrix(),
                                               gtVel, "world");

        while (!mStop && mIMUQueue.try_push_back(imuData) != boost::queue_op_status::success);
        while (!mStop && mGTQueue.try_push_back(gtData) != boost::queue_op_status::success);
    }
    mAllIMURead = true;
}

std::shared_ptr<ImageData> MAVDataset::getNextImageData() {
    auto imageData = mImageQueue.pull_front();
    mNumImagesRead++;
    return imageData;
}

std::shared_ptr<IMUData> MAVDataset::getNextIMUData() {
    return mIMUQueue.pull_front();
}

std::shared_ptr<GTData> MAVDataset::getNextGTData() {
    return mGTQueue.pull_front();
}

