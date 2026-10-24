#include <gtest/gtest.h>

#include "data/mav_dataloader.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>

class MAVDatasetTest : public ::testing::Test {
protected:
    std::filesystem::path dataset;
    std::optional<std::string> originalDatasetPath;

    void SetUp() override {
        if (const char* value = std::getenv("MAV_PATH")) originalDatasetPath = value;
        dataset = std::filesystem::temp_directory_path() /
            ("schlam_loader_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(dataset / "cam0/data");
        std::filesystem::create_directories(dataset / "imu0");
        std::filesystem::create_directories(dataset / "state_groundtruth_estimate0");
        setenv("MAV_PATH", dataset.c_str(), 1);
        std::ofstream(dataset / "imu0/sensor.yaml") <<
            "gyroscope_noise_density: 0.001\ngyroscope_random_walk: 0.001\n"
            "accelerometer_noise_density: 0.001\naccelerometer_random_walk: 0.001\n";
        std::ofstream(dataset / "imu0/data.csv") <<
            "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
            "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
            "1000000000,0,0,0,0,0,0\n";
        std::ofstream(dataset / "state_groundtruth_estimate0/data.csv") <<
            "#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],"
            "q_RS_y [],q_RS_z [],v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1]\n"
            "1000000000,0,0,0,1,0,0,0,0,0,0\n";
    }

    void TearDown() override {
        if (originalDatasetPath) setenv("MAV_PATH", originalDatasetPath->c_str(), 1);
        else unsetenv("MAV_PATH");
        std::filesystem::remove_all(dataset);
    }

    void calibration(const std::string& distortionYaml) {
        std::ofstream(dataset / "cam0/sensor.yaml") <<
            "camera_model: pinhole\nintrinsics: [40, 45, 31.5, 31.5]\n" << distortionYaml <<
            "T_BS:\n  data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]\n";
    }

    cv::Mat image(const std::string& timestamp, cv::Size size = {64, 64}) {
        cv::Mat pixels(size, CV_8UC1);
        for (int y = 0; y < size.height; ++y) {
            for (int x = 0; x < size.width; ++x) pixels.at<uint8_t>(y, x) = 3 * x;
        }
        std::ofstream file(dataset / "cam0/data" / (timestamp + ".pgm"), std::ios::binary);
        file << "P5\n" << size.width << ' ' << size.height << "\n255\n";
        file.write(reinterpret_cast<const char*>(pixels.data), pixels.total());
        return pixels;
    }

    void expectIntrinsics(const ImageData& frame) {
        Eigen::Matrix3d expected;
        expected << 40, 0, 31.5, 0, 45, 31.5, 0, 0, 1;
        EXPECT_TRUE(frame.mIntrinsics.isApprox(expected));
        EXPECT_EQ(frame.mCoordinateFrame, "cam0");
    }
};

TEST_F(MAVDatasetTest, RadTanCorrectionUsesCalibratedRaysAndPreservesMetadata) {
    calibration("distortion_model: rad-tan\ndistortion_coefficients: [-0.25, 0.05, 0.01, -0.015]\n");
    image("1000000000");
    image("2000000000");
    MAVDataset loader(std::make_shared<tft::Transformer>());
    for (int frameIndex = 0; frameIndex < 2; ++frameIndex) {
        const auto frame = loader.getNextImageData();
        ASSERT_EQ(frame->mImage.size(), cv::Size(64, 64));
        expectIntrinsics(*frame);
        EXPECT_EQ(frame->mTimestamp, longNanoToTimePoint((frameIndex + 1) * 1000000000ULL));
        // The input ramp has intensity 3*x. Compute its source coordinate from
        // the calibrated radial/tangential projection, independent of remapping.
        const double x = (10 - 31.5) / 40;
        const double y = (16 - 31.5) / 45;
        const double r2 = x * x + y * y;
        const double radial = 1 - 0.25 * r2 + 0.05 * r2 * r2;
        const double distortedX = x * radial + 2 * 0.01 * x * y - 0.015 * (r2 + 2 * x * x);
        const double expectedIntensity = 3 * (40 * distortedX + 31.5);
        EXPECT_NEAR(frame->mImage.at<uint8_t>(16, 10), expectedIntensity, 1);
        EXPECT_NE(frame->mImage.at<uint8_t>(16, 10), 30);
    }
}

TEST_F(MAVDatasetTest, ZeroRadTanDistortionPreservesImage) {
    calibration("distortion_model: rad-tan\ndistortion_coefficients: [0, 0, 0, 0]\n");
    const auto original = image("1000000000");
    MAVDataset loader(std::make_shared<tft::Transformer>());
    const auto frame = loader.getNextImageData();
    EXPECT_EQ(cv::norm(original, frame->mImage, cv::NORM_INF), 0);
    expectIntrinsics(*frame);
}

TEST_F(MAVDatasetTest, EuRoCDistortionModelNameIsSupported) {
    calibration("distortion_model: radial-tangential\ndistortion_coefficients: [-0.25, 0.05, 0.01, -0.015]\n");
    image("1000000000");
    MAVDataset loader(std::make_shared<tft::Transformer>());
    const auto frame = loader.getNextImageData();
    EXPECT_NE(frame->mImage.at<uint8_t>(16, 10), 30);
    expectIntrinsics(*frame);
}

TEST_F(MAVDatasetTest, CalibrationWithoutDistortionPreservesImage) {
    calibration("");
    const auto original = image("1000000000");
    MAVDataset loader(std::make_shared<tft::Transformer>());
    EXPECT_EQ(cv::norm(original, loader.getNextImageData()->mImage, cv::NORM_INF), 0);
}

TEST_F(MAVDatasetTest, CorrectionMapsFollowImageResolution) {
    calibration("distortion_model: rad-tan\ndistortion_coefficients: [-0.25, 0.05, 0.01, -0.015]\n");
    image("1000000000");
    image("2000000000", {72, 60});
    MAVDataset loader(std::make_shared<tft::Transformer>());
    EXPECT_EQ(loader.getNextImageData()->mImage.size(), cv::Size(64, 64));
    EXPECT_EQ(loader.getNextImageData()->mImage.size(), cv::Size(72, 60));
}

TEST_F(MAVDatasetTest, UnsupportedModelAndInvalidCoefficientsFailBeforeLoading) {
    calibration("distortion_model: unknown\ndistortion_coefficients: [0, 0, 0, 0]\n");
    EXPECT_THROW(MAVDataset(std::make_shared<tft::Transformer>()), std::runtime_error);
    calibration("distortion_model: equidistant\ndistortion_coefficients: [0, 0, 0, 0]\n");
    EXPECT_THROW(MAVDataset(std::make_shared<tft::Transformer>()), std::runtime_error);
    calibration("distortion_model: rad-tan\ndistortion_coefficients: [0, 0, 0]\n");
    EXPECT_THROW(MAVDataset(std::make_shared<tft::Transformer>()), std::runtime_error);
}
