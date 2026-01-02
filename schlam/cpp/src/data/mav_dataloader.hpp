
#pragma once
#include "image_data.hpp"
#include "imu_data.hpp"
#include "src/tft/transformer.hpp"
#include "gt_data.hpp"

#include <boost/thread/concurrent_queues/sync_bounded_queue.hpp>

#include <csv.h>

#include <filesystem>
#include <thread>

class MAVDataloader {

public:
  MAVDataloader(const std::filesystem::path &aDatasetPath,
                std::shared_ptr<tft::Transformer> aTransformer);

  ~MAVDataloader();

  std::shared_ptr<ImageData> getNextImageData();
  std::shared_ptr<std::pair<IMUData, GTData>> getNextIMUData();

  bool empty();

private:
  void loadImageData();
  void loadIMUData();

  void loadCamera0Config();
  void loadIMUConfig();

  std::thread mImageLoader;
  std::thread mIMULoader;

  boost::sync_bounded_queue<std::shared_ptr<ImageData>> mImageQueue;
  boost::sync_bounded_queue<std::shared_ptr<std::pair<IMUData, GTData>>> mIMUQueue;

  std::filesystem::path mDatasetPath;
  std::filesystem::path mCamera0Path;

  const std::string mIMUCF{"imu"};
  std::shared_ptr<io::CSVReader<7>> mIMUCSVReader;
  double mGyroscopeNoiseDensity{0.0};
  double mGyroscopeRandomWalk{0.0};
  double mAccelerometerNoiseDensity{0.0};
  double mAccelerometerRandomWalk{0.0};


  const std::string mGTCF{"gt/body"};
  const std::string mGTCamera0CF{"gt/cam0"};
  std::shared_ptr<io::CSVReader<11>> mGTCSVReader;

  const std::string mCamera0CF{"cam0"};
  Eigen::Matrix3d mCamera0Intrinsics;
  Eigen::Matrix3d mCamera0Rotation;
  Eigen::Vector3d mCamera0Translation;

  std::uint32_t mImageIndex{0};
  std::uint32_t mIMUIndex{0};
};

std::chrono::system_clock::time_point
stringNanoToTimePoint(const std::string &nano_str);
std::chrono::system_clock::time_point
longNanoToTimePoint(const std::uint64_t &aNanos);