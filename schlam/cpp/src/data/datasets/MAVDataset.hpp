
#pragma once
#include "../image_data.hpp"
#include "../imu_data.hpp"
#include "tft/transformer.hpp"
#include "../gt_data.hpp"

#include <boost/thread/concurrent_queues/sync_bounded_queue.hpp>

#include <csv.h>

#include <filesystem>
#include <thread>

#include "../DatasetInterface.hpp"

class MAVDataset : public DatasetInterface {

public:
  MAVDataset(std::shared_ptr<tft::Transformer> aTransformer, const std::string& sequence);

  ~MAVDataset();

  double getImageAspectRatio() override;

  bool hasIMU() override;
  bool hasGT() override;

  std::shared_ptr<ImageData> getNextImageData() override;
  std::shared_ptr<IMUData> getNextIMUData() override;
  std::shared_ptr<GTData> getNextGTData() override;

  bool empty() override;

  Eigen::Matrix3d getCameraIntrinsics() override;

  float percentageRead() override;

private:
  void loadImageData();
  void loadIMUData();

  void loadCamera0Config();
  void loadIMUConfig();

  std::thread mImageLoader;
  std::thread mIMULoader;

  boost::sync_bounded_queue<std::shared_ptr<ImageData>> mImageQueue;
  boost::sync_bounded_queue<std::shared_ptr<IMUData>> mIMUQueue;
  boost::sync_bounded_queue<std::shared_ptr<GTData>> mGTQueue;

  std::size_t mNumImagesRead{0};
  std::size_t mNumImages{0};

  std::filesystem::path mDatasetPath;
  std::filesystem::path mCamera0Path;
  const std::string mSequence;

  std::shared_ptr<io::CSVReader<7>> mIMUCSVReader;
  double mGyroscopeNoiseDensity{0.0};
  double mGyroscopeRandomWalk{0.0};
  double mAccelerometerNoiseDensity{0.0};
  double mAccelerometerRandomWalk{0.0};

  const std::string mGTCF{"gt/body"};
  const std::string mGTCamera0CF{"gt/cam0"};
  std::shared_ptr<io::CSVReader<11>> mGTCSVReader;

  const std::string mCamera0CF{"gt/cam0"};
  Eigen::Matrix3d mCamera0Intrinsics;
  cv::Mat mCamera0DistortionCoefficients;
  Eigen::Matrix3d mCamera0Rotation;
  Eigen::Vector3d mCamera0Translation;

  std::atomic_bool mAllImagesRead{false};
  std::atomic_bool mAllIMURead{false};
  std::atomic_bool mStop{false};
};

