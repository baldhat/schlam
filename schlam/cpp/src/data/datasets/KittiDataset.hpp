//
// Created by jonathan_michel on 10/24/26.
//

#ifndef SCHLAM_KITTIDATASET_H
#define SCHLAM_KITTIDATASET_H


#include "data/DatasetInterface.hpp"
#include "../image_data.hpp"
#include "../imu_data.hpp"
#include "tft/transformer.hpp"
#include "../gt_data.hpp"

#include <boost/thread/concurrent_queues/sync_bounded_queue.hpp>

#include <csv.h>

#include <filesystem>
#include <thread>

class KittiDataset : public DatasetInterface {
public:
    KittiDataset(std::shared_ptr<tft::Transformer> aTransformer, const std::string& aSequence);

    ~KittiDataset();

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

    void loadGTData();

    void loadCamera0Config();

    void loadGTConfig();

    std::thread mImageLoader;
    std::thread mGTLoader;

    boost::sync_bounded_queue<std::shared_ptr<ImageData> > mImageQueue;
    boost::sync_bounded_queue<std::shared_ptr<IMUData> > mIMUQueue;
    boost::sync_bounded_queue<std::shared_ptr<GTData> > mGTQueue;

    std::size_t mNumImagesRead{0};
    std::size_t mNumImages{0};

    std::filesystem::path mDatasetPath;
    std::filesystem::path mCamera0Path;

    std::vector<Eigen::Matrix<double, 3, 4>> mPoses;
    std::vector<utils::TTimestamp> mTimePoints;

    const std::string mGTCF{"gt/body"};
    const std::string mGTCamera0CF{"gt/cam0"};
    std::shared_ptr<io::CSVReader<11> > mGTCSVReader;

    const std::string mCamera0CF{"gt/cam0"};
    Eigen::Matrix3d mCamera0Intrinsics;
    Eigen::Matrix3d mCamera0Rotation;
    Eigen::Vector3d mCamera0Translation;

    std::atomic_bool mAllImagesRead{false};
    std::atomic_bool mAllGTRead{false};

    const std::string mSequence;
};


#endif //SCHLAM_KITTIDATASET_H
