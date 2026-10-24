//
// Created by jonathan_michel on 10/23/26.
//

#ifndef SCHLAM_DATASETINTERFACE_H
#define SCHLAM_DATASETINTERFACE_H

#include "image_data.hpp"
#include "imu_data.hpp"
#include "gt_data.hpp"

#include <memory>

#include <Eigen/Core>

class DatasetInterface {
public:
    virtual double getImageAspectRatio() = 0;

    virtual bool hasIMU() = 0;
    virtual bool hasGT() = 0;

    virtual std::shared_ptr<ImageData> getNextImageData() = 0;
    virtual std::shared_ptr<IMUData> getNextIMUData() = 0;
    virtual std::shared_ptr<GTData> getNextGTData() = 0;

    virtual bool empty() = 0;
    virtual Eigen::Matrix3d getCameraIntrinsics() = 0;
};

#endif //SCHLAM_DATASETINTERFACE_H
