//
// Created by jonathan_michel on 10/23/26.
//

#ifndef SCHLAM_DATASETINTERFACE_H
#define SCHLAM_DATASETINTERFACE_H

class DatasetInterface {

    virtual std::shared_ptr<ImageData> getNextImageData() = 0;
    virtual std::shared_ptr<std::pair<IMUData, GTData>> getNextIMUData() = 0;
    virtual bool empty() = 0;
    virtual Eigen::Matrix3d getCameraIntrinsics() = 0;
};

#endif //SCHLAM_DATASETINTERFACE_H
