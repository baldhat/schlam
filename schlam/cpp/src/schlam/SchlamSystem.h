//
// Created by jonathan_michel on 10/25/26.
//

#ifndef SCHLAM_SCHLAMSYSTEM_H
#define SCHLAM_SCHLAMSYSTEM_H
#include <memory>
#include <thread>
#include <boost/thread/concurrent_queues/sync_bounded_queue.hpp>

#include "MotionModel.h"
#include "Optimizer.h"
#include "tft/transformer.hpp"


struct MapPointMatch;
class IMUData;
class GTData;
class ORBFeatureDetector;
class IPlotter;
class DatasetInterface;

struct SynchronizedData {
    std::shared_ptr<ImageData> imageData;
    std::shared_ptr<IMUData> imuData;
    std::shared_ptr<GTData> gtData;
    std::shared_ptr<tft::RigidTransform3D> cameraToWorld;
};

enum class TrackingState {
    INITIALIZING,
    TRACKING,
    LOST
};

class SchlamSystem {
public:
    SchlamSystem(
        std::shared_ptr<DatasetInterface> dataset,
        std::shared_ptr<tft::Transformer> transformer,
        std::shared_ptr<IPlotter> plotter,
        std::shared_ptr<ORBFeatureDetector> featureDetector,
        std::shared_ptr<Optimizer> optimizer,
        MotionModel& motionModel);

    void run();
private:
    std::shared_ptr<DatasetInterface> dataset_;
    std::shared_ptr<tft::Transformer> transformer_;
    std::shared_ptr<IPlotter> plotter_;
    std::shared_ptr<ORBFeatureDetector> featureDetector_;
    std::shared_ptr<Optimizer> optimizer_;
    MotionModel& motionModel_;

    std::unique_ptr<Map> map_;
    std::shared_ptr<tft::RigidTransform3D> cameraToIMU_;
    std::uint32_t framesSinceLastKeyframe_{0};
    TrackingState state_{TrackingState::INITIALIZING};
    std::thread localMappingThread_;
    std::atomic_bool localMappingIdle_{false};
    std::atomic_bool stopLocalMapping_{false};
    boost::sync_bounded_queue<KeyFrameId> localMappingQueue_;

    // Reference frame
    std::shared_ptr<tft::RigidTransform3D> referenceCameraToWorld_;
    std::shared_ptr<tft::RigidTransform3D> referenceGTCameraPose_;
    std::shared_ptr<Frame> referenceFrame_;
    const char *referenceFrameName_ = "cam_ref";

    SynchronizedData getNextSynchronizedData();

    bool scaleToBaseline(
        std::shared_ptr<tft::RigidTransform3D> referenceGTCameraPose,
        std::shared_ptr<tft::RigidTransform3D> cameraToWorld,
        Eigen::Vector3d& t_opt,
        std::vector<Eigen::Matrix<double, 3, 1>>& pts_opt
    );

    void buildMap(
        std::string prevFrameName,
        std::shared_ptr<Frame> referenceFrame,
        std::shared_ptr<Frame> frame,
        std::vector<std::shared_ptr<KeyPoint>> pts2D_1,
        std::vector<std::shared_ptr<KeyPoint>> pts2D_2,
        std::vector<Eigen::Vector3d> pts_opt
    );

    void triangulateUnmatchedFeatures();

    const KeyFrameId addKeyFrame(std::shared_ptr<Frame> frame, std::vector<MapPointMatch> matches,
                                 std::vector<bool> inliers);

    void initialize(const SynchronizedData& data);
    void track(const SynchronizedData& data);
    void localMapping();

    void setFrameAsReference(
        std::shared_ptr<Frame> frame,
        const std::shared_ptr<tft::RigidTransform3D>& cameraToWorld);
};

#endif //SCHLAM_SCHLAMSYSTEM_H
