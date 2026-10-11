#include "data/mav_dataloader.hpp"
#include "plotting/plotter.hpp"
#include "schlam/Ransac.h"
#include "tft/rigid_transform_3d.hpp"

#include <chrono>
#include <cmath>
#include <thread>

#include "schlam/ORBFeatureDetector.h"
#include "schlam/Matcher.h"
#include "schlam/Optimizer.h"
#include "schlam/Frame.h"
#include "schlam/Map.h"
#include "schlam/MotionModel.h"


#define optimizePoints false

bool getNextData(std::shared_ptr<tft::Transformer> pTransformer,
                 std::shared_ptr<MAVDataloader> dataloader,
                 std::shared_ptr<Plotter> plotter,
                 const std::shared_ptr<tft::RigidTransform3D> cameraToIMU,
                 std::shared_ptr<std::pair<IMUData, GTData>> &imuAndGT,
                 std::shared_ptr<ImageData> &imageData,
                 std::shared_ptr<tft::RigidTransform3D> &cameraToWorld,
                 bool &continues) {
    continues = false;
    imageData = dataloader->getNextImageData();
    plotter->updateTimestamp(imageData->mTimestamp);
    // The GT stream starts later than the images and is not timestamp-aligned
    // with the IMU rows returned alongside it.
    if (imageData->mTimestamp < imuAndGT->second.mTimestamp) {
        continues = true;
        return false;
    }
    while (imuAndGT->second.mTimestamp < imageData->mTimestamp) {
        auto nextIMUAndGT = dataloader->getNextIMUData();
        if (nextIMUAndGT->second.mTimestamp <= imuAndGT->second.mTimestamp) {
            break; // No more GT samples are available.
        }
        imuAndGT = std::move(nextIMUAndGT);
    }
    if (imuAndGT->second.mTimestamp < imageData->mTimestamp) {
        return true;
    }
    const auto& gtData = imuAndGT->second;
    const auto imuToWorld = std::make_shared<tft::RigidTransform3D>(
        "imu", "world", gtData.mRotation, gtData.mPosition);
    cameraToWorld = imuToWorld * cameraToIMU;
    pTransformer->registerTransform(imuToWorld);
    // Register the composed pose directly so cam0s cached world pose is refreshed.
    pTransformer->registerTransform(cameraToWorld);

    pTransformer->findTransform("imu", "world");
    pTransformer->findTransform("cam0", "world");
    return false;
}

bool scaleToBaseline(std::shared_ptr<tft::RigidTransform3D> referenceGTCameraPose,
                     std::shared_ptr<tft::RigidTransform3D> cameraToWorld,
                     Eigen::Vector3d& t_opt,
                     std::vector<Eigen::Matrix<double, 3, 1>>& pts_opt) {
    const double gtBaseline = (cameraToWorld->mTranslation -
                               referenceGTCameraPose->mTranslation).norm();
    const double estimatedBaseline = t_opt.norm();
    if (!std::isfinite(gtBaseline) || gtBaseline <= 0 ||
        !std::isfinite(estimatedBaseline) || estimatedBaseline <= 0) {
        return true;
    }
    const double scale = gtBaseline / estimatedBaseline;
    t_opt *= scale;
    for (auto& point : pts_opt) {
        point *= scale;
    }
    return false;
}

int main() {
    auto pTransformer = std::make_shared<tft::Transformer>();

    auto dataloader = std::make_shared<MAVDataloader>(pTransformer);

    auto plotter = std::make_shared<Plotter>(pTransformer, dataloader->getCameraIntrinsics());
    auto render_loop = std::thread([plotter] { plotter->run(); });

    auto initFeatureDetector = std::make_shared<ORBFeatureDetector>(1000, plotter, 8);
    auto featureDetector = std::make_shared<ORBFeatureDetector>(1000, plotter, 8);

    auto optimizer = std::make_shared<Optimizer>();

    auto motionModel = MotionModel();

    const auto cameraToIMU = pTransformer->findTransform("cam0", "imu");
    auto imuAndGT = dataloader->getNextIMUData();

    int i = 0;
    std::string prevFrameName;
    std::shared_ptr<tft::RigidTransform3D> referenceGTCameraPose;
    std::shared_ptr<tft::RigidTransform3D> referenceCameraToWorld;

    std::shared_ptr<Map> map;

    std::unique_ptr<Frame> referenceFrame;
    const auto selectReference = [&](std::unique_ptr<Frame> frame,
                                     const std::shared_ptr<tft::RigidTransform3D>& cameraToWorld) {
        prevFrameName = "cam_ref";
        referenceCameraToWorld = std::make_shared<tft::RigidTransform3D>(
            prevFrameName, "world", cameraToWorld->mRotation,
            cameraToWorld->mTranslation);
        referenceGTCameraPose = cameraToWorld;
        referenceFrame = std::move(frame);
        plotter->resetCameraPaths(referenceGTCameraPose->mTranslation);
    };

    // initialization
    while (!dataloader->empty()) {
        auto now = std::chrono::system_clock::now();

        std::shared_ptr<ImageData> imageData;
        std::shared_ptr<tft::RigidTransform3D> cameraToWorld;
        bool continues;
        if (getNextData(pTransformer, dataloader, plotter, cameraToIMU, imuAndGT, imageData, cameraToWorld, continues)) break;
        if (continues) continue;

        plotter->updateFrustum(imageData);

        auto frame = std::make_unique<Frame>(imageData);

        initFeatureDetector->detectFeatures(*frame);

        if (!referenceFrame) {
            selectReference(std::move(frame), cameraToWorld);
            continue;
        }

        plotter->addGroundTruthCameraPosition(cameraToWorld->mTranslation);

        auto oldFrame = referenceFrame.get();
        auto oldFeatures = oldFrame->mKeypointTree->getFeatures();
        auto newFeatures = frame->mKeypointTree->getFeatures();
        auto [matches, _] = matchWindow(oldFrame->mKeypointTree,
                                     frame->mKeypointTree, 20, 100);
        plotter->plotMatches(oldFrame->mImage, frame->mImage, matches);

        if (matches[0].size() < 100) {
            selectReference(std::move(frame), cameraToWorld);
            std::cout << "Selecting new reference" << std::endl;
            continue;
        }

        auto reconstructOpt = reconstructInitial(matches[0], matches[1], imageData->mIntrinsics);

        if (reconstructOpt.has_value()) {
            auto [R, t, pts, inliers] = reconstructOpt.value();
            auto filtered_pts = filterByInlierMask(pts, inliers);
            auto pts2D_1 = filterByInlierMask(matches[0], inliers);
            auto pts2D_2 = filterByInlierMask(matches[1], inliers);

            auto [R_opt, t_opt, pts_opt] = optimizer->optimizePoseAndPoints({pts2D_1, pts2D_2}, R, t, filtered_pts, imageData->mIntrinsics);

            // Use the GT camera baseline to put translation and points in metres.
            if (scaleToBaseline(referenceGTCameraPose, cameraToWorld, t_opt, pts_opt)) continue;

            const std::string frameName = "cam_pred" + std::to_string(i++);
            const auto referenceToNew = std::make_shared<tft::RigidTransform3D>(
                prevFrameName, frameName, R_opt, t_opt);
            pTransformer->registerTransform(referenceCameraToWorld);
            pTransformer->findTransform(prevFrameName, "world");
            const auto newCameraToWorld = referenceCameraToWorld * referenceToNew->inverse();
            pTransformer->registerTransform(newCameraToWorld);
            pTransformer->findTransform(frameName, "world");
            plotter->addPredictedCameraPosition(newCameraToWorld->mTranslation);
            plotter->updateFrustum(imageData, newCameraToWorld);

            plotter->updatePointCloud(pts_opt, prevFrameName);

            motionModel.addPose(pTransformer->findTransform(prevFrameName, "world"), referenceFrame->mTimestamp);
            motionModel.addPose(pTransformer->findTransform(frameName, "world"), frame->mTimestamp);

            std::vector<MapPoint> mapPoints;
            for (std::size_t i = 0; i < pts_opt.size(); i++) {
                mapPoints.emplace_back(pts_opt[i], pts2D_2[i]->getDescriptor());
            }
            map = std::make_shared<Map>(mapPoints, prevFrameName);
            break; // Finished initialization
        }

        auto end = std::chrono::system_clock::now();
        std::cout << "Frame handling took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - now).count()
                <<
                " ms" << std::endl;
        while (plotter->shouldPause()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    std::cout << "Finished initialization" << std::endl;

    // tracking
    while (!dataloader->empty()) {
        std::shared_ptr<ImageData> imageData;
        std::shared_ptr<tft::RigidTransform3D> cameraToWorld;
        bool continues;
        if (getNextData(pTransformer, dataloader, plotter, cameraToIMU, imuAndGT, imageData, cameraToWorld, continues)) break;
        if (continues) continue;

        plotter->addGroundTruthCameraPosition(cameraToWorld->mTranslation);

        auto estimatedPose = motionModel.estimateLinearAt(imageData->mTimestamp);
        plotter->updateFrustum(imageData, estimatedPose);

        pTransformer->registerTransform(estimatedPose);
        pTransformer->findTransform(estimatedPose->mSource, "world");

        auto frame = std::make_shared<Frame>(imageData);
        featureDetector->detectFeatures(*frame);

        auto [matches, matched] = matchByProjection(map, frame, estimatedPose, pTransformer, 50, 20);
        std::cout << "Matched " << matches[0].size() << " out of " << map->mPoints.size() << " map points" << std::endl;
        plotter->plotMatches(frame->mImage, frame->mImage, matches);

        std::vector<Eigen::Vector3d> matchedMapPoints;
        for (std::size_t i = 0; i < map->mPoints.size(); i++) {
            if (matched[i]) {
                matchedMapPoints.emplace_back(map->mPoints[i].mPosition);
            }
        }
        const auto mapToCamera = pTransformer->findTransform(map->mFrame, estimatedPose->mSource);
        const auto [rotation, translation] = optimizer->optimizePose(matches[1],
            mapToCamera->mRotation, mapToCamera->mTranslation,
            matchedMapPoints, imageData->mIntrinsics);
        const auto optimizedMapToCamera = std::make_shared<tft::RigidTransform3D>(
            map->mFrame, estimatedPose->mSource, rotation, translation);
        const auto mapToWorld = pTransformer->findTransform(map->mFrame, "world");
        const auto optimizedCameraToWorld = mapToWorld * optimizedMapToCamera->inverse();
        pTransformer->registerTransform(optimizedCameraToWorld);
        pTransformer->findTransform(optimizedCameraToWorld->mSource, "world");
        plotter->updateFrustum(imageData, optimizedCameraToWorld);


        if (!matchedMapPoints.empty()) {
            motionModel.addPose(optimizedCameraToWorld, imageData->mTimestamp);
            plotter->addPredictedCameraPosition(optimizedCameraToWorld->mTranslation);
        }

        while (plotter->shouldPause() || matchedMapPoints.size() < 60) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    render_loop.join();
    return 0;
}
