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


#define optimizePoints false

int main() {
    auto pTransformer = std::make_shared<tft::Transformer>();

    auto plotter = std::make_shared<Plotter>(pTransformer);
    auto render_loop = std::thread([plotter] { plotter->run(); });

    auto dataloader = std::make_shared<MAVDataloader>(pTransformer);

    auto featureDetector = std::make_shared<ORBFeatureDetector>(1000, plotter, 1);

    auto optimizer = std::make_shared<Optimizer>();

    const auto cameraToIMU = pTransformer->findTransform("cam0", "imu");
    auto imuAndGT = dataloader->getNextIMUData();

    int i = 0;
    std::string prevFrameName;
    std::shared_ptr<tft::RigidTransform3D> referenceGTCameraPose;
    std::shared_ptr<tft::RigidTransform3D> referenceCameraToWorld;

    std::unique_ptr<Frame> referenceFrame;
    const auto selectReference = [&](std::unique_ptr<Frame> frame,
                                     const std::shared_ptr<tft::RigidTransform3D>& cameraToWorld) {
        prevFrameName = "cam_ref";
        referenceCameraToWorld = std::make_shared<tft::RigidTransform3D>(
            prevFrameName, "world", cameraToWorld->mRotation,
            cameraToWorld->mTranslation);
        referenceGTCameraPose = cameraToWorld;
        referenceFrame = std::move(frame);
    };

    while (!dataloader->empty()) {
        auto now = std::chrono::system_clock::now();

        auto imageData = dataloader->getNextImageData();
        plotter->updateTimestamp(imageData->mTimestamp);
        // The GT stream starts later than the images and is not timestamp-aligned
        // with the IMU rows returned alongside it.
        if (imageData->mTimestamp < imuAndGT->second.mTimestamp) {
            continue;
        }
        while (imuAndGT->second.mTimestamp < imageData->mTimestamp) {
            auto nextIMUAndGT = dataloader->getNextIMUData();
            if (nextIMUAndGT->second.mTimestamp <= imuAndGT->second.mTimestamp) {
                break; // No more GT samples are available.
            }
            imuAndGT = std::move(nextIMUAndGT);
        }
        if (imuAndGT->second.mTimestamp < imageData->mTimestamp) {
            break;
        }
        const auto& gtData = imuAndGT->second;
        const auto imuToWorld = std::make_shared<tft::RigidTransform3D>(
            "imu", "world", gtData.mRotation, gtData.mPosition);
        const auto cameraToWorld = imuToWorld * cameraToIMU;
        pTransformer->registerTransform(imuToWorld);
        // Register the composed pose directly so cam0s cached world pose is refreshed.
        pTransformer->registerTransform(cameraToWorld);

        pTransformer->findTransform("imu", "world");
        pTransformer->findTransform("cam0", "world");

        plotter->updateFrustum(imageData);

        auto frame = std::make_unique<Frame>(imageData);

        featureDetector->detectFeatures(*frame);

        if (!referenceFrame) {
            selectReference(std::move(frame), cameraToWorld);
            continue;
        }

        auto oldFrame = referenceFrame.get();
        auto oldFeatures = oldFrame->mKeypointTree->getFeatures();
        auto newFeatures = frame->mKeypointTree->getFeatures();
        auto matches = matchWindow(oldFrame->mKeypointTree.get(),
                                     frame->mKeypointTree.get(), 20, 100);
        plotter->plotMatches(*oldFrame, *frame, matches);

        if (matches[0].size() < 100) {
            // Replace the candidate; only an accepted reference is registered for plotting.
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

            auto updated = optimizer->optimize({pts2D_1, pts2D_2}, R, t, filtered_pts, imageData->mIntrinsics);

            // Use the GT camera baseline to put translation and points in metres.
            const double gtBaseline = (cameraToWorld->mTranslation -
                                      referenceGTCameraPose->mTranslation).norm();
            const double estimatedBaseline = std::get<1>(updated).norm();
            if (!std::isfinite(gtBaseline) || gtBaseline <= 0 ||
                !std::isfinite(estimatedBaseline) || estimatedBaseline <= 0) {
                continue;
            }
            const double scale = gtBaseline / estimatedBaseline;
            std::get<1>(updated) *= scale;
            for (auto& point : std::get<2>(updated)) {
                point *= scale;
            }

            const std::string frameName = "cam_pred" + std::to_string(i++);
            const auto referenceToNew = std::make_shared<tft::RigidTransform3D>(
                prevFrameName, frameName, std::get<0>(updated), std::get<1>(updated));
            pTransformer->registerTransform(referenceCameraToWorld);
            pTransformer->findTransform(prevFrameName, "world");
            const auto newCameraToWorld = referenceCameraToWorld * referenceToNew->inverse();
            pTransformer->registerTransform(newCameraToWorld);
            pTransformer->findTransform(frameName, "world");

            plotter->updatePointCloud(std::get<2>(updated), prevFrameName);

            prevFrameName = frameName;
            referenceCameraToWorld = newCameraToWorld;
            referenceGTCameraPose = cameraToWorld;
            referenceFrame = std::move(frame);
            while (true) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        auto end = std::chrono::system_clock::now();
        std::cout << "Frame handling took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - now).count()
                <<
                " ms" << std::endl;
        while (plotter->shouldPause()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    render_loop.join();
    return 0;
}
