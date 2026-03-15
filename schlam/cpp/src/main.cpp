#include "data/mav_dataloader.hpp"
#include "plotting/plotter.hpp"
#include "schlam/Ransac.h"
#include "tft/rigid_transform_3d.hpp"

#include <chrono>
#include <thread>

#include "schlam/ORBFeatureDetector.h"
#include "schlam/Matcher.h"
#include "schlam/Optimizer.h"
#include "schlam/Frame.h"

int main() {
    auto pTransformer = std::make_shared<tft::Transformer>();

    auto plotter = std::make_shared<Plotter>(pTransformer);
    auto render_loop = std::thread([plotter] { plotter->run(); });

    auto dataloader = std::make_shared<MAVDataloader>(pTransformer);

    auto featureDetector = std::make_shared<ORBFeatureDetector>(1000, plotter, 1);

    auto optimizer = std::make_shared<Optimizer>();

    auto imageData = dataloader->getNextImageData();
    auto prevFrame = std::make_unique<Frame>(imageData);
    featureDetector->detectFeatures(*prevFrame);

    bool initialized = false;
    int i = 0;
    std::string prevFrameName;

    std::vector<std::unique_ptr<Frame> > frames;
    frames.emplace_back(std::move(prevFrame));

    while (!dataloader->empty()) {
        auto now = std::chrono::system_clock::now();

        imageData = dataloader->getNextImageData();
        auto newIMUData = dataloader->getNextIMUData();
        std::shared_ptr<IMUData> imuData = std::make_shared<IMUData>(newIMUData->first);
        std::shared_ptr<GTData> gtData = std::make_shared<GTData>(newIMUData->second);
        std::uint32_t cnt = 0;
        while (imuData->mTimestamp < imageData->mTimestamp) {
            newIMUData = dataloader->getNextIMUData();
            imuData = std::make_shared<IMUData>(newIMUData->first);
            gtData = std::make_shared<GTData>(newIMUData->second);
            cnt++;
        }

        if (!initialized) {
            pTransformer->registerTransform(std::make_shared<tft::RigidTransform3D>(
                "imu_pred0", "world", gtData->mRotation,
                gtData->mPosition));
            initialized = true;
            prevFrameName = "imu_pred0";
        }

        pTransformer->registerTransform(std::make_shared<tft::RigidTransform3D>(
            "imu", "world", gtData->mRotation,
            gtData->mPosition));

        pTransformer->findTransform("imu", "world");
        pTransformer->findTransform("cam0", "world");

        plotter->updateFrustum(imageData);

        auto frame = std::make_unique<Frame>(imageData);

        featureDetector->detectFeatures(*frame);

        auto oldFrame = frames.back().get();
        auto oldFeatures = oldFrame->mKeypointTree->getFeatures();
        auto newFeatures = frame->mKeypointTree->getFeatures();
        // auto matches = matchWindow(oldFrame->mKeypointTree.get(),
        //                              frame->mKeypointTree.get(), 20, 100);
        auto matches = matchWindow(oldFrame->mKeypointTree.get(),
                                     frame->mKeypointTree.get(), 20, 100);
        //plotter->plotMatches(*oldFrame, *frame, matches);

        auto reconstructOpt = reconstructInitial(matches[0], matches[1], imageData->mIntrinsics);

        if (reconstructOpt.has_value()) {
            auto [R, t, pts, inliers] = reconstructOpt.value();
            auto filtered_pts = filterByInlierMask(pts, inliers);
            auto pts2D_1 = filterByInlierMask(matches[0], inliers);
            auto pts2D_2 = filterByInlierMask(matches[1], inliers);

            auto updated = optimizer->optimize({pts2D_1, pts2D_2}, R, t, filtered_pts, imageData->mIntrinsics);

            plotter->updatePointCloud(std::get<2>(updated), "cam0");

            std::string frameName = "imu_pred" + std::to_string(++i);
            pTransformer->registerTransform(std::make_shared<tft::RigidTransform3D>(
                frameName, prevFrameName, std::get<0>(updated),
                std::get<1>(updated)));
            pTransformer->findTransform(frameName, "world");
            prevFrameName = frameName;

            frames.push_back(std::move(frame));
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
