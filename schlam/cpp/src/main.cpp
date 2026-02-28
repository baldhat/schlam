#include "data/mav_dataloader.hpp"
#include "plotting/plotter.hpp"
#include "schlam/Ransac.h"
#include "tft/rigid_transform_3d.hpp"

#include <chrono>
#include <thread>

#include "schlam/ORBFeatureDetector.h"
#include "schlam/Matcher.h"
#include "schlam/Optimizer.h"

int main() {
    auto pTransformer = std::make_shared<tft::Transformer>();

    auto plotter = std::make_shared<Plotter>(pTransformer);
    auto render_loop = std::thread([plotter] { plotter->run(); });

    auto dataloader = std::make_shared<MAVDataloader>(pTransformer);

    auto featureDetector = std::make_shared<ORBFeatureDetector>(500, plotter, 8);

    auto optimizer = std::make_shared<Optimizer>();

    auto oldImageData = dataloader->getNextImageData();
    auto oldFeatures = featureDetector->getFeatures(oldImageData->mImage);

    while (!dataloader->empty()) {
        auto now = std::chrono::system_clock::now();

        auto newImageData = dataloader->getNextImageData();
        auto newIMUData = dataloader->getNextIMUData();
        std::shared_ptr<IMUData> imuData = std::make_shared<IMUData>(newIMUData->first);
        std::shared_ptr<GTData> gtData = std::make_shared<GTData>(newIMUData->second);
        std::uint32_t cnt = 0;
        while (imuData->mTimestamp < newImageData->mTimestamp) {
            newIMUData = dataloader->getNextIMUData();
            imuData = std::make_shared<IMUData>(newIMUData->first);
            gtData = std::make_shared<GTData>(newIMUData->second);
            cnt++;
        }

        pTransformer->registerTransform(std::make_shared<tft::RigidTransform3D>(
            "imu", "world", gtData->mRotation,
            gtData->mPosition));

        pTransformer->findTransform("imu", "world");
        pTransformer->findTransform("cam0", "world");

        plotter->updateFrustum(newImageData);

        auto newFeatures = featureDetector->getFeatures(newImageData->mImage);

        auto matches = match(oldFeatures, newFeatures, 20);
        plotter->plotMatches(oldImageData->mImage, newImageData->mImage, oldFeatures, newFeatures, matches);
        auto [matchedOldFeatures, matchedNewFeatures] = getMatched(oldFeatures, newFeatures, matches);


        auto reconstructOpt = reconstructInitial(matchedOldFeatures, matchedNewFeatures, newImageData->mIntrinsics);

        if (reconstructOpt.has_value()) {
            auto [R, t, pts, inliers] = reconstructOpt.value();
            auto filtered_pts = filterByInlierMask(pts, inliers);
            auto pts2D_1 = filterByInlierMask(matchedOldFeatures, inliers);
            auto pts2D_2 = filterByInlierMask(matchedNewFeatures, inliers);
            if (filtered_pts.size() > 0) {
                plotter->updatePointCloud(filtered_pts, "cam0");
            }

            auto updated = optimizer->optimize({pts2D_1, pts2D_2}, R, t, filtered_pts, newImageData->mIntrinsics);
        }

        if (reconstructOpt.has_value()) {
            oldFeatures = newFeatures;
            oldImageData = newImageData;
        }

        auto end = std::chrono::system_clock::now();
        std::cout << "Frame handling took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - now).count()
                <<
                " ms" << std::endl;
    }

    render_loop.join();
    return 0;
}
