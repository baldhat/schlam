#include "data/mav_dataloader.hpp"
#include "plotting/plotter.hpp"
#include "src/tft/rigid_transform_3d.hpp"

#include <chrono>
#include <thread>

#include "schlam/ORBFeatureDetector.h"
#include "schlam/Matcher.h"

const std::filesystem::path mavDataPath(
    "/home/baldhat/dev/slam/MAV/vicon_room1/V1_01_easy/V1_01_easy/mav0/");

int main() {
    auto pTransformer = std::make_shared<tft::Transformer>();

    auto plotter = std::make_shared<Plotter>(pTransformer);
    auto render_loop = std::thread([plotter] { plotter->run(); });

    auto dataloader = std::make_shared<MAVDataloader>(mavDataPath, pTransformer);

    auto featureDetector = std::make_shared<ORBFeatureDetector>(500, plotter, 8);

    auto oldImageData = dataloader->getNextImageData();
    auto oldFeatures = featureDetector->getFeatures(oldImageData->mImage);

    while (!dataloader->empty()) {
        std::cout << "Handling new image..." << std::endl;
        auto newImageData = dataloader->getNextImageData();
        auto newIMUData = dataloader->getNextIMUData();
        auto imuData = newIMUData->first;
        auto gtData = newIMUData->second;

        pTransformer->registerTransform(std::make_shared<tft::RigidTransform3D>(
            "imu", "world", gtData.mRotation,
            gtData.mPosition));

        pTransformer->findTransform("imu", "world");
        pTransformer->findTransform("cam0", "world");

        plotter->addFrustum(newImageData);

        auto now = std::chrono::system_clock::now();
        auto newFeatures = featureDetector->getFeatures(newImageData->mImage);
        auto end = std::chrono::system_clock::now();
        std::cout << "ORBFeatures took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - now).count() <<
                " ms" << std::endl;

        now = std::chrono::system_clock::now();
        auto matches = match(oldFeatures, newFeatures, 8);
        end = std::chrono::system_clock::now();
        std::cout << "Found " << matches.size() << " matches in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - now).count() <<
                " ms" << std::endl;
        plotter->plotMatches(oldImageData->mImage, newImageData->mImage, oldFeatures, newFeatures, matches);

        oldFeatures = newFeatures;
        oldImageData = newImageData;
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }

    render_loop.join();
    return 0;
}
