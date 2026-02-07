#include "data/mav_dataloader.hpp"
#include "plotting/plotter.hpp"
#include "src/tft/rigid_transform_3d.hpp"

#include <chrono>
#include <thread>

#include "schlam/ORBFeatureDetector.h"

const std::filesystem::path mavDataPath(
    "/home/baldhat/dev/slam/MAV/vicon_room1/V1_01_easy/V1_01_easy/mav0/");

int main() {
    auto pTransformer = std::make_shared<tft::Transformer>();

    auto plotter = std::make_shared<Plotter>(pTransformer);
    auto render_loop = std::thread([plotter] { plotter->run(); });

    auto dataloader = std::make_shared<MAVDataloader>(mavDataPath, pTransformer);

    auto featureDetector = std::make_shared<ORBFeatureDetector>(500, plotter, 8);

    while (!dataloader->empty()) {
        std::cout << "Handling new image..." << std::endl;
        auto imageData = dataloader->getNextImageData();
        auto data = dataloader->getNextIMUData();
        auto imuData = data->first;
        auto gtData = data->second;

        pTransformer->registerTransform(std::make_shared<tft::RigidTransform3D>(
            "imu", "world", gtData.mRotation,
            gtData.mPosition));

        pTransformer->findTransform("imu", "world");
        pTransformer->findTransform("cam0", "world");

        plotter->addFrustum(imageData);

        featureDetector->getFeatures(imageData->mImage);
    }

    render_loop.join();
    return 0;
}
