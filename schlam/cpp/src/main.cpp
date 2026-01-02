#include "data/mav_dataloader.hpp"
#include "plotting/plotter.hpp"
#include "src/tft/rigid_transform_3d.hpp"

#include <chrono>
#include <thread>

const std::filesystem::path mavDataPath(
    "/root/dev/data/schlam/vicon_room1/V1_01_easy/V1_01_easy/mav0/");

int main() {
  auto pTransformer = std::make_shared<tft::Transformer>();

  Plotter plotter(pTransformer);
  auto render_loop = std::thread(std::bind(&Plotter::run, &plotter));

  auto dataloader = std::make_shared<MAVDataloader>(mavDataPath, pTransformer);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  while (!dataloader->empty()) {
    std::cout << "Handling new image..." << std::endl;
    auto imageData = dataloader->getNextImageData();
    auto data = dataloader->getNextIMUData();
    auto imuData = data->first;
    auto gtData = data->second;

    pTransformer->registerTransform(std::make_shared<tft::RigidTransform3D>(
        "imu", "world", gtData.mRotation,
        gtData.mPosition));

    plotter.plotFrustum(imageData);
    std::this_thread::sleep_for(std::chrono::minutes(10));
  }

  render_loop.join();
  return 0;
}
