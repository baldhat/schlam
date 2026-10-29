#include "data/datasets/MAVDataset.hpp"
#include "data/datasets/KittiDataset.hpp"

#include "plotting/PangolinPlotter.hpp"
#include "plotting/DummyPlotter.h"

#include "schlam/ORBFeatureDetector.h"
#include "schlam/Optimizer.h"
#include "schlam/MotionModel.h"
#include "schlam/SchlamSystem.h"

#include <thread>


int main() {
    auto pTransformer = std::make_shared<tft::Transformer>();

    //auto dataloader = std::make_shared<KittiDataset>(pTransformer, "02");
    auto dataloader = std::make_shared<MAVDataset>(pTransformer, "V1_02_medium");

    /*std::shared_ptr<IPlotter> plotter = std::make_shared<PangolinPlotter>(
        pTransformer, dataloader->getCameraIntrinsics(),
        PangolinPlotter::Theme{}, dataloader->getImageAspectRatio(),
        true);*/
    std::shared_ptr<IPlotter> plotter = std::make_shared<DummyPlotter>();
    auto render_loop = std::thread([plotter] { plotter->run(); });

    auto featureDetector = std::make_shared<ORBFeatureDetector>(1000, plotter, 6);

    auto optimizer = std::make_shared<Optimizer>();

    auto motionModel = MotionModel();

    SchlamSystem schlamSystem(dataloader, pTransformer, plotter, featureDetector, optimizer, motionModel);
    schlamSystem.run();

    std::cout << "Finished with " << dataloader->percentageRead() << "% of data processed." << std::endl;

    if (render_loop.joinable()) {
        render_loop.join();
    }
    std::cout << "Plotter finished..." << std::endl;
    return 0;
}
