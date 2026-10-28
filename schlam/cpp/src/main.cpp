#include "data/datasets/MAVDataset.hpp"
#include "plotting/PangolinPlotter.hpp"
#include "schlam/Ransac.h"
#include "tft/rigid_transform_3d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

#include "data/datasets/KittiDataset.hpp"
#include "plotting/DummyPlotter.h"
#include "schlam/ORBFeatureDetector.h"
#include "schlam/Matcher.h"
#include "schlam/Optimizer.h"
#include "schlam/Frame.h"
#include "schlam/Map.h"
#include "schlam/MotionModel.h"
#include "schlam/Profiler.h"
#include "schlam/SchlamSystem.h"


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
