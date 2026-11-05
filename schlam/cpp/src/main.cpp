#include "data/datasets/MAVDataset.hpp"
#include "data/datasets/KittiDataset.hpp"

#include "plotting/PangolinPlotter.hpp"
#include "plotting/DummyPlotter.h"

#include "schlam/ORBFeatureDetector.h"
#include "schlam/Optimizer.h"
#include "schlam/MotionModel.h"
#include "schlam/SchlamSystem.h"
#include "config/ConfigParser.hpp"
#include "random.hpp"

#include <thread>
#include <opencv2/core.hpp>


int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "Usage: app [config.yaml]" << std::endl;
        return 2;
    }
    const auto settings = schlam::config::ConfigParser::load(
        argc == 2 ? argv[1] : SCHLAM_DEFAULT_CONFIG_PATH);
    utils::randomEngine().seed(settings.reconstruction.randomSeed);
    cv::theRNG() = cv::RNG(settings.reconstruction.randomSeed);
    auto pTransformer = std::make_shared<tft::Transformer>();

    std::shared_ptr<DatasetInterface> dataloader;
    if (settings.dataset.type == "MAVDataset") {
        dataloader = std::make_shared<MAVDataset>(pTransformer, settings.dataset.sequence);
    } else {
        dataloader = std::make_shared<KittiDataset>(pTransformer, settings.dataset.sequence);
    }

    std::shared_ptr<IPlotter> plotter;
    std::thread render_loop;
    if (settings.plotter.type == "PangolinPlotter") {
        plotter = std::make_shared<PangolinPlotter>(
            pTransformer, dataloader->getCameraIntrinsics(),
            PangolinPlotter::Theme{}, dataloader->getImageAspectRatio(),
            settings.plotter.zUp);
        render_loop = std::thread([plotter] { plotter->run(); });
    } else {
        plotter = std::make_shared<DummyPlotter>();
    }

    auto featureDetector = std::make_shared<ORBFeatureDetector>(settings.features, plotter);

    auto optimizer = std::make_shared<Optimizer>(settings.optimizer);

    auto motionModel = MotionModel(settings.motion.maximumExtrapolationIntervals);

    SchlamSystem schlamSystem(dataloader, pTransformer, plotter, featureDetector, optimizer, motionModel, settings);
    schlamSystem.run();

    std::cout << "Finished with " << dataloader->percentageRead() << "% of data processed." << std::endl;

    if (render_loop.joinable()) {
        render_loop.join();
    }
    std::cout << "Plotter finished..." << std::endl;
    return 0;
}
