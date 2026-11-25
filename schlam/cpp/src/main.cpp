#include "data/datasets/MAVDataset.hpp"
#include "data/datasets/KittiDataset.hpp"
#include "data/datasets/FFMPEGDataset.hpp"

#include "plotting/DummyPlotter.h"

#if SCHLAM_WITH_PANGOLIN
#include "plotting/PangolinPlotter.hpp"
#include <thread>
#endif

#include "schlam/feature_detection/orb/ORBFeatureDetector.h"
#include "schlam/optimization/Optimizer.h"
#include "schlam/MotionModel.h"
#include "schlam/SchlamSystem.h"
#include "schlam/SchlamValidator.h"
#include "config/ConfigParser.hpp"
#include "random.hpp"

#include <iostream>
#include <stdexcept>
#include <opencv2/core.hpp>

extern "C" {
#include <libavutil/log.h>
}

int main(int argc, char** argv) try {
    const auto settings = schlam::config::ConfigParser::load(
        argc == 2 ? argv[1] : SCHLAM_DEFAULT_CONFIG_PATH);
    utils::randomEngine().seed(settings.reconstruction.randomSeed);
    cv::theRNG() = cv::RNG(settings.reconstruction.randomSeed);
    auto pTransformer = std::make_shared<tft::Transformer>();

    std::shared_ptr<DatasetInterface> dataloader;
    if (settings.dataset.type == "MAVDataset") {
        dataloader = std::make_shared<MAVDataset>(pTransformer, settings.dataset.sequence);
    } else if (settings.dataset.type == "FFMPEGDataset") {
        av_log_set_level(AV_LOG_QUIET);
        dataloader = std::make_shared<FFMPEGDataset>(settings.dataset.calibrationFile, settings.dataset.source);
    } else {
        dataloader = std::make_shared<KittiDataset>(pTransformer, settings.dataset.sequence);
    }

    std::shared_ptr<IPlotter> plotter;
#if SCHLAM_WITH_PANGOLIN
    std::thread render_loop;
    if (settings.plotter.type == "PangolinPlotter") {
        plotter = std::make_shared<PangolinPlotter>(
            pTransformer, dataloader->getCameraIntrinsics(),
            PangolinPlotter::Theme{}, dataloader->getImageAspectRatio(),
            settings.plotter.zUp);
        render_loop = std::thread([plotter] { plotter->run(); });
    }
#endif
    if (!plotter) {
        plotter = std::make_shared<DummyPlotter>();
    }

    auto featureDetector = std::make_shared<ORBFeatureDetector>(settings.features, plotter);

    auto optimizer = std::make_shared<Optimizer>(settings.optimizer);

    auto motionModel = MotionModel(settings.motion.maximumExtrapolationIntervals);

    SchlamSystem schlamSystem(dataloader, pTransformer, plotter, featureDetector, optimizer, motionModel, settings);
    const auto result = schlamSystem.run();

    if (settings.dataset.type == "FFMPEGDataset") {
        std::cout << "Processed " << result.processedFrames << " video frames." << std::endl;
    } else {
        std::cout << "Finished with " << dataloader->percentageRead() << "% of data processed." << std::endl;
    }
    std::cout << "Created " << result.mapPoints.size() << " map points and "
              << result.predictedPath.size() << " predicted poses at "
              << result.averageFrameRate << " FPS." << std::endl;

    int exitCode = 0;
    try {
        const auto validation = SchlamValidator().validate(result);
        if (dataloader->hasGT()) {
            std::cout << "Trajectory validation: " << validation.message << " ("
                      << validation.matchedPoses << "/" << validation.estimatedPoses
                      << " poses associated)." << std::endl;
        }
        if (validation.trackingCoverage) {
            std::cout << "Tracking coverage of processed frames: "
                      << 100.0 * *validation.trackingCoverage << "%." << std::endl;
        }
        if (validation.absoluteTranslation) {
            std::cout << "Sim(3) alignment scale: " << validation.scale
                      << "; ATE RMSE: " << validation.absoluteTranslation->rmse << " m." << std::endl;
        }
        if (validation.relativeTranslation) {
            std::cout << "RPE RMSE: " << validation.relativeTranslation->rmse << " m, "
                      << validation.relativeRotationRadians->rmse << " rad ("
                      << validation.relativeTranslation->count << " adjacent pose pairs)." << std::endl;
        }
    } catch (const std::invalid_argument& error) {
        std::cerr << "Trajectory validation failed: " << error.what() << std::endl;
        exitCode = 1;
    }

#if SCHLAM_WITH_PANGOLIN
    if (render_loop.joinable()) {
        render_loop.join();
    }
#endif
    std::cout << "Plotter finished..." << std::endl;
    return exitCode;
} catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
}
