#include "ConfigParser.hpp"

#include <yaml-cpp/yaml.h>

#include <type_traits>
#include <cmath>
#include <stdexcept>

namespace schlam::config {
namespace {

YAML::Node section(const YAML::Node& root, const char* name) {
    const auto node = root[name];
    return node ? node : YAML::Node(YAML::NodeType::Undefined);
}

template<class T>
void read(const YAML::Node& node, const char* key, T& value) {
    if (!node || !node[key]) return;
    if constexpr (std::is_same_v<T, std::uint8_t>)
        value = static_cast<std::uint8_t>(node[key].as<unsigned>());
    else
        value = node[key].as<T>();
}
}

Config ConfigParser::load(const std::filesystem::path& path) {
    const auto root = YAML::LoadFile(path.string());
    Config c;
    auto n = section(root, "dataset");
    read(n, "type", c.dataset.type); read(n, "sequence", c.dataset.sequence);
    n.reset(section(root, "plotter"));
    read(n, "type", c.plotter.type); read(n, "zUp", c.plotter.zUp);
    n.reset(section(root, "features"));
    read(n, "count", c.features.count); read(n, "levels", c.features.levels); read(n, "scaleFactor", c.features.scaleFactor);
    read(n, "fastThreshold", c.features.fastThreshold); read(n, "fastArcLength", c.features.fastArcLength);
    read(n, "orientationRadius", c.features.orientationRadius); read(n, "border", c.features.border);
    read(n, "harrisBlockSize", c.features.harrisBlockSize); read(n, "harrisK", c.features.harrisK);
    read(n, "descriptorBlurSize", c.features.descriptorBlurSize); read(n, "descriptorBlurSigma", c.features.descriptorBlurSigma);
    n.reset(section(root, "reconstruction"));
    read(n, "randomSeed", c.reconstruction.randomSeed); read(n, "ransacCandidates", c.reconstruction.ransacCandidates);
    read(n, "homographySelectionRatio", c.reconstruction.homographySelectionRatio);
    read(n, "homographyChi2", c.reconstruction.homographyChi2); read(n, "essentialChi2", c.reconstruction.essentialChi2);
    read(n, "scoreChi2", c.reconstruction.scoreChi2); read(n, "reprojectionErrorSquared", c.reconstruction.reprojectionErrorSquared);
    read(n, "triangulationCosParallax", c.reconstruction.triangulationCosParallax); read(n, "homographyTriangulationCosParallax", c.reconstruction.homographyTriangulationCosParallax);
    read(n, "solutionAmbiguityRatio", c.reconstruction.solutionAmbiguityRatio); read(n, "minimumTriangulatedRatio", c.reconstruction.minimumTriangulatedRatio);
    read(n, "minimumParallaxDegrees", c.reconstruction.minimumParallaxDegrees); read(n, "essentialParallaxRank", c.reconstruction.essentialParallaxRank);
    read(n, "homographyParallaxRank", c.reconstruction.homographyParallaxRank);
    n.reset(section(root, "initialization"));
    read(n, "matchHammingDistance", c.initialization.matchHammingDistance); read(n, "matchWindowPixels", c.initialization.matchWindowPixels); read(n, "minimumMatches", c.initialization.minimumMatches);
    n.reset(section(root, "tracking"));
    read(n, "matchHammingDistance", c.tracking.matchHammingDistance); read(n, "projectionWindowPixels", c.tracking.projectionWindowPixels);
    read(n, "minimumPoseInliers", c.tracking.minimumPoseInliers);
    read(n, "recoveryTimeoutSeconds", c.tracking.recoveryTimeoutSeconds);
    if (!std::isfinite(c.tracking.recoveryTimeoutSeconds) || c.tracking.recoveryTimeoutSeconds < 0.0) {
        throw std::invalid_argument("tracking.recoveryTimeoutSeconds must be nonnegative and finite");
    }
    n.reset(section(root, "mapPoints"));
    read(n, "maximumViewingAngleDegrees", c.mapPoints.maximumViewingAngleDegrees); read(n, "descriptorHistorySize", c.mapPoints.descriptorHistorySize);
    read(n, "observationDistanceFactor", c.mapPoints.observationDistanceFactor);
    read(n, "cullingWindowSeconds", c.mapPoints.cullingWindowSeconds);
    if (!std::isfinite(c.mapPoints.observationDistanceFactor) || c.mapPoints.observationDistanceFactor < 1.0) {
        throw std::invalid_argument("mapPoints.observationDistanceFactor must be finite and at least one");
    }
    if (!std::isfinite(c.mapPoints.cullingWindowSeconds) || c.mapPoints.cullingWindowSeconds < 0.0) {
        throw std::invalid_argument("mapPoints.cullingWindowSeconds must be nonnegative and finite");
    }
    read(n, "minimumVisibilityCount", c.mapPoints.minimumVisibilityCount); read(n, "minimumTrackingSuccessRate", c.mapPoints.minimumTrackingSuccessRate);
    n.reset(section(root, "mapping"));
    read(n, "keyframeMinimumInliers", c.mapping.keyframeMinimumInliers); read(n, "keyframeMaximumInliers", c.mapping.keyframeMaximumInliers);
    read(n, "keyframeMinimumFrameGapWhenBusy", c.mapping.keyframeMinimumFrameGapWhenBusy); read(n, "triangulationPreviousKeyframes", c.mapping.triangulationPreviousKeyframes);
    read(n, "triangulationMatchHammingDistance", c.mapping.triangulationMatchHammingDistance); read(n, "queueCapacity", c.mapping.queueCapacity);
    n.reset(section(root, "optimizer"));
    read(n, "pyramidScaleFactor", c.optimizer.pyramidScaleFactor); read(n, "monocularChi2Threshold", c.optimizer.monocularChi2Threshold);
    read(n, "initialIterations", c.optimizer.initialIterations); read(n, "poseIterations", c.optimizer.poseIterations);
    read(n, "localIterations", c.optimizer.localIterations); read(n, "fullMapIterations", c.optimizer.fullMapIterations);
    read(n, "localMaxVariableKeyframes", c.optimizer.localMaxVariableKeyframes); read(n, "localMaxFixedKeyframes", c.optimizer.localMaxFixedKeyframes);
    read(n, "localMinimumSharedPoints", c.optimizer.localMinimumSharedPoints);
    read(n, "fullMapMinimumInlierObservations", c.optimizer.fullMapMinimumInlierObservations); read(n, "fullMapMinimumInlierRatio", c.optimizer.fullMapMinimumInlierRatio);
    n.reset(section(root, "motion"));
    std::string predictionModel = "linear";
    read(n, "predictionModel", predictionModel);
    if (predictionModel == "none") c.motion.predictionModel = MotionPredictionModel::NONE;
    else if (predictionModel == "linear") c.motion.predictionModel = MotionPredictionModel::LINEAR;
    else if (predictionModel == "fourthOrder") c.motion.predictionModel = MotionPredictionModel::FOURTH_ORDER;
    else throw std::invalid_argument("motion.predictionModel must be none, linear, or fourthOrder");
    read(n, "maximumExtrapolationIntervals", c.motion.maximumExtrapolationIntervals);
    return c;
}

}  // namespace schlam::config
