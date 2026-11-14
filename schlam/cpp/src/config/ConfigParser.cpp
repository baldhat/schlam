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
    std::string featureType = "orb";
    read(n, "type", featureType);
    if (featureType == "orb") c.features.type = FeatureType::ORB;
    else if (featureType == "xfeat") c.features.type = FeatureType::XFEAT;
    else throw std::invalid_argument("features.type must be orb or xfeat");
    const auto xfeat = n ? section(n, "xfeat") : YAML::Node(YAML::NodeType::Undefined);
    std::string imageSizing = "resize";
    read(xfeat, "imageSizing", imageSizing);
    if (imageSizing == "resize") c.features.xfeat.imageSizing = XFeatImageSizing::RESIZE;
    else if (imageSizing == "pad") c.features.xfeat.imageSizing = XFeatImageSizing::PAD;
    else throw std::invalid_argument("features.xfeat.imageSizing must be resize or pad");
    read(xfeat, "modelPath", c.features.xfeat.modelPath);
    read(xfeat, "inferenceThreads", c.features.xfeat.inferenceThreads);
    read(xfeat, "detectionThreshold", c.features.xfeat.detectionThreshold);
    read(xfeat, "matchDistance", c.features.xfeat.matchDistance);
    if (!c.features.xfeat.modelPath.empty()) {
        const std::filesystem::path modelPath(c.features.xfeat.modelPath);
        c.features.xfeat.modelPath = std::filesystem::absolute(path.parent_path() / modelPath).lexically_normal().string();
    }
    if (c.features.type == FeatureType::XFEAT) {
        if (c.features.xfeat.inferenceThreads < 1 || !std::isfinite(c.features.xfeat.detectionThreshold) ||
            c.features.xfeat.detectionThreshold < 0.0 || c.features.xfeat.detectionThreshold > 1.0 ||
            !std::isfinite(c.features.xfeat.matchDistance) || c.features.xfeat.matchDistance <= 0.0 ||
            c.features.xfeat.matchDistance > 2.0) {
            throw std::invalid_argument("Invalid XFeat settings: threads >= 1, detectionThreshold in [0,1], matchDistance in (0,2]");
        }
        c.initialization.matchDistance = c.features.xfeat.matchDistance;
        c.tracking.matchDistance = c.features.xfeat.matchDistance;
        c.mapping.triangulationMatchDistance = c.features.xfeat.matchDistance;
    }
    read(n, "count", c.features.count);
    const auto orb = n ? section(n, "orb") : YAML::Node(YAML::NodeType::Undefined);
    read(orb, "levels", c.features.orb.levels); read(orb, "scaleFactor", c.features.orb.scaleFactor);
    read(orb, "fastThreshold", c.features.orb.fastThreshold); read(orb, "fastArcLength", c.features.orb.fastArcLength);
    read(orb, "orientationRadius", c.features.orb.orientationRadius); read(orb, "border", c.features.orb.border);
    read(orb, "harrisBlockSize", c.features.orb.harrisBlockSize); read(orb, "harrisK", c.features.orb.harrisK);
    read(orb, "descriptorBlurSize", c.features.orb.descriptorBlurSize); read(orb, "descriptorBlurSigma", c.features.orb.descriptorBlurSigma);
    read(orb, "initializationMatchDistance", c.features.orb.initializationMatchDistance);
    read(orb, "trackingMatchDistance", c.features.orb.trackingMatchDistance);
    read(orb, "triangulationMatchDistance", c.features.orb.triangulationMatchDistance);
    if (c.features.type == FeatureType::ORB) {
        c.initialization.matchDistance = c.features.orb.initializationMatchDistance;
        c.tracking.matchDistance = c.features.orb.trackingMatchDistance;
        c.mapping.triangulationMatchDistance = c.features.orb.triangulationMatchDistance;
    }
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
    if (c.features.type == FeatureType::ORB) read(n, "matchHammingDistance", c.initialization.matchDistance);
    read(n, "matchDistance", c.initialization.matchDistance);
    read(n, "matchWindowPixels", c.initialization.matchWindowPixels); read(n, "minimumMatches", c.initialization.minimumMatches);
    n.reset(section(root, "tracking"));
    if (c.features.type == FeatureType::ORB) read(n, "matchHammingDistance", c.tracking.matchDistance);
    read(n, "matchDistance", c.tracking.matchDistance);
    read(n, "projectionWindowPixels", c.tracking.projectionWindowPixels);
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
    if (c.features.type == FeatureType::ORB) read(n, "triangulationMatchHammingDistance", c.mapping.triangulationMatchDistance);
    read(n, "triangulationMatchDistance", c.mapping.triangulationMatchDistance);
    read(n, "queueCapacity", c.mapping.queueCapacity);
    for (const auto distance : {c.initialization.matchDistance, c.tracking.matchDistance, c.mapping.triangulationMatchDistance}) {
        if (!std::isfinite(distance) || distance <= 0.0 ||
            (c.features.type == FeatureType::XFEAT && distance > 2.0))
            throw std::invalid_argument("Descriptor match distances must be positive and finite (at most 2 for XFeat)");
    }
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
