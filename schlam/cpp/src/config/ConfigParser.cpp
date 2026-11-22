#include "ConfigParser.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace schlam::config {
namespace {

YAML::Node section(const YAML::Node& root, const char* name) {
    const auto node = root[name];
    return node ? node : YAML::Node{};
}

std::filesystem::path resolveDatasetFile(const std::filesystem::path& configPath, const std::string& file) {
    if (file.starts_with("~/")) {
        const char* home = std::getenv("HOME");
        if (!home || !*home) {
            throw std::invalid_argument("Cannot resolve dataset file without HOME: " + file);
        }
        return std::filesystem::path(home) / file.substr(2);
    }
    return configPath.parent_path() / file;
}

bool hasInputScheme(const std::string& source) {
    constexpr auto letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    constexpr auto schemeCharacters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+.-";
    const auto colon = source.find(':');
    return colon != std::string::npos && source.find_first_of(letters) == 0 &&
           source.find_first_not_of(schemeCharacters) == colon;
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
    if (c.dataset.type == "RTPDataset") {
        c.dataset.type = "FFMPEGDataset";
    }
    if (c.dataset.type == "FFMPEGDataset") {
        std::string calibrationFile;
        read(n, "calibrationFile", calibrationFile);
        read(n, "source", c.dataset.source);
        if (!n["source"]) {
            read(n, "streamFile", c.dataset.source);
        }
        if (calibrationFile.empty() || c.dataset.source.empty()) {
            throw std::invalid_argument("FFMPEGDataset requires dataset.calibrationFile and dataset.source");
        }
        c.dataset.calibrationFile = resolveDatasetFile(path, calibrationFile);
        if (!hasInputScheme(c.dataset.source)) {
            c.dataset.source = resolveDatasetFile(path, c.dataset.source).string();
        }
    }
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
    read(n, "minimumPoseInliers", c.tracking.minimumPoseInliers); read(n, "minimumMatchesBeforeLost", c.tracking.minimumMatchesBeforeLost);
    n.reset(section(root, "mapPoints"));
    read(n, "maximumViewingAngleDegrees", c.mapPoints.maximumViewingAngleDegrees); read(n, "descriptorHistorySize", c.mapPoints.descriptorHistorySize);
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
    read(n, "maximumExtrapolationIntervals", c.motion.maximumExtrapolationIntervals);
    return c;
}

}  // namespace schlam::config
