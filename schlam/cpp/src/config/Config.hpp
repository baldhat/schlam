#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace schlam::config {

struct Dataset {
    std::string type = "MAVDataset";
    std::string sequence = "V1_02_medium";
};

struct Plotter {
    std::string type = "PangolinPlotter";
    bool zUp = true;
};

struct Features {
    std::uint32_t count = 1000;
    std::uint8_t levels = 6;
    double scaleFactor = 1.2;
    int fastThreshold = 7;
    int fastArcLength = 9;
    std::uint32_t orientationRadius = 15;
    int border = 19;
    int harrisBlockSize = 7;
    double harrisK = 0.04;
    int descriptorBlurSize = 7;
    double descriptorBlurSigma = 2.0;
};

struct Reconstruction {
    std::uint32_t randomSeed = 12344321;
    int ransacCandidates = 500;
    double homographySelectionRatio = 0.45;
    double homographyChi2 = 5.991;
    double essentialChi2 = 3.841;
    double scoreChi2 = 5.991;
    double reprojectionErrorSquared = 4.0;
    double triangulationCosParallax = 0.99998;
    double homographyTriangulationCosParallax = 0.9999;
    double solutionAmbiguityRatio = 0.75;
    double minimumTriangulatedRatio = 0.9;
    double minimumParallaxDegrees = 1.0;
    int essentialParallaxRank = 30;
    int homographyParallaxRank = 50;
};

struct Initialization {
    int matchHammingDistance = 20;
    double matchWindowPixels = 100.0;
    std::size_t minimumMatches = 50;
};

struct Tracking {
    int matchHammingDistance = 50;
    double projectionWindowPixels = 50.0;
    std::size_t minimumPoseInliers = 12;
    // Seconds since the last accepted pose; zero disables recovery.
    double recoveryTimeoutSeconds = 0.5;
};

struct MapPoints {
    double maximumViewingAngleDegrees = 60.0;
    double observationDistanceFactor = 2.0;
    // Quality culling only applies within this dataset-time window after creation.
    double cullingWindowSeconds = 3.0;
    std::size_t descriptorHistorySize = 10;
    std::uint64_t minimumVisibilityCount = 8;
    double minimumTrackingSuccessRate = 0.25;
};

struct Mapping {
    std::size_t keyframeMinimumInliers = 25;
    std::size_t keyframeMaximumInliers = 100;
    std::size_t keyframeMinimumFrameGapWhenBusy = 5;
    std::size_t triangulationPreviousKeyframes = 5;
    int triangulationMatchHammingDistance = 30;
    std::size_t queueCapacity = 1;
};

struct Optimizer {
    double pyramidScaleFactor = 1.2;
    double monocularChi2Threshold = 5.991;
    int initialIterations = 50;
    int poseIterations = 50;
    int localIterations = 50;
    int fullMapIterations = 50;
    std::size_t localMaxVariableKeyframes = 20;
    std::size_t localMaxFixedKeyframes = 40;
    std::uint32_t localMinimumSharedPoints = 15;
    std::size_t fullMapMinimumInlierObservations = 2;
    double fullMapMinimumInlierRatio = 0.8;
};

enum class MotionPredictionModel { NONE, LINEAR, FOURTH_ORDER };

struct Motion {
    MotionPredictionModel predictionModel = MotionPredictionModel::FOURTH_ORDER;
    double maximumExtrapolationIntervals = 2.0;
};

struct Config {
    Dataset dataset;
    Plotter plotter;
    Features features;
    Reconstruction reconstruction;
    Initialization initialization;
    Tracking tracking;
    MapPoints mapPoints;
    Mapping mapping;
    Optimizer optimizer;
    Motion motion;
};

}  // namespace schlam::config
