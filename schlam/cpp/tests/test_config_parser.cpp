#include "config/ConfigParser.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {
class MotionConfigTest : public ::testing::Test {
protected:
    std::filesystem::path directory = std::filesystem::temp_directory_path() /
        ("schlam-motion-config-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

    void SetUp() override { std::filesystem::create_directory(directory); }
    void TearDown() override { std::filesystem::remove_all(directory); }

    schlam::config::Config load(const std::string& yaml) {
        const auto path = directory / "config.yaml";
        {
            std::ofstream stream(path);
            stream << yaml;
        }
        return schlam::config::ConfigParser::load(path);
    }
};
}

TEST_F(MotionConfigTest, MissingOptionsKeepLinearPredictionAndHalfSecondRecovery) {
    const auto config = load("{}\n");
    EXPECT_EQ(config.motion.predictionModel, schlam::config::MotionPredictionModel::LINEAR);
    EXPECT_DOUBLE_EQ(config.motion.maximumExtrapolationIntervals, 2.0);
    EXPECT_DOUBLE_EQ(config.tracking.recoveryTimeoutSeconds, 0.5);
}

TEST_F(MotionConfigTest, LoadsEachPredictionModelAndItsLimitsFromYaml) {
    using schlam::config::MotionPredictionModel;
    for (const auto& [name, expected] : {
            std::pair{"none", MotionPredictionModel::NONE},
            std::pair{"linear", MotionPredictionModel::LINEAR},
            std::pair{"fourthOrder", MotionPredictionModel::FOURTH_ORDER}}) {
        SCOPED_TRACE(name);
        const auto config = load(std::string("motion:\n  predictionModel: ") + name +
            "\n  maximumExtrapolationIntervals: 3.0\ntracking:\n  recoveryTimeoutSeconds: 0.75\n");
        EXPECT_EQ(config.motion.predictionModel, expected);
        EXPECT_DOUBLE_EQ(config.motion.maximumExtrapolationIntervals, 3.0);
        EXPECT_DOUBLE_EQ(config.tracking.recoveryTimeoutSeconds, 0.75);
    }
}

TEST_F(MotionConfigTest, ZeroRecoveryTimeoutDisablesRecovery) {
    EXPECT_DOUBLE_EQ(load("tracking:\n  recoveryTimeoutSeconds: 0\n").tracking.recoveryTimeoutSeconds, 0.0);
}

TEST_F(MotionConfigTest, RejectsUnknownPredictionModel) {
    EXPECT_THROW(load("motion:\n  predictionModel: fourthOder\n"), std::invalid_argument);
}

TEST_F(MotionConfigTest, RejectsNegativeAndNonfiniteRecoveryTimeouts) {
    for (const auto* timeout : {"-0.1", ".nan", ".inf", "-.inf"}) {
        SCOPED_TRACE(timeout);
        EXPECT_THROW(load(std::string("tracking:\n  recoveryTimeoutSeconds: ") + timeout + "\n"), std::invalid_argument);
    }
}

TEST_F(MotionConfigTest, LoadsMapPointCullingAndDistanceSettings) {
    const auto defaults = load("{}\n");
    EXPECT_DOUBLE_EQ(defaults.mapPoints.cullingWindowSeconds, 3.0);
    EXPECT_DOUBLE_EQ(defaults.mapPoints.observationDistanceFactor, 2.0);
    const auto config = load("mapPoints:\n  cullingWindowSeconds: 1.25\n  observationDistanceFactor: 3.5\n");
    EXPECT_DOUBLE_EQ(config.mapPoints.cullingWindowSeconds, 1.25);
    EXPECT_DOUBLE_EQ(config.mapPoints.observationDistanceFactor, 3.5);
    EXPECT_DOUBLE_EQ(load("mapPoints:\n  cullingWindowSeconds: 0\n").mapPoints.cullingWindowSeconds, 0.0);
    EXPECT_DOUBLE_EQ(load("mapPoints:\n  observationDistanceFactor: 1\n").mapPoints.observationDistanceFactor, 1.0);
}

TEST_F(MotionConfigTest, RejectsInvalidMapPointCullingWindowsAndDistanceFactors) {
    for (const auto* value : {"-1", ".nan", ".inf", "-.inf"}) {
        EXPECT_THROW(load(std::string("mapPoints:\n  cullingWindowSeconds: ") + value + "\n"), std::invalid_argument);
    }
    for (const auto* value : {"0", "0.99", "-1", ".nan", ".inf", "-.inf"}) {
        EXPECT_THROW(load(std::string("mapPoints:\n  observationDistanceFactor: ") + value + "\n"), std::invalid_argument);
    }
}

TEST_F(MotionConfigTest, LoadsSeparateOrbAndXFeatProfilesWithSharedFeatureCount) {
    const auto config = load("features:\n  type: orb\n  count: 123\n  orb:\n    levels: 4\n    scaleFactor: 1.3\n"
        "    fastThreshold: 12\n    border: 22\n    initializationMatchDistance: 17\n    trackingMatchDistance: 47\n"
        "    triangulationMatchDistance: 27\n  xfeat:\n    detectionThreshold: 0.03\n    inferenceThreads: 2\n    matchDistance: 0.2\n");
    EXPECT_EQ(config.features.type, schlam::config::FeatureType::ORB);
    EXPECT_EQ(config.features.count, 123);
    EXPECT_EQ(config.features.orb.levels, 4);
    EXPECT_EQ(config.features.orb.fastThreshold, 12);
    EXPECT_EQ(config.features.orb.border, 22);
    EXPECT_DOUBLE_EQ(config.features.orb.scaleFactor, 1.3);
    EXPECT_DOUBLE_EQ(config.initialization.matchDistance, 17);
    EXPECT_DOUBLE_EQ(config.tracking.matchDistance, 47);
    EXPECT_DOUBLE_EQ(config.mapping.triangulationMatchDistance, 27);
    EXPECT_EQ(config.features.xfeat.inferenceThreads, 2);
}

TEST_F(MotionConfigTest, XFeatSelectionUsesCosineThresholdsAndResolvesConfigRelativeModelPaths) {
    const auto config = load("features:\n  type: xfeat\n  xfeat:\n    modelPath: assets/model.onnx\n    imageSizing: pad\n"
        "initialization:\n  matchHammingDistance: 20\ntracking:\n  matchHammingDistance: 50\nmapping:\n  triangulationMatchHammingDistance: 30\n");
    EXPECT_EQ(config.features.type, schlam::config::FeatureType::XFEAT);
    EXPECT_EQ(config.features.xfeat.imageSizing, schlam::config::XFeatImageSizing::PAD);
    EXPECT_EQ(config.features.xfeat.modelPath, (directory / "assets/model.onnx").string());
    EXPECT_DOUBLE_EQ(config.initialization.matchDistance, 0.18);
    EXPECT_DOUBLE_EQ(config.tracking.matchDistance, 0.18);
    EXPECT_DOUBLE_EQ(config.mapping.triangulationMatchDistance, 0.18);
    const auto overrides = load("features:\n  type: xfeat\n  xfeat:\n    matchDistance: 0.2\n"
        "tracking:\n  matchDistance: 0.15\n");
    EXPECT_DOUBLE_EQ(overrides.initialization.matchDistance, 0.2);
    EXPECT_DOUBLE_EQ(overrides.tracking.matchDistance, 0.15);
    EXPECT_DOUBLE_EQ(overrides.mapping.triangulationMatchDistance, 0.2);
}

TEST_F(MotionConfigTest, RejectsUnknownExtractorsAndInvalidXFeatOptions) {
    EXPECT_THROW(load("features:\n  type: sift\n"), std::invalid_argument);
    for (const auto* option : {"inferenceThreads: 0", "detectionThreshold: -0.1", "detectionThreshold: 1.1",
                              "detectionThreshold: .nan", "matchDistance: 0", "matchDistance: 2.1", "matchDistance: .inf", "imageSizing: crop"}) {
        EXPECT_THROW(load(std::string("features:\n  type: xfeat\n  xfeat:\n    ") + option + "\n"), std::invalid_argument);
    }
}
