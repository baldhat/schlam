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
