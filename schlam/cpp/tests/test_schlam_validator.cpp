#include <gtest/gtest.h>

#include "schlam/SchlamValidator.h"

#include <limits>
#include <numbers>
#include <stdexcept>

namespace {
utils::TTimestamp timeAt(int index) {
    return utils::TTimestamp(std::chrono::seconds(1700000000) + std::chrono::milliseconds(100 * index));
}

SchlamResult trajectory() {
    SchlamResult result;
    const std::vector<Eigen::Vector3d> positions{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 1}, {0, 0, 1}};
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto rotation = Eigen::AngleAxisd(0.15 * i, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        result.predictedPath.push_back({timeAt(i), rotation, positions[i]});
    }
    result.groundTruthPath = result.predictedPath;
    result.processedFrames = positions.size();
    result.finalTrackingState = TrackingState::TRACKING;
    result.averageFrameRate = 30.0;
    return result;
}

SchlamValidator withoutAlignment(std::size_t delta = 1) {
    SchlamValidator::Options options;
    options.alignment = SchlamValidator::Alignment::NONE;
    options.relativePoseDelta = delta;
    return SchlamValidator(options);
}

TEST(SchlamValidator, IdenticalTrajectoryHasZeroErrorsAndReportsMetadata) {
    auto input = trajectory();
    input.processedFrames = 10;
    input.finalTrackingState = TrackingState::LOST;
    const auto report = SchlamValidator().validate(input);
    ASSERT_EQ(report.status, SchlamValidator::Status::SUCCESS);
    ASSERT_TRUE(report.absoluteTranslation);
    ASSERT_TRUE(report.relativeTranslation);
    ASSERT_TRUE(report.relativeRotationRadians);
    EXPECT_NEAR(report.absoluteTranslation->rmse, 0.0, 1e-12);
    EXPECT_NEAR(report.relativeTranslation->rmse, 0.0, 1e-12);
    EXPECT_NEAR(report.relativeRotationRadians->rmse, 0.0, 1e-12);
    EXPECT_EQ(report.absoluteTranslation->count, 5);
    EXPECT_EQ(report.relativeTranslation->count, 4);
    EXPECT_DOUBLE_EQ(*report.trackingCoverage, 0.5);
    EXPECT_DOUBLE_EQ(report.associationCoverage, 1.0);
    EXPECT_DOUBLE_EQ(report.averageFrameRate, 30.0);
    EXPECT_EQ(report.finalTrackingState, TrackingState::LOST);
}

TEST(SchlamValidator, RecoversGlobalSimilarityIncludingCameraOrientation) {
    auto input = trajectory();
    const auto rotation = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    const Eigen::Vector3d translation(3, -2, 7);
    for (auto& pose : input.groundTruthPath) {
        pose.position = 2.5 * rotation * pose.position + translation;
        pose.rotation = rotation * pose.rotation;
    }
    const auto report = SchlamValidator().validate(input);
    ASSERT_EQ(report.status, SchlamValidator::Status::SUCCESS);
    EXPECT_NEAR(report.scale, 2.5, 1e-12);
    EXPECT_TRUE(report.rotation.isApprox(rotation, 1e-12));
    EXPECT_TRUE(report.translation.isApprox(translation, 1e-12));
    EXPECT_NEAR(report.absoluteTranslation->rmse, 0.0, 1e-12);
    EXPECT_NEAR(report.relativeTranslation->rmse, 0.0, 1e-12);
    EXPECT_NEAR(report.relativeRotationRadians->rmse, 0.0, 1e-12);
    EXPECT_TRUE(input.predictedPath.front().position.isZero());
}

TEST(SchlamValidator, RigidAlignmentPreservesScaleError) {
    auto input = trajectory();
    for (auto& pose : input.groundTruthPath) pose.position *= 2.0;
    SchlamValidator::Options options;
    options.alignment = SchlamValidator::Alignment::SE3;
    const auto report = SchlamValidator(options).validate(input);
    ASSERT_EQ(report.status, SchlamValidator::Status::SUCCESS);
    EXPECT_DOUBLE_EQ(report.scale, 1.0);
    EXPECT_GT(report.absoluteTranslation->rmse, 0.1);
    EXPECT_GT(report.relativeTranslation->rmse, 0.1);
}

TEST(SchlamValidator, KnownTranslationErrorsHaveExpectedStatistics) {
    auto input = trajectory();
    for (std::size_t i = 0; i < input.predictedPath.size(); ++i) {
        input.predictedPath[i].rotation.setIdentity();
        input.groundTruthPath[i].rotation.setIdentity();
        input.predictedPath[i].position.x() += i;
    }
    const auto report = withoutAlignment().validate(input);
    const auto stats = *report.absoluteTranslation;
    EXPECT_NEAR(stats.rmse, std::sqrt(6.0), 1e-12);
    EXPECT_DOUBLE_EQ(stats.mean, 2.0);
    EXPECT_DOUBLE_EQ(stats.median, 2.0);
    EXPECT_NEAR(stats.standardDeviation, std::sqrt(2.0), 1e-12);
    EXPECT_DOUBLE_EQ(stats.minimum, 0.0);
    EXPECT_DOUBLE_EQ(stats.maximum, 4.0);
    EXPECT_DOUBLE_EQ(report.relativeTranslation->rmse, 1.0);
    EXPECT_DOUBLE_EQ(withoutAlignment(2).validate(input).relativeTranslation->rmse, 2.0);
}

TEST(SchlamValidator, RotationDriftUsesRelativeCameraToWorldRotationsInRadians) {
    auto input = trajectory();
    for (std::size_t i = 0; i < input.predictedPath.size(); ++i) {
        input.groundTruthPath[i].rotation.setIdentity();
        input.predictedPath[i].rotation = Eigen::AngleAxisd(i * std::numbers::pi / 6,
            Eigen::Vector3d::UnitY()).toRotationMatrix();
    }
    const auto report = withoutAlignment().validate(input);
    EXPECT_NEAR(report.relativeRotationRadians->rmse, std::numbers::pi / 6, 1e-12);
    // Orientation affects translation RPE even when camera centres match exactly.
    EXPECT_GT(report.relativeTranslation->rmse, 0.1);
    EXPECT_DOUBLE_EQ(report.absoluteTranslation->rmse, 0.0);
}

TEST(SchlamValidator, TimestampToleranceIsInclusiveAndNanosecondPrecise) {
    auto input = trajectory();
    for (auto& pose : input.groundTruthPath) pose.timestamp += std::chrono::milliseconds(20);
    auto report = SchlamValidator().validate(input);
    EXPECT_EQ(report.matchedPoses, 5);
    for (auto& pose : input.groundTruthPath) pose.timestamp += std::chrono::nanoseconds(1);
    report = SchlamValidator().validate(input);
    EXPECT_EQ(report.matchedPoses, 0);
    EXPECT_EQ(report.status, SchlamValidator::Status::INSUFFICIENT_MATCHES);
    EXPECT_FALSE(report.absoluteTranslation);
}

TEST(SchlamValidator, NearestReferenceIsSelectedAndNeverReused) {
    auto input = trajectory();
    auto earlier = input.groundTruthPath[0];
    earlier.timestamp -= std::chrono::milliseconds(10);
    earlier.position.x() = 50;
    input.groundTruthPath.insert(input.groundTruthPath.begin(), earlier);
    EXPECT_NEAR(SchlamValidator().validate(input).absoluteTranslation->rmse, 0.0, 1e-12);

    input = trajectory();
    input.predictedPath[1].timestamp = input.predictedPath[0].timestamp + std::chrono::milliseconds(1);
    auto report = withoutAlignment().validate(input);
    EXPECT_EQ(report.matchedPoses, 4);
    EXPECT_DOUBLE_EQ(report.associationCoverage, 0.8);
    // Pairs 2->3 and 3->4 only; the unmatched pose must not produce a 0->2 pair.
    EXPECT_EQ(report.relativeTranslation->count, 2);
}

TEST(SchlamValidator, MissingMatchesDoNotCompressRelativePoseDelta) {
    auto input = trajectory();
    input.groundTruthPath.erase(input.groundTruthPath.begin() + 2);
    auto report = withoutAlignment(2).validate(input);
    ASSERT_TRUE(report.relativeTranslation);
    EXPECT_EQ(report.relativeTranslation->count, 1); // Only predicted indices 1->3.
    EXPECT_DOUBLE_EQ(report.relativeTranslation->rmse, 0.0);
    report = withoutAlignment(10).validate(input);
    EXPECT_EQ(report.status, SchlamValidator::Status::SUCCESS);
    EXPECT_FALSE(report.relativeTranslation);
    EXPECT_FALSE(report.relativeRotationRadians);
}

TEST(SchlamValidator, MissingGroundTruthAndShortPathsDoNotReturnZeroErrors) {
    EXPECT_FALSE(SchlamValidator().validate({}).absoluteTranslation);
    auto input = trajectory();
    input.groundTruthPath.clear();
    EXPECT_EQ(SchlamValidator().validate(input).status, SchlamValidator::Status::INSUFFICIENT_MATCHES);
    input = trajectory();
    input.predictedPath.resize(2);
    EXPECT_EQ(SchlamValidator().validate(input).status, SchlamValidator::Status::INSUFFICIENT_MATCHES);
}

TEST(SchlamValidator, RejectsDegenerateAlignmentButAllowsPlanarMotion) {
    auto input = trajectory();
    for (auto& pose : input.predictedPath) pose.position.setZero();
    auto report = SchlamValidator().validate(input);
    EXPECT_EQ(report.status, SchlamValidator::Status::DEGENERATE_ALIGNMENT);
    EXPECT_FALSE(report.absoluteTranslation);
    input = trajectory();
    for (auto& pose : input.predictedPath) pose.position = Eigen::Vector3d(pose.position.x(), 0, 0);
    EXPECT_EQ(SchlamValidator().validate(input).status, SchlamValidator::Status::DEGENERATE_ALIGNMENT);
    input = trajectory();
    for (auto& pose : input.predictedPath) pose.position.z() = 0.0;
    input.groundTruthPath = input.predictedPath;
    EXPECT_EQ(SchlamValidator().validate(input).status, SchlamValidator::Status::SUCCESS);
}

TEST(SchlamValidator, SimilarityDoesNotFitAReflection) {
    auto input = trajectory();
    for (auto& pose : input.groundTruthPath) pose.position.x() *= -1;
    const auto report = SchlamValidator().validate(input);
    ASSERT_EQ(report.status, SchlamValidator::Status::SUCCESS);
    EXPECT_NEAR(report.rotation.determinant(), 1.0, 1e-12);
    EXPECT_GT(report.absoluteTranslation->rmse, 0.1);
}

TEST(SchlamValidator, InvalidDataAndOptionsThrow) {
    SchlamValidator::Options options;
    options.relativePoseDelta = 0;
    EXPECT_THROW(SchlamValidator{options}, std::invalid_argument);
    options.relativePoseDelta = 1;
    options.maximumTimestampDifference = std::chrono::nanoseconds(-1);
    EXPECT_THROW(SchlamValidator{options}, std::invalid_argument);
    auto input = trajectory();
    input.predictedPath[1].timestamp = input.predictedPath[0].timestamp;
    EXPECT_THROW(static_cast<void>(SchlamValidator().validate(input)), std::invalid_argument);
    input = trajectory();
    std::swap(input.groundTruthPath[1], input.groundTruthPath[2]);
    EXPECT_THROW(static_cast<void>(SchlamValidator().validate(input)), std::invalid_argument);
    input = trajectory();
    input.predictedPath[0].position.x() = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(static_cast<void>(SchlamValidator().validate(input)), std::invalid_argument);
    input = trajectory();
    input.groundTruthPath[0].rotation *= 2;
    EXPECT_THROW(static_cast<void>(SchlamValidator().validate(input)), std::invalid_argument);
    input = trajectory();
    input.processedFrames = 1;
    EXPECT_THROW(static_cast<void>(SchlamValidator().validate(input)), std::invalid_argument);
}

TEST(SchlamValidator, RoundedEuRoCCalibrationIsAcceptedWithoutChangingInput) {
    auto input = trajectory();
    const Eigen::Matrix3d calibration = (Eigen::Matrix3d() <<
        -0.0167748, -0.99986, -0.00131135,
        -0.995625, 0.0168273, -0.0918834,
        0.0918923, -0.000235506, -0.995769).finished();
    ASSERT_GT((calibration.transpose() * calibration - Eigen::Matrix3d::Identity()).norm(), 1e-6);
    for (auto& pose : input.predictedPath) pose.rotation *= calibration;
    input.groundTruthPath = input.predictedPath;
    const auto original = input;
    const auto report = SchlamValidator().validate(input);
    ASSERT_EQ(report.status, SchlamValidator::Status::SUCCESS);
    EXPECT_NEAR(report.absoluteTranslation->rmse, 0.0, 1e-12);
    EXPECT_NEAR(report.relativeTranslation->rmse, 0.0, 1e-12);
    EXPECT_NEAR(report.relativeRotationRadians->rmse, 0.0, 1e-12);
    for (std::size_t i = 0; i < input.predictedPath.size(); ++i) {
        EXPECT_TRUE(input.predictedPath[i].rotation.isApprox(original.predictedPath[i].rotation, 0));
        EXPECT_TRUE(input.groundTruthPath[i].rotation.isApprox(original.groundTruthPath[i].rotation, 0));
    }
}

TEST(SchlamValidator, RoundedRotationsAreProjectedBeforeComputingRelativeErrors) {
    auto input = trajectory();
    const Eigen::Matrix3d exact = Eigen::AngleAxisd(std::numbers::pi / 4, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Matrix3d rounded = (Eigen::Matrix3d() <<
        0.70711, -0.70711, 0, 0.70711, 0.70711, 0, 0, 0, 1).finished();
    for (auto& pose : input.predictedPath) pose.rotation *= rounded;
    for (auto& pose : input.groundTruthPath) pose.rotation *= exact;
    const auto report = SchlamValidator().validate(input);
    ASSERT_EQ(report.status, SchlamValidator::Status::SUCCESS);
    EXPECT_NEAR(report.relativeTranslation->rmse, 0.0, 1e-12);
    EXPECT_NEAR(report.relativeRotationRadians->rmse, 0.0, 1e-12);
}

TEST(SchlamValidator, RejectsLargeRotationErrorsAndReflections) {
    auto input = trajectory();
    input.predictedPath[0].rotation(0, 1) += 0.001;
    EXPECT_THROW(static_cast<void>(SchlamValidator().validate(input)), std::invalid_argument);
    input = trajectory();
    input.predictedPath[0].rotation(0, 0) = -1;
    EXPECT_THROW(static_cast<void>(SchlamValidator().validate(input)), std::invalid_argument);
    input = trajectory();
    input.groundTruthPath[0].rotation(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(static_cast<void>(SchlamValidator().validate(input)), std::invalid_argument);
}

TEST(SchlamValidator, InvalidPoseDiagnosticsIdentifyPathIndexAndReason) {
    auto input = trajectory();
    input.groundTruthPath[2].position.x() = std::numeric_limits<double>::quiet_NaN();
    try {
        static_cast<void>(SchlamValidator().validate(input));
        FAIL() << "Expected invalid ground truth to be rejected";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("groundTruthPath pose 2"), std::string::npos);
        EXPECT_NE(message.find("timestamp ticks"), std::string::npos);
        EXPECT_NE(message.find("position contains NaN or infinity"), std::string::npos);
    }
    input = trajectory();
    input.predictedPath[3].rotation *= 2;
    try {
        static_cast<void>(SchlamValidator().validate(input));
        FAIL() << "Expected invalid rotation to be rejected";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("predictedPath pose 3"), std::string::npos);
        EXPECT_NE(message.find("orthogonality error="), std::string::npos);
        EXPECT_NE(message.find("determinant="), std::string::npos);
    }
}
} // namespace
