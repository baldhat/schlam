#ifndef SCHLAM_SCHLAMVALIDATOR_H
#define SCHLAM_SCHLAMVALIDATOR_H

#include "SchlamSystem.h"

#include <chrono>
#include <optional>
#include <string>

// Offline trajectory evaluation. Does not change the input or use map-point errors
// as a substitute for camera trajectory accuracy. See docs/validation.md.
class SchlamValidator {
public:
    enum class Alignment { SIM3, SE3, NONE };
    enum class Status { SUCCESS, INSUFFICIENT_MATCHES, DEGENERATE_ALIGNMENT };

    struct Options {
        Alignment alignment{Alignment::SIM3};
        std::chrono::nanoseconds maximumTimestampDifference{std::chrono::milliseconds(20)};
        // Relative errors use overlapping pairs this many predicted poses apart.
        // Missing timestamp matches do not shorten the requested interval.
        std::size_t relativePoseDelta{1};
    };

    struct ErrorStatistics {
        std::size_t count{0};
        double rmse{0.0};
        double mean{0.0};
        double median{0.0};
        double standardDeviation{0.0}; // Population standard deviation.
        double minimum{0.0};
        double maximum{0.0};
    };

    struct Result {
        Status status{Status::INSUFFICIENT_MATCHES};
        std::string message;
        std::size_t estimatedPoses{0};
        std::size_t referencePoses{0};
        std::size_t matchedPoses{0};
        double associationCoverage{0.0}; // matched / estimated poses
        // Absent when processedFrames is unknown (e.g. a manually constructed result).
        std::optional<double> trackingCoverage; // estimated / processed frames
        double averageFrameRate{0.0};
        TrackingState finalTrackingState{TrackingState::INITIALIZING};

        // One global transform: p_aligned = scale * rotation * p_estimated + translation.
        double scale{1.0};
        Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
        Eigen::Vector3d translation{Eigen::Vector3d::Zero()};
        // Translation units follow GT (metres for EuRoC/KITTI); angles are radians.
        // Absent for unsuccessful evaluations; RPE is also absent if no delta pairs exist.
        std::optional<ErrorStatistics> absoluteTranslation;
        std::optional<ErrorStatistics> relativeTranslation;
        std::optional<ErrorStatistics> relativeRotationRadians;
    };

    SchlamValidator() = default;
    explicit SchlamValidator(Options options);

    // Near-SO(3) rotations (orthogonality/determinant errors <= 1e-4) are
    // projected onto SO(3) in a private copy to accommodate rounded calibration.
    // Malformed paths/options throw std::invalid_argument with pose context. Missing data and
    // unobservable alignment return a status, never misleading zero errors.
    [[nodiscard]] Result validate(const SchlamResult& result) const;

private:
    Options options_;
};

#endif
