#include "SchlamValidator.h"

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <utility>

namespace {
std::vector<PredictedPose> normalizedPath(const std::vector<PredictedPose>& path, const char* name) {
    auto normalized = path;
    for (std::size_t i = 0; i < path.size(); ++i) {
        const auto& pose = path[i];
        const auto fail = [&](const std::string& reason) {
            throw std::invalid_argument(std::string(name) + " pose " + std::to_string(i) +
                " (timestamp ticks " + std::to_string(pose.timestamp.time_since_epoch().count()) + "): " + reason);
        };
        if (!pose.position.allFinite()) fail("position contains NaN or infinity");
        if (!pose.rotation.allFinite()) fail("rotation contains NaN or infinity");
        // Dataset calibration matrices may be rounded to six decimal places.
        // Admit only small serialization errors, then project onto SO(3) so the
        // RPE computation can safely use transposes as rotation inverses.
        constexpr double rotationTolerance = 1e-4;
        const double orthogonalityError = (pose.rotation.transpose() * pose.rotation - Eigen::Matrix3d::Identity()).norm();
        const double determinant = pose.rotation.determinant();
        if (!std::isfinite(orthogonalityError) || !std::isfinite(determinant) ||
            orthogonalityError > rotationTolerance || std::abs(determinant - 1.0) > rotationTolerance) {
            std::ostringstream reason;
            reason << "rotation is not close to SO(3): orthogonality error=" << orthogonalityError
                   << ", determinant=" << determinant << ", tolerance=" << rotationTolerance;
            fail(reason.str());
        }
        const Eigen::JacobiSVD<Eigen::Matrix3d> svd(pose.rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
        normalized[i].rotation = svd.matrixU() * svd.matrixV().transpose();
        if (i > 0 && pose.timestamp <= path[i - 1].timestamp) {
            fail("timestamps must be strictly increasing");
        }
    }
    return normalized;
}

SchlamValidator::ErrorStatistics statistics(std::vector<double> errors) {
    SchlamValidator::ErrorStatistics stats;
    stats.count = errors.size();
    double sumSquares = 0.0;
    for (double error : errors) {
        stats.mean += error;
        sumSquares += error * error;
    }
    stats.mean /= stats.count;
    stats.rmse = std::sqrt(sumSquares / stats.count);
    double variance = 0.0;
    for (double error : errors) variance += (error - stats.mean) * (error - stats.mean);
    stats.standardDeviation = std::sqrt(variance / stats.count);
    std::sort(errors.begin(), errors.end());
    stats.minimum = errors.front();
    stats.maximum = errors.back();
    const auto middle = errors.size() / 2;
    stats.median = errors.size() % 2 ? errors[middle] : (errors[middle - 1] + errors[middle]) / 2.0;
    return stats;
}
} // namespace

SchlamValidator::SchlamValidator(Options options) : options_(std::move(options)) {
    if (options_.maximumTimestampDifference.count() < 0 || options_.relativePoseDelta == 0 ||
        (options_.alignment != Alignment::SIM3 && options_.alignment != Alignment::SE3 &&
         options_.alignment != Alignment::NONE)) {
        throw std::invalid_argument("Invalid trajectory evaluation options");
    }
}

SchlamValidator::Result SchlamValidator::validate(const SchlamResult& result) const {
    const auto estimated = normalizedPath(result.predictedPath, "predictedPath");
    const auto gt = normalizedPath(result.groundTruthPath, "groundTruthPath");
    if (result.processedFrames && result.predictedPath.size() > result.processedFrames) {
        throw std::invalid_argument("Estimated pose count exceeds processed frame count");
    }
    Result report;
    report.estimatedPoses = result.predictedPath.size();
    report.referencePoses = result.groundTruthPath.size();
    report.averageFrameRate = result.averageFrameRate;
    report.finalTrackingState = result.finalTrackingState;
    if (result.processedFrames) {
        report.trackingCoverage = static_cast<double>(report.estimatedPoses) / result.processedFrames;
    }

    // Chronological, nearest unmatched reference association. Each GT pose is used
    // at most once; ties prefer the earlier timestamp. Compare durations so epoch
    // timestamps do not lose nanosecond precision through conversion to doubles.
    std::vector<std::optional<std::size_t>> references(report.estimatedPoses);
    std::vector<std::size_t> matched;
    std::size_t nextReference = 0;
    for (std::size_t i = 0; i < estimated.size() && nextReference < gt.size(); ++i) {
        const auto timestamp = estimated[i].timestamp;
        const auto begin = gt.begin() + nextReference;
        auto nearest = std::lower_bound(begin, gt.end(), timestamp,
            [](const PredictedPose& pose, utils::TTimestamp time) { return pose.timestamp < time; });
        if (nearest == gt.end()) {
            --nearest;
        } else if (nearest != begin && timestamp - (nearest - 1)->timestamp <= nearest->timestamp - timestamp) {
            --nearest;
        }
        const auto difference = nearest->timestamp > timestamp ? nearest->timestamp - timestamp : timestamp - nearest->timestamp;
        if (difference > options_.maximumTimestampDifference) continue;
        references[i] = static_cast<std::size_t>(nearest - gt.begin());
        nextReference = *references[i] + 1;
        matched.push_back(i);
    }
    report.matchedPoses = matched.size();
    if (report.estimatedPoses) {
        report.associationCoverage = static_cast<double>(report.matchedPoses) / report.estimatedPoses;
    }
    const std::size_t minimumMatches = options_.alignment == Alignment::NONE ? 1 : 3;
    if (matched.size() < minimumMatches) {
        report.message = "Too few timestamp-associated poses for the requested alignment";
        return report;
    }

    if (options_.alignment != Alignment::NONE) {
        Eigen::Vector3d estimatedMean = Eigen::Vector3d::Zero();
        Eigen::Vector3d referenceMean = Eigen::Vector3d::Zero();
        for (auto i : matched) {
            estimatedMean += estimated[i].position;
            referenceMean += gt[*references[i]].position;
        }
        estimatedMean /= matched.size();
        referenceMean /= matched.size();
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        double variance = 0.0;
        for (auto i : matched) {
            const Eigen::Vector3d x = estimated[i].position - estimatedMean;
            const Eigen::Vector3d y = gt[*references[i]].position - referenceMean;
            covariance += y * x.transpose();
            variance += x.squaredNorm();
        }
        covariance /= matched.size();
        variance /= matched.size();
        if (!covariance.allFinite() || !std::isfinite(variance)) {
            throw std::invalid_argument("Trajectory positions overflow the alignment computation");
        }
        const Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const auto singular = svd.singularValues();
        // A line or stationary path cannot determine a unique global rotation.
        // Planar trajectories (rank 2) are valid.
        if (variance <= 0.0 ||
            singular[0] <= 0.0 || singular[1] <= 1e-10 * singular[0]) {
            report.status = Status::DEGENERATE_ALIGNMENT;
            report.message = "Position alignment requires non-collinear motion in both trajectories";
            return report;
        }
        Eigen::Vector3d signs = Eigen::Vector3d::Ones();
        if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.0) signs[2] = -1.0;
        report.rotation = svd.matrixU() * signs.asDiagonal() * svd.matrixV().transpose();
        if (options_.alignment == Alignment::SIM3) report.scale = singular.dot(signs) / variance;
        report.translation = referenceMean - report.scale * report.rotation * estimatedMean;
    }

    std::vector<double> absoluteErrors, relativeTranslationErrors, relativeRotationErrors;
    for (auto i : matched) {
        const Eigen::Vector3d aligned = report.scale * report.rotation * estimated[i].position + report.translation;
        absoluteErrors.push_back((gt[*references[i]].position - aligned).norm());
        if (options_.relativePoseDelta >= estimated.size() - i) continue;
        const auto j = i + options_.relativePoseDelta;
        if (!references[j]) continue;
        const auto& a = estimated[i];
        const auto& b = estimated[j];
        const auto& p = gt[*references[i]];
        const auto& q = gt[*references[j]];
        // E_ij = (T_gt_i^-1 T_gt_j)^-1 (T_est_i^-1 T_est_j).
        // Global rotation/translation cancel. Apply the fitted scale exactly once.
        const Eigen::Vector3d estimatedDelta = report.scale * a.rotation.transpose() * (b.position - a.position);
        const Eigen::Vector3d referenceDelta = p.rotation.transpose() * (q.position - p.position);
        relativeTranslationErrors.push_back((estimatedDelta - referenceDelta).norm());
        const Eigen::Matrix3d rotationError = (p.rotation.transpose() * q.rotation).transpose() *
            (a.rotation.transpose() * b.rotation);
        relativeRotationErrors.push_back(Eigen::AngleAxisd(rotationError).angle());
    }
    report.absoluteTranslation = statistics(std::move(absoluteErrors));
    if (!relativeTranslationErrors.empty()) {
        report.relativeTranslation = statistics(std::move(relativeTranslationErrors));
        report.relativeRotationRadians = statistics(std::move(relativeRotationErrors));
    }
    report.status = Status::SUCCESS;
    report.message = "Trajectory evaluation succeeded";
    return report;
}
