//
// Created by jonathan_michel on 10/25/26.
//

#include "SchlamSystem.h"
#include "optimization/Optimizer.h"
#include "MotionModel.h"

#include <thread>
#include <spdlog/common.h>
#include <tracy/Tracy.hpp>

#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.hpp"
#include "Matcher.h"
#include "feature_detection/orb/ORBFeatureDetector.h"
#include "helpers/Profiler.h"
#include "Ransac.h"
#include "helpers/utils.h"
#include "data/DatasetInterface.hpp"
#include "tft/transformer.hpp"
#include "plotting/IPlotter.hpp"

SchlamSystem::SchlamSystem(
        std::shared_ptr<DatasetInterface> dataset,
        std::shared_ptr<tft::Transformer> transformer,
        std::shared_ptr<IPlotter> plotter,
        std::shared_ptr<ORBFeatureDetector> featureDetector,
        std::shared_ptr<Optimizer> optimizer,
        MotionModel& motionModel,
        schlam::config::Config settings)
    : dataset_(dataset)
    , transformer_(transformer)
    , plotter_(plotter)
    , featureDetector_(featureDetector)
    , optimizer_(optimizer)
    , motionModel_(motionModel)
    , settings_(std::move(settings))
    , reconstructor_(settings_.reconstruction)
    , localMappingQueue_(settings_.mapping.queueCapacity) {

    cameraToIMU_ = transformer_->findTransform("gt/cam0", "gt/body");
}

SchlamSystem::~SchlamSystem() {
    finishLocalMapping();
}

SchlamResult SchlamSystem::run() {
    tracy::SetThreadName("Main");
    std::size_t processedFrames = 0;
    double processingMilliseconds = 0.0;

    // initialization
    while (state_ != TrackingState::LOST && !dataset_->empty()) {
        const auto frameStart = std::chrono::steady_clock::now();

        auto data = getNextSynchronizedData();

        Profiler frameTimer(*plotter_, frameStart);

        plotter_->updateFrustum(data.imageData);

        switch (state_) {
            case TrackingState::INITIALIZING:
                initialize(data);
                break;
            case TrackingState::TRACKING:
                track(data);
                break;
            case TrackingState::LOST:
                break;

        }

        FrameMarkNamed("Frame");
        const double frameMilliseconds = frameTimer.finish();
        processingMilliseconds += frameMilliseconds;
        ++processedFrames;
        std::cout << "Frame handling took " << frameMilliseconds << " ms" << std::endl;
        while (plotter_->shouldPause()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    finishLocalMapping();

    SchlamResult result = buildResult(processedFrames, processingMilliseconds);
    return result;
}

SynchronizedData SchlamSystem::getNextSynchronizedData() {
    const auto recordGroundTruth = [this](const std::shared_ptr<GTData>& gt) {
        if (!gt) return;
        const auto bodyToWorld = std::make_shared<tft::RigidTransform3D>(
            "gt/body", "world", gt->mRotation, gt->mPosition);
        const auto cameraToWorld = bodyToWorld * cameraToIMU_;
        groundTruthPath_.push_back({gt->mTimestamp,
            cameraToWorld->mRotation, cameraToWorld->mTranslation});
    };
    auto imageData = dataset_->getNextImageData();
    auto imuData = dataset_->getNextIMUData();
    auto gtData = dataset_->getNextGTData();
    recordGroundTruth(gtData);

    auto imuBegin = imuData ? imuData->mTimestamp.time_since_epoch() : std::chrono::nanoseconds(0);
    auto gtBegin = gtData ? gtData->mTimestamp.time_since_epoch() : std::chrono::nanoseconds(0);
    auto highestTimestamp = std::max(
        imageData->mTimestamp.time_since_epoch(),
        std::max(imuBegin, gtBegin)
    );

    while (imageData && imageData->mTimestamp.time_since_epoch() < highestTimestamp && !dataset_->empty()) {
        imageData = dataset_->getNextImageData();
    }

    while (imuData && imuData->mTimestamp.time_since_epoch() < highestTimestamp && !dataset_->empty()) {
        imuData = dataset_->getNextIMUData();
    }

    while (gtData && gtData->mTimestamp.time_since_epoch() < highestTimestamp && !dataset_->empty()) {
        gtData = dataset_->getNextGTData();
        recordGroundTruth(gtData);
    }

    plotter_->updateTimestamp(imageData->mTimestamp);

    std::shared_ptr<tft::RigidTransform3D> cameraToWorld = nullptr;
    if (gtData) {
        const auto imuToWorld = std::make_shared<tft::RigidTransform3D>(
            "gt/body", "world", gtData->mRotation, gtData->mPosition);
        cameraToWorld = imuToWorld * cameraToIMU_;
        transformer_->registerTransform(imuToWorld);
        transformer_->registerTransform(cameraToWorld);
    }
    return {imageData, imuData, gtData, cameraToWorld};
}

void SchlamSystem::initialize(const SynchronizedData& data) {
    ZoneScoped;
    auto frame = std::make_shared<Frame>(data.imageData);

    featureDetector_->extractFeatures(*frame);

    if (!referenceFrame_) {
        setFrameAsReference(frame, data.cameraToWorld);
        return;
    }

    plotter_->addGroundTruthCameraPosition(data.cameraToWorld->mTranslation);

    auto oldFrame = referenceFrame_.get();
    auto [matches, _] = matchWindow(oldFrame->mKeypointTree,
                                 frame->mKeypointTree,
                                 settings_.initialization.matchHammingDistance,
                                 settings_.initialization.matchWindowPixels);
    plotter_->plotMatches(oldFrame->mImage, frame->mImage, matches);
    TracyPlot("Matches", static_cast<int64_t>(matches.size()));

    if (matches[0].size() < settings_.initialization.minimumMatches) {
        setFrameAsReference(frame, data.cameraToWorld);
        TracyPlot("New reference", data.imageData->mTimestamp.time_since_epoch().count());
        std::cout << "Selecting new reference" << std::endl;
        return;
    }

    auto reconstructOpt = reconstructor_.reconstructInitial(matches[0], matches[1], data.imageData->mIntrinsics);

    if (reconstructOpt.has_value()) {
        auto [R, t, pts, inliers] = reconstructOpt.value();
        auto filtered_pts = filterByInlierMask(pts, inliers);
        auto pts2D_1 = filterByInlierMask(matches[0], inliers);
        auto pts2D_2 = filterByInlierMask(matches[1], inliers);
        TracyPlot("Inliers", static_cast<int64_t>(filtered_pts.size()));

        auto [R_opt, t_opt, pts_opt] = optimizer_->optimizePoseAndPoints(
{pts2D_1, pts2D_2}, R, t, filtered_pts, data.imageData->mIntrinsics
        );

        // Use the GT camera baseline to put translation and points in metres.
        if (scaleToBaseline(referenceGTCameraPose_, data.cameraToWorld, t_opt, pts_opt)) return;

        const std::string frameName = "pred/cam0";
        const auto referenceToNew = std::make_shared<tft::RigidTransform3D>(
            referenceFrameName_, frameName, R_opt, t_opt);
        referenceFrame_->mMapToCamera = tft::RigidTransform3D::identity(referenceFrameName_, referenceFrameName_);
        frame->mMapToCamera = referenceToNew;
        transformer_->registerTransform(referenceCameraToWorld_);
        const auto newCameraToWorld = referenceCameraToWorld_ * referenceToNew->inverse();
        transformer_->registerTransform(newCameraToWorld);
        plotter_->addPredictedCameraPosition(newCameraToWorld->mTranslation);
        plotter_->updateFrustum(data.imageData, newCameraToWorld);

        motionModel_.addPose(transformer_->findTransform(referenceFrameName_, "world"), referenceFrame_->mTimestamp);
        motionModel_.addPose(transformer_->findTransform(frameName, "world"), frame->mTimestamp);

        buildMap(referenceFrameName_, referenceFrame_, frame, pts2D_1, pts2D_2, pts_opt);
        predictedPath_.push_back({referenceFrame_->mTimestamp,
            referenceCameraToWorld_->mRotation, referenceCameraToWorld_->mTranslation});
        predictedPath_.push_back({frame->mTimestamp,
            newCameraToWorld->mRotation, newCameraToWorld->mTranslation});
        plotter_->setMap(map_.get());

        state_ = TrackingState::TRACKING;
        localMappingThread_ = std::thread(&SchlamSystem::localMapping, this);
    }
}

void SchlamSystem::track(const SynchronizedData& data) {
    ZoneScoped;
     // TODO Try if polynomic estimator works better
    auto estimatedPose = motionModel_.estimateLinearAt(data.imageData->mTimestamp);
    plotter_->updateFrustum(data.imageData, estimatedPose);
    plotter_->addGroundTruthCameraPosition(data.cameraToWorld->mTranslation);

    transformer_->registerTransform(estimatedPose);

    auto frame = std::make_shared<Frame>(data.imageData);
    featureDetector_->extractFeatures(*frame);

    // TODO Match only with features in predicted scale
    TrackingSnapshot trackingSnapshot = map_->getTrackingSnapshot();
    TracyPlot("MapPoints", static_cast<int64_t>(trackingSnapshot.mapPoints.size()));
    const auto matches = matchMapPointsByProjection(trackingSnapshot, frame, estimatedPose, transformer_,
        settings_.tracking.matchHammingDistance, settings_.tracking.projectionWindowPixels);
    TracyPlot("Matches", static_cast<int64_t>(matches.size()));
    std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matchedFeatures;
    std::vector<Eigen::Vector3d> matchedMapPoints;
    matchedMapPoints.reserve(matches.size());
    matchedFeatures[0].reserve(matches.size());
    matchedFeatures[1].reserve(matches.size());
    for (const auto& match : matches) {
        matchedMapPoints.push_back(trackingSnapshot.mapPoints.at(match.mapPointId).mPosition);
        matchedFeatures[0].push_back(match.projectedFeature);
        matchedFeatures[1].push_back(match.feature);
    }
    plotter_->plotMatches(frame->mImage, frame->mImage, matchedFeatures);

    const auto mapToCamera = transformer_->findTransform(trackingSnapshot.frame, estimatedPose->mSource);
    auto [rotation, translation, inliers] = optimizer_->optimizePoseWithInliers(matchedFeatures[1],
        mapToCamera->mRotation, mapToCamera->mTranslation,
        matchedMapPoints, data.imageData->mIntrinsics);
    const Eigen::Vector3d acceptedCameraCenter = -rotation.transpose() * translation;
    for (std::size_t j = 0; j < matches.size(); ++j) {
        if (inliers[j] && !trackingSnapshot.mapPoints.at(matches[j].mapPointId).isViewDirectionCompatible(acceptedCameraCenter))
            inliers[j] = false;
    }
    const auto inlierCount = std::count(inliers.begin(), inliers.end(), true);
    TracyPlot("Inliers", inlierCount);
    if (inlierCount >= settings_.tracking.minimumPoseInliers) {
        frame->mMapToCamera = std::make_shared<tft::RigidTransform3D>(
            trackingSnapshot.frame, estimatedPose->mSource, rotation, translation);
        recordMapPointVisibility(trackingSnapshot, *frame, *frame->mMapToCamera);
        for (std::size_t j = 0; j < matches.size(); ++j) {
            if (inliers[j]) {
                trackingSnapshot.newObservations.insert_or_assign(matches[j].mapPointId, Observation{matches[j].feature->getDescriptor(), acceptedCameraCenter});
            }
        }
        const auto mapToWorld = transformer_->findTransform(trackingSnapshot.frame, "world");

        map_->applyTrackingSnapshot(trackingSnapshot);

        if (map_->decideNewKeyframe(inlierCount, framesSinceLastKeyframe_, localMappingIdle_)) {
            framesSinceLastKeyframe_ = 0;
            const KeyFrameId frameId = map_->addKeyFrame(frame, matches, inliers);
            if (!localMappingIdle_) {
                optimizer_->stopCovisibility();
            }
            localMappingQueue_.push_back(frameId);
        }

        const auto cameratoworld = mapToWorld * frame->mMapToCamera->inverse();
        transformer_->registerTransform(cameratoworld);
        plotter_->updateFrustum(data.imageData, cameratoworld);
        motionModel_.addPose(cameratoworld, data.imageData->mTimestamp);
        predictedPath_.push_back({data.imageData->mTimestamp,
            cameratoworld->mRotation, cameratoworld->mTranslation});
        plotter_->addPredictedCameraPosition(cameratoworld->mTranslation);
    }

    framesSinceLastKeyframe_++;
    if (matchedMapPoints.size() < settings_.tracking.minimumMatchesBeforeLost) {
        std::cout << "Lost tracking!" << std::endl;
        state_ = TrackingState::LOST;
    }
}

void SchlamSystem::localMapping() {
    tracy::SetThreadName("Local Mapping");
    while (!stopLocalMapping_ || !localMappingQueue_.empty()) {
        localMappingIdle_ = true;

        if (localMappingQueue_.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        auto frameId = localMappingQueue_.pull_front();
        localMappingIdle_ = false;
        ZoneScoped;

        auto frame = map_->mKeyFrames.at(frameId);
        map_->cullMapPoints();

        triangulateUnmatchedFeatures();

        optimizer_->optimizeCovisibilityGraph(map_.get(), std::prev(map_->mKeyFrames.end())->second,
            settings_.optimizer.localMaxVariableKeyframes, settings_.optimizer.localMaxFixedKeyframes);

    }
    std::cout << "Local mapping finished!" << std::endl;
}

bool SchlamSystem::scaleToBaseline(std::shared_ptr<tft::RigidTransform3D> referenceGTCameraPose,
                     std::shared_ptr<tft::RigidTransform3D> cameraToWorld,
                     Eigen::Vector3d& t_opt,
                     std::vector<Eigen::Matrix<double, 3, 1>>& pts_opt) {
    const double gtBaseline = (cameraToWorld->mTranslation -
                               referenceGTCameraPose->mTranslation).norm();
    const double estimatedBaseline = t_opt.norm();
    if (!std::isfinite(gtBaseline) || gtBaseline <= 0 ||
        !std::isfinite(estimatedBaseline) || estimatedBaseline <= 0) {
        return true;
    }
    const double scale = gtBaseline / estimatedBaseline;
    t_opt *= scale;
    for (auto& point : pts_opt) {
        point *= scale;
    }
    return false;
}

void SchlamSystem::buildMap(std::string prevFrameName,
              std::shared_ptr<Frame> referenceFrame,
              std::shared_ptr<Frame> frame,
              std::vector<std::shared_ptr<KeyPoint>> pts2D_1,
              std::vector<std::shared_ptr<KeyPoint>> pts2D_2, std::vector<Eigen::Vector3d> pts_opt) {
    map_ = std::make_unique<Map>(prevFrameName, settings_.mapping, settings_.mapPoints);
    const auto referenceKeyFrameId = map_->addKeyFrame(referenceFrame);
    const auto initialKeyFrameId = map_->addKeyFrame(frame);
    const auto pointIds = map_->addMapPoints(pts_opt, pts2D_2, initialKeyFrameId);
    for (std::size_t pointIndex = 0; pointIndex < pointIds.size(); ++pointIndex) {
        const auto referenceFeatureIndex = pts2D_1[pointIndex]->getIndex().value();
        const auto initialFeatureIndex = pts2D_2[pointIndex]->getIndex().value();
        map_->addObservation(referenceKeyFrameId, referenceFeatureIndex, pointIds[pointIndex]);
        map_->addObservation(initialKeyFrameId, initialFeatureIndex, pointIds[pointIndex]);
    }
}

void SchlamSystem::triangulateUnmatchedFeatures() {
    // TODO use covisibility graph instead?
    std::size_t lastNKeyframes = settings_.mapping.triangulationPreviousKeyframes;
    auto start = std::prev(map_->mKeyFrames.end(), std::min(lastNKeyframes, map_->mKeyFrames.size() - 1));
    auto latest = std::prev(map_->mKeyFrames.end());

    for (auto it = start; it != latest; ++it) {
        auto matches = matchUnmatchedFeatures(it->second, latest->second,
            settings_.mapping.triangulationMatchHammingDistance);
        auto transform = latest->second.mMapToCamera * it->second.mMapToCamera->inverse();
        auto optionalResult = reconstructor_.triangulate(matches[0], matches[1], latest->second.mIntrinsics, transform);
        if (optionalResult.has_value()) {
            // add new MapPoints and observations
            auto [pts, inliers] = optionalResult.value();
            std::size_t numInliers{0};
            std::vector<MapPointId> newMapPointIds;
            for (std::size_t i = 0; i < pts.size(); i++) {
                if (inliers[i]) {
                    numInliers++;
                    const auto pointInMap = it->second.mMapToCamera->inverse()->apply(pts[i]);
                    const auto pointId = map_->addMapPoint(pointInMap, matches[0][i], latest->second.mId);
                    map_->mPoints.at(pointId).addTriangulationObservation(matches[1][i]->getDescriptor());
                    newMapPointIds.push_back(pointId);
                    map_->addObservation(it->second.mId, matches[0][i]->getIndex().value(), pointId);
                    map_->addObservation(latest->second.mId, matches[1][i]->getIndex().value(), pointId);
                }
            }
            std::cout << "Triangulated " << numInliers << " new MapPoints!" << std::endl;
            // TODO look for matches in the other keyframes
        }
    }
}

void SchlamSystem::setFrameAsReference(std::shared_ptr<Frame> frame,
                                       const std::shared_ptr<tft::RigidTransform3D>& cameraToWorld) {
    referenceCameraToWorld_ = std::make_shared<tft::RigidTransform3D>(
        referenceFrameName_, "world",
        cameraToWorld->mRotation, cameraToWorld->mTranslation);
    referenceGTCameraPose_ = cameraToWorld;
    referenceFrame_ = std::move(frame);
    plotter_->resetCameraPaths(referenceGTCameraPose_->mTranslation);
};

SchlamResult SchlamSystem::buildResult(std::size_t processedFrames, double processingMilliseconds) const {
    SchlamResult result;
    if (map_) {
        const auto snapshot = map_->getSnapshot();
        const auto mapToWorld = transformer_->findTransform(map_->mFrame, "world");
        result.mapPoints.reserve(snapshot.points.size());
        for (const auto& point : snapshot.points) {
            result.mapPoints.push_back(mapToWorld->apply(point));
        }
    }
    result.predictedPath = predictedPath_;
    result.groundTruthPath = groundTruthPath_;
    result.processedFrames = processedFrames;
    result.finalTrackingState = state_;
    if (processingMilliseconds > 0.0) {
        result.averageFrameRate = 1000.0 * processedFrames / processingMilliseconds;
    }
    return result;
}

void SchlamSystem::finishLocalMapping() {
    stopLocalMapping_ = true;
    if (localMappingThread_.joinable()) localMappingThread_.join();
}
