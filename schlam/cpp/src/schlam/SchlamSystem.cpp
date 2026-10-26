//
// Created by jonathan_michel on 10/25/26.
//

#include "SchlamSystem.h"

#include <thread>

#include "Map.h"
#include "Matcher.h"
#include "ORBFeatureDetector.h"
#include "Profiler.h"
#include "Ransac.h"
#include "utils.h"
#include "data/DatasetInterface.hpp"
#include "plotting/IPlotter.hpp"

SchlamSystem::SchlamSystem(
        std::shared_ptr<DatasetInterface> dataset,
        std::shared_ptr<tft::Transformer> transformer,
        std::shared_ptr<IPlotter> plotter,
        std::shared_ptr<ORBFeatureDetector> featureDetector,
        std::shared_ptr<Optimizer> optimizer,
        MotionModel& motionModel)
    : dataset_(dataset)
    , transformer_(transformer)
    , plotter_(plotter)
    , featureDetector_(featureDetector)
    , optimizer_(optimizer)
    , motionModel_(motionModel)
    , localMappingQueue_(1) {

    cameraToIMU_ = transformer_->findTransform("gt/cam0", "gt/body");
}

void SchlamSystem::run() {

    // initialization
    while (!dataset_->empty()) {
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
                return;

        }

        std::cout << "Frame handling took " << frameTimer.finish() << " ms" << std::endl;
        while (plotter_->shouldPause()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    std::cout << "Finished sequence, doing full map optimization pass..." << std::endl;
    map_->cullMapPoints();
    optimizer_->optimizeFullMap(map_.get(), true);
    std::cout << "Finished!" << std::endl;
}

SynchronizedData SchlamSystem::getNextSynchronizedData() {
    auto imageData = dataset_->getNextImageData();
    auto imuData = dataset_->getNextIMUData();
    auto gtData = dataset_->getNextGTData();

    auto imuBegin = imuData ? imuData->mTimestamp.time_since_epoch() : std::chrono::nanoseconds(0);
    auto gtBegin = gtData ? gtData->mTimestamp.time_since_epoch() : std::chrono::nanoseconds(0);
    auto highestTimestamp = std::max(
        imageData->mTimestamp.time_since_epoch(),
        std::max(imuBegin, gtBegin)
    );

    while (imageData && imageData->mTimestamp.time_since_epoch() < highestTimestamp) {
        imageData = dataset_->getNextImageData();
    }

    while (imuData && imuData->mTimestamp.time_since_epoch() < highestTimestamp) {
        imuData = dataset_->getNextIMUData();
    }

    while (gtData && gtData->mTimestamp.time_since_epoch() < highestTimestamp) {
        gtData = dataset_->getNextGTData();
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
    auto frame = std::make_shared<Frame>(data.imageData);

    featureDetector_->detectFeatures(*frame);

    if (!referenceFrame_) {
        setFrameAsReference(frame, data.cameraToWorld);
        return;
    }

    plotter_->addGroundTruthCameraPosition(data.cameraToWorld->mTranslation);

    auto oldFrame = referenceFrame_.get();
    auto [matches, _] = matchWindow(oldFrame->mKeypointTree,
                                 frame->mKeypointTree, 20, 100);
    plotter_->plotMatches(oldFrame->mImage, frame->mImage, matches);

    if (matches[0].size() < 50) {
        setFrameAsReference(frame, data.cameraToWorld);
        std::cout << "Selecting new reference" << std::endl;
        return;
    }

    auto reconstructOpt = reconstructInitial(matches[0], matches[1], data.imageData->mIntrinsics);

    if (reconstructOpt.has_value()) {
        auto [R, t, pts, inliers] = reconstructOpt.value();
        auto filtered_pts = filterByInlierMask(pts, inliers);
        auto pts2D_1 = filterByInlierMask(matches[0], inliers);
        auto pts2D_2 = filterByInlierMask(matches[1], inliers);

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
        plotter_->setMap(map_.get());

        state_ = TrackingState::TRACKING;
        localMappingThread_ = std::thread(&SchlamSystem::localMapping, this);
    }
}

void SchlamSystem::track(const SynchronizedData& data) {
     // TODO Try if polynomic estimator works better
    auto estimatedPose = motionModel_.estimateLinearAt(data.imageData->mTimestamp);
    plotter_->updateFrustum(data.imageData, estimatedPose);
    plotter_->addGroundTruthCameraPosition(data.cameraToWorld->mTranslation);

    transformer_->registerTransform(estimatedPose);

    auto frame = std::make_shared<Frame>(data.imageData);
    featureDetector_->detectFeatures(*frame);

    // TODO Match only with features in predicted scale
    // TODO: Use map TrackingSnapshot instead of direct map access
    // Content:
    // - MapPoint Snapshots
    // -
    TrackingSnapshot trackingSnapshot = map_->getTrackingSnapshot();
    const auto matches = matchMapPointsByProjection(trackingSnapshot, frame, estimatedPose, transformer_, 50, 50);
    std::cout << "Matched " << matches.size() << " out of " <<trackingSnapshot.mapPoints.size() << " map points" << std::endl;
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
    std::cout << "Retained " << inlierCount << " pose inliers" << std::endl;
    if (inlierCount >= 12) {
        frame->mMapToCamera = std::make_shared<tft::RigidTransform3D>(
            trackingSnapshot.frame, estimatedPose->mSource, rotation, translation);
        recordMapPointVisibility(trackingSnapshot, *frame, *frame->mMapToCamera);
        for (std::size_t j = 0; j < matches.size(); ++j) {
            if (inliers[j]) {
                trackingSnapshot.mapPoints.at(matches[j].mapPointId).addInlierObservation(
                    matches[j].feature->getDescriptor(), acceptedCameraCenter);
            }
        }
        const auto mapToWorld = transformer_->findTransform(trackingSnapshot.frame, "world");

        map_->applyTrackingSnapshot(trackingSnapshot);

        if (map_->decideNewKeyframe(inlierCount, framesSinceLastKeyframe_)) {
            framesSinceLastKeyframe_ = 0;
            const KeyFrameId frameId = addKeyFrame(frame, matches, inliers);
            if (!localMappingIdle_) {
                optimizer_->stopCovisibility();
            }
            localMappingQueue_.push_back(frameId);
        }

        const auto cameratoworld = mapToWorld * frame->mMapToCamera->inverse();
        transformer_->registerTransform(cameratoworld);
        plotter_->updateFrustum(data.imageData, cameratoworld);
        motionModel_.addPose(cameratoworld, data.imageData->mTimestamp);
        plotter_->addPredictedCameraPosition(cameratoworld->mTranslation);
    }

    framesSinceLastKeyframe_++;
    if (matchedMapPoints.size() < 12) {
        state_ = TrackingState::LOST;
    }
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
    map_ = std::make_unique<Map>(prevFrameName);
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
    std::size_t lastNKeyframes = 5;
    auto start = std::prev(map_->mKeyFrames.end(), std::min(lastNKeyframes, map_->mKeyFrames.size() - 1));
    auto latest = std::prev(map_->mKeyFrames.end());

    for (auto it = start; it != latest; ++it) {
        auto matches = matchUnmatchedFeatures(it->second, latest->second, 30);
        auto transform = latest->second.mMapToCamera * it->second.mMapToCamera->inverse();
        auto optionalResult = triangulate(matches[0], matches[1], latest->second.mIntrinsics, transform);
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

const KeyFrameId SchlamSystem::addKeyFrame(
    std::shared_ptr<Frame> frame,
    const std::vector<MapPointMatch> matches,
    const std::vector<bool> inliers) {
    const auto frameId = map_->addKeyFrame(frame);
    for (std::size_t j = 0; j < matches.size(); ++j) {
        if (!inliers[j]) continue;

        const auto featureIndex = matches[j].feature->getIndex().value();
        map_->addObservation(frameId, featureIndex, matches[j].mapPointId);
    }
    return frameId;
}

void SchlamSystem::localMapping() {
    while (!stopLocalMapping_) {
        localMappingIdle_ = true;
        auto frameId = localMappingQueue_.pull_front();
        localMappingIdle_ = false;

        auto frame = map_->mKeyFrames.at(frameId);
        map_->cullMapPoints();

        // add map points
        triangulateUnmatchedFeatures();

        // optimize
        if (map_->mKeyFrames.size() < 30) {
            optimizer_->optimizeFullMap(map_.get());
        } else {
            optimizer_->optimizeCovisibilityGraph(map_.get(), std::prev(map_->mKeyFrames.end())->second);
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
