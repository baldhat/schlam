#include "data/datasets/mav_dataloader.hpp"
#include "plotting/plotter.hpp"
#include "schlam/Ransac.h"
#include "tft/rigid_transform_3d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>

#include "schlam/ORBFeatureDetector.h"
#include "schlam/Matcher.h"
#include "schlam/Optimizer.h"
#include "schlam/Frame.h"
#include "schlam/Map.h"
#include "schlam/MotionModel.h"


bool getNextData(std::shared_ptr<tft::Transformer> pTransformer,
                 std::shared_ptr<MAVDataloader> dataloader,
                 std::shared_ptr<Plotter> plotter,
                 const std::shared_ptr<tft::RigidTransform3D> cameraToIMU,
                 std::shared_ptr<std::pair<IMUData, GTData>> &imuAndGT,
                 std::shared_ptr<ImageData> &imageData,
                 std::shared_ptr<tft::RigidTransform3D> &cameraToWorld,
                 bool &continues) {
    continues = false;
    imageData = dataloader->getNextImageData();
    plotter->updateTimestamp(imageData->mTimestamp);
    // The GT stream starts later than the images and is not timestamp-aligned
    // with the IMU rows returned alongside it.
    if (imageData->mTimestamp < imuAndGT->second.mTimestamp) {
        continues = true;
        return false;
    }
    while (imuAndGT->second.mTimestamp < imageData->mTimestamp) {
        auto nextIMUAndGT = dataloader->getNextIMUData();
        if (nextIMUAndGT->second.mTimestamp <= imuAndGT->second.mTimestamp) {
            break; // No more GT samples are available.
        }
        imuAndGT = std::move(nextIMUAndGT);
    }
    if (imuAndGT->second.mTimestamp < imageData->mTimestamp) {
        return true;
    }
    const auto& gtData = imuAndGT->second;
    const auto imuToWorld = std::make_shared<tft::RigidTransform3D>(
        "imu", "world", gtData.mRotation, gtData.mPosition);
    cameraToWorld = imuToWorld * cameraToIMU;
    pTransformer->registerTransform(imuToWorld);
    // Register the composed pose directly so cam0s cached world pose is refreshed.
    pTransformer->registerTransform(cameraToWorld);

    pTransformer->findTransform("imu", "world");
    pTransformer->findTransform("cam0", "world");
    return false;
}

bool scaleToBaseline(std::shared_ptr<tft::RigidTransform3D> referenceGTCameraPose,
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

std::unique_ptr<Map> buildMap(std::string prevFrameName,
              std::shared_ptr<Frame> referenceFrame,
              std::shared_ptr<Frame> frame,
              std::vector<std::shared_ptr<KeyPoint>> pts2D_1,
              std::vector<std::shared_ptr<KeyPoint>> pts2D_2, std::vector<Eigen::Vector3d> pts_opt) {
    auto map = std::make_unique<Map>(prevFrameName);
    const auto referenceKeyFrameId = map->addKeyFrame(referenceFrame);
    const auto initialKeyFrameId = map->addKeyFrame(frame);
    const auto pointIds = map->addMapPoints(pts_opt, pts2D_2, initialKeyFrameId);
    for (std::size_t pointIndex = 0; pointIndex < pointIds.size(); ++pointIndex) {
        const auto referenceFeatureIndex = pts2D_1[pointIndex]->getIndex().value();
        const auto initialFeatureIndex = pts2D_2[pointIndex]->getIndex().value();
        map->addObservation(referenceKeyFrameId, referenceFeatureIndex, pointIds[pointIndex]);
        map->addObservation(initialKeyFrameId, initialFeatureIndex, pointIds[pointIndex]);
    }
    return std::move(map);
}

void triangulateUnmatchedFeatures(Map* map) {
    std::size_t lastNKeyframes = 5;
    auto start = std::prev(map->mKeyFrames.end(), std::min(lastNKeyframes, map->mKeyFrames.size() - 1));
    auto latest = std::prev(map->mKeyFrames.end());

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
                    const auto pointId = map->addMapPoint(pointInMap, matches[0][i], latest->second.mId);
                    map->mPoints.at(pointId).addTriangulationObservation(matches[1][i]->getDescriptor());
                    newMapPointIds.push_back(pointId);
                    map->addObservation(it->second.mId, matches[0][i]->getIndex().value(), pointId);
                    map->addObservation(latest->second.mId, matches[1][i]->getIndex().value(), pointId);
                }
            }
            std::cout << "Triangulated " << numInliers << " new MapPoints!" << std::endl;

            // TODO look for matches in the other keyframes

        }
    }
}

void handleNewKeyframe(std::shared_ptr<Optimizer> optimizer,
                       MotionModel &motionModel,
                       Map* map,
                       std::shared_ptr<Frame> frame,
                       const std::vector<MapPointMatch> matches,
                       const std::vector<bool> inliers) {
    // add keyframe
    const auto frameId = map->addKeyFrame(frame);
    for (std::size_t j = 0; j < matches.size(); ++j) {
        if (!inliers[j]) continue;

        const auto featureIndex = matches[j].feature->getIndex().value();
        map->addObservation(frameId, featureIndex, matches[j].mapPointId);
    }

    map->cullMapPoints();

    // add map points
    triangulateUnmatchedFeatures(map);

    // optimize
    if (map->mKeyFrames.size() < 30) {
        if (optimizer->optimizeFullMap(map)) {
            frame->mMapToCamera = map->mKeyFrames.at(frameId).mMapToCamera;
            motionModel = MotionModel{};
        }
    } else {
        if (optimizer->optimizeCovisibilityGraph(map, std::prev(map->mKeyFrames.end())->second)) {
            frame->mMapToCamera = map->mKeyFrames.at(frameId).mMapToCamera;
            motionModel = MotionModel{};
        }
    }


}




class FrameTimer {
public:
    FrameTimer(Plotter& plotter, std::chrono::steady_clock::time_point start)
        : mPlotter(plotter), mStart(start) {}

    ~FrameTimer() { finish(); }

    double finish() {
        if (!mFinished) {
            const auto elapsed = std::chrono::steady_clock::now() - mStart;
            mElapsedMs = std::chrono::duration<double, std::milli>(elapsed).count();
            mPlotter.recordFrameTime(elapsed);
            mFinished = true;
        }
        return mElapsedMs;
    }

    FrameTimer(const FrameTimer&) = delete;
    FrameTimer& operator=(const FrameTimer&) = delete;

private:
    Plotter& mPlotter;
    std::chrono::steady_clock::time_point mStart;
    double mElapsedMs{0.0};
    bool mFinished{false};
};

int main() {
    auto pTransformer = std::make_shared<tft::Transformer>();

    auto dataloader = std::make_shared<MAVDataloader>(pTransformer);

    auto plotter = std::make_shared<Plotter>(pTransformer, dataloader->getCameraIntrinsics());
    auto render_loop = std::thread([plotter] { plotter->run(); });

    auto initFeatureDetector = std::make_shared<ORBFeatureDetector>(1000, plotter, 6);
    auto featureDetector = std::make_shared<ORBFeatureDetector>(1000, plotter, 6);

    auto optimizer = std::make_shared<Optimizer>();

    auto motionModel = MotionModel();

    const auto cameraToIMU = pTransformer->findTransform("cam0", "imu");
    auto imuAndGT = dataloader->getNextIMUData();

    int i = 0;
    std::string prevFrameName;
    std::shared_ptr<tft::RigidTransform3D> referenceGTCameraPose;
    std::shared_ptr<tft::RigidTransform3D> referenceCameraToWorld;

    std::unique_ptr<Map> map;

    std::shared_ptr<Frame> referenceFrame;
    const auto selectReference = [&](std::shared_ptr<Frame> frame,
                                     const std::shared_ptr<tft::RigidTransform3D>& cameraToWorld) {
        prevFrameName = "cam_ref";
        referenceCameraToWorld = std::make_shared<tft::RigidTransform3D>(
            prevFrameName, "world", cameraToWorld->mRotation,
            cameraToWorld->mTranslation);
        referenceGTCameraPose = cameraToWorld;
        referenceFrame = std::move(frame);
        plotter->resetCameraPaths(referenceGTCameraPose->mTranslation);
    };

    // initialization
    while (!dataloader->empty()) {
        const auto frameStart = std::chrono::steady_clock::now();
        std::shared_ptr<ImageData> imageData;
        std::shared_ptr<tft::RigidTransform3D> cameraToWorld;
        bool continues;
        if (getNextData(pTransformer, dataloader, plotter, cameraToIMU, imuAndGT, imageData, cameraToWorld, continues)) break;
        if (continues) continue;
        FrameTimer frameTimer(*plotter, frameStart);

        plotter->updateFrustum(imageData);

        auto frame = std::make_shared<Frame>(imageData);

        initFeatureDetector->detectFeatures(*frame);

        if (!referenceFrame) {
            selectReference(frame, cameraToWorld);
            continue;
        }

        plotter->addGroundTruthCameraPosition(cameraToWorld->mTranslation);

        auto oldFrame = referenceFrame.get();
        auto [matches, _] = matchWindow(oldFrame->mKeypointTree,
                                     frame->mKeypointTree, 20, 100);
        plotter->plotMatches(oldFrame->mImage, frame->mImage, matches);

        if (matches[0].size() < 50) {
            selectReference(frame, cameraToWorld);
            std::cout << "Selecting new reference" << std::endl;
            continue;
        }

        auto reconstructOpt = reconstructInitial(matches[0], matches[1], imageData->mIntrinsics);

        if (reconstructOpt.has_value()) {
            auto [R, t, pts, inliers] = reconstructOpt.value();
            auto filtered_pts = filterByInlierMask(pts, inliers);
            auto pts2D_1 = filterByInlierMask(matches[0], inliers);
            auto pts2D_2 = filterByInlierMask(matches[1], inliers);

            auto [R_opt, t_opt, pts_opt] = optimizer->optimizePoseAndPoints({pts2D_1, pts2D_2}, R, t, filtered_pts, imageData->mIntrinsics);

            // Use the GT camera baseline to put translation and points in metres.
            if (scaleToBaseline(referenceGTCameraPose, cameraToWorld, t_opt, pts_opt)) continue;

            const std::string frameName = "cam_pred" + std::to_string(i++);
            const auto referenceToNew = std::make_shared<tft::RigidTransform3D>(
                prevFrameName, frameName, R_opt, t_opt);
            referenceFrame->mMapToCamera = tft::RigidTransform3D::identity(prevFrameName, prevFrameName);
            frame->mMapToCamera = referenceToNew;
            pTransformer->registerTransform(referenceCameraToWorld);
            pTransformer->findTransform(prevFrameName, "world");
            const auto newCameraToWorld = referenceCameraToWorld * referenceToNew->inverse();
            pTransformer->registerTransform(newCameraToWorld);
            pTransformer->findTransform(frameName, "world");
            plotter->addPredictedCameraPosition(newCameraToWorld->mTranslation);
            plotter->updateFrustum(imageData, newCameraToWorld);

            motionModel.addPose(pTransformer->findTransform(prevFrameName, "world"), referenceFrame->mTimestamp);
            motionModel.addPose(pTransformer->findTransform(frameName, "world"), frame->mTimestamp);

            map = buildMap(prevFrameName, referenceFrame, frame, pts2D_1, pts2D_2, pts_opt);
            plotter->setMap(map.get());

            break; // Finished initialization
        }

        std::cout << "Frame handling took " << frameTimer.finish() << " ms" << std::endl;
        while (plotter->shouldPause()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    std::cout << "Finished initialization" << std::endl;

    // tracking
    std::uint32_t framesSinceLastKeyframe = 0;
    while (!dataloader->empty()) {
        const auto frameStart = std::chrono::steady_clock::now();
        std::shared_ptr<ImageData> imageData;
        std::shared_ptr<tft::RigidTransform3D> cameraToWorld;
        bool continues;
        if (getNextData(pTransformer, dataloader, plotter, cameraToIMU, imuAndGT, imageData, cameraToWorld, continues)) break;
        if (continues) continue;
        FrameTimer frameTimer(*plotter, frameStart);

        plotter->addGroundTruthCameraPosition(cameraToWorld->mTranslation);

        auto estimatedPose = motionModel.estimateLinearAt(imageData->mTimestamp);
        plotter->updateFrustum(imageData, estimatedPose);

        pTransformer->registerTransform(estimatedPose);
        pTransformer->findTransform(estimatedPose->mSource, "world");

        auto frame = std::make_shared<Frame>(imageData);
        featureDetector->detectFeatures(*frame);

        // TODO:
        // - match only with features in predicted scale
        const auto matches = matchMapPointsByProjection(map.get(), frame, estimatedPose, pTransformer, 50, 50);
        std::cout << "Matched " << matches.size() << " out of " << map->mPoints.size() << " map points" << std::endl;
        std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matchedFeatures;
        std::vector<Eigen::Vector3d> matchedMapPoints;
        matchedMapPoints.reserve(matches.size());
        matchedFeatures[0].reserve(matches.size());
        matchedFeatures[1].reserve(matches.size());
        for (const auto& match : matches) {
            matchedMapPoints.push_back(map->mPoints.at(match.mapPointId).mPosition);
            matchedFeatures[0].push_back(match.projectedFeature);
            matchedFeatures[1].push_back(match.feature);
        }
        plotter->plotMatches(frame->mImage, frame->mImage, matchedFeatures);

        const auto mapToCamera = pTransformer->findTransform(map->mFrame, estimatedPose->mSource);
        auto [rotation, translation, inliers] = optimizer->optimizePoseWithInliers(matchedFeatures[1],
            mapToCamera->mRotation, mapToCamera->mTranslation,
            matchedMapPoints, imageData->mIntrinsics);
        const Eigen::Vector3d acceptedCameraCenter = -rotation.transpose() * translation;
        for (std::size_t j = 0; j < matches.size(); ++j) {
            if (inliers[j] && !map->mPoints.at(matches[j].mapPointId).isViewDirectionCompatible(acceptedCameraCenter))
                inliers[j] = false;
        }
        const auto inlierCount = std::count(inliers.begin(), inliers.end(), true);
        std::cout << "Retained " << inlierCount << " pose inliers" << std::endl;
        if (inlierCount >= 12) {
            frame->mMapToCamera = std::make_shared<tft::RigidTransform3D>(
                map->mFrame, estimatedPose->mSource, rotation, translation);
            recordMapPointVisibility(map.get(), *frame, *frame->mMapToCamera);
            for (std::size_t j = 0; j < matches.size(); ++j) {
                if (inliers[j]) {
                    map->mPoints.at(matches[j].mapPointId).addInlierObservation(
                        matches[j].feature->getDescriptor(), acceptedCameraCenter);
                }
            }
            const auto mapToWorld = pTransformer->findTransform(map->mFrame, "world");

            if (map->decideNewKeyframe(inlierCount, framesSinceLastKeyframe)) {
                framesSinceLastKeyframe = 0;
                handleNewKeyframe(optimizer, motionModel, map.get(), frame, matches, inliers);
            }

            const auto optimizedCameraToWorld = mapToWorld * frame->mMapToCamera->inverse();
            pTransformer->registerTransform(optimizedCameraToWorld);
            pTransformer->findTransform(optimizedCameraToWorld->mSource, "world");
            plotter->updateFrustum(imageData, optimizedCameraToWorld);
            motionModel.addPose(optimizedCameraToWorld, imageData->mTimestamp);
            plotter->addPredictedCameraPosition(optimizedCameraToWorld->mTranslation);
        }

        framesSinceLastKeyframe++;

        std::cout << "Frame handling took " << frameTimer.finish() << " ms" << std::endl;
        while (plotter->shouldPause() || matchedMapPoints.size() < 12) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    std::cout << "Finished sequence, doing full map optimization pass..." << std::endl;
    map->cullMapPoints();
    optimizer->optimizeFullMap(map.get(), true);
    std::cout << "Finished!" << std::endl;

    render_loop.join();
    return 0;
}
