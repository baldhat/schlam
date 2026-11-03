//
// Created by baldhat on 2/8/26.
//

#ifndef SCHLAM_MATCHER_H
#define SCHLAM_MATCHER_H

#include <array>
#include <memory>
#include <tuple>
#include <vector>

#include "MapTypes.h"


class Frame;
class Map;
class KeyPoint;
class QuadTreeNode;
class TrackingSnapshot;
class KeyFrame;

namespace tft {
    class RigidTransform3D;
    class Transformer;
}



std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matchGlobally(const std::vector<std::shared_ptr<KeyPoint>>& keypoints1,
                                                     const std::vector<std::shared_ptr<KeyPoint>>& keypoints2,
                                                     const double aMaxDistance);

std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>
matchGlobally(const std::shared_ptr<QuadTreeNode>& aTree1,
                                                     const std::shared_ptr<QuadTreeNode>& aTree2,
                                                     const double aMaxDistance);

std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>
getMatched(const std::vector<std::shared_ptr<KeyPoint>>& aKeypoints1,
                                                  const std::vector<std::shared_ptr<KeyPoint>>& aKeypoints2,
                                                  const std::vector<std::array<std::uint32_t, 2>>& aMatches);


std::tuple<std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>, std::vector<bool>>
matchWindow(const std::shared_ptr<QuadTreeNode>& aTree1,
                                                   const std::shared_ptr<QuadTreeNode>& aTree2,
                                                   const double aMaxDistance, const double aWindowSize);


std::tuple<std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>, std::vector<bool>>
matchWindow(const std::vector<std::shared_ptr<KeyPoint>>& aKeypoints,
                                                   const std::shared_ptr<QuadTreeNode>& aTree2,
                                                   const double aMaxDistance, const double aWindowSize);


struct MapPointMatch {
    MapPointId mapPointId;
    std::shared_ptr<KeyPoint> feature;
    std::shared_ptr<KeyPoint> projectedFeature;
};

// Count projected points only after the frame pose has been accepted.
void recordMapPointVisibility(TrackingSnapshot& snapshot, const Frame& frame,
                              const tft::RigidTransform3D& mapToCamera);


std::vector<MapPointMatch>
matchMapPointsByProjection(TrackingSnapshot& snapshot,
                          const std::shared_ptr<Frame>& frame,
                          const std::shared_ptr<tft::RigidTransform3D>& estimatedPose,
                          const std::shared_ptr<tft::Transformer>& aTransformer,
                          const double aMaxDistance, const double aWindowSize);


// The returned mask follows ascending MapPointId order.
std::tuple<std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>, std::vector<bool>>
matchByProjection(Map* map, std::shared_ptr<Frame> frame,
                       std::shared_ptr<tft::RigidTransform3D> estimatedPose,
                       std::shared_ptr<tft::Transformer> aTransformer,
                       const double aMaxDistance, const double aWindowSize);

std::array<std::vector<std::shared_ptr<KeyPoint>>, 2> matchUnmatchedFeatures(KeyFrame& frame1, KeyFrame& frame2, const double aMaxDistance);


#endif //SCHLAM_MATCHER_H
