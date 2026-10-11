//
// Created by baldhat on 2/8/26.
//

#ifndef SCHLAM_MATCHER_H
#define SCHLAM_MATCHER_H

#include "KeyPoint.h"
#include "QuadTreeNode.h"
#include "Map.h"

#include <array>
#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>

#include "tft/transformer.hpp"

class Frame;

std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>
matchGlobally(const std::shared_ptr<QuadTreeNode>& aTree1,
                                                     const std::shared_ptr<QuadTreeNode>& aTree2,
                                                     const double aMaxDistance);

std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>
getMatched(const std::vector<std::shared_ptr<KeyPoint>>& aKeypoints1,
                                                  const std::vector<std::shared_ptr<KeyPoint>>& aKeypoints2,
                                                  const std::vector<std::array<std::uint32_t, 2>>& aMatches);

// The returned mask indexes the source tree's features.
std::tuple<std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>, std::vector<bool>>
matchWindow(const std::shared_ptr<QuadTreeNode>& aTree1,
                                                   const std::shared_ptr<QuadTreeNode>& aTree2,
                                                   const double aMaxDistance, const double aWindowSize);

// The returned mask indexes aKeypoints, including unmatched entries.
std::tuple<std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>, std::vector<bool>>
matchWindow(const std::vector<std::shared_ptr<KeyPoint>>& aKeypoints,
                                                   const std::shared_ptr<QuadTreeNode>& aTree2,
                                                   const double aMaxDistance, const double aWindowSize);


struct MapPointMatch {
    std::size_t mapPointIndex;
    std::shared_ptr<KeyPoint> feature;
    std::shared_ptr<KeyPoint> projectedFeature;
};

// Each match carries its map point explicitly. Uses the latest inlier descriptor
// first, with the representative descriptor as a fallback in the same window.
std::vector<MapPointMatch>
matchMapPointsByProjection(const std::shared_ptr<Map>& map,
                          const std::shared_ptr<Frame>& frame,
                          const std::shared_ptr<tft::RigidTransform3D>& estimatedPose,
                          const std::shared_ptr<tft::Transformer>& aTransformer,
                          const double aMaxDistance, const double aWindowSize);

// Compatibility interface: the returned mask indexes map->mPoints.
std::tuple<std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>, std::vector<bool>>
matchByProjection(const std::shared_ptr<Map> map, std::shared_ptr<Frame> frame,
                       std::shared_ptr<tft::RigidTransform3D> estimatedPose,
                       std::shared_ptr<tft::Transformer> aTransformer,
                       const double aMaxDistance, const double aWindowSize);

#endif //SCHLAM_MATCHER_H
