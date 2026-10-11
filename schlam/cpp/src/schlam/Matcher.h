//
// Created by baldhat on 2/8/26.
//

#ifndef SCHLAM_MATCHER_H
#define SCHLAM_MATCHER_H

#include "KeyPoint.h"
#include "QuadTreeNode.h"
#include "Map.h"

#include <array>
#include <memory>
#include <vector>

#include "tft/transformer.hpp"


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


// The returned mask indexes map->mPoints, including points outside the image.
std::tuple<std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>, std::vector<bool>>
matchByProjection(const std::shared_ptr<Map> map, std::shared_ptr<Frame> frame,
                       std::shared_ptr<tft::RigidTransform3D> estimatedPose,
                       std::shared_ptr<tft::Transformer> aTransformer,
                       const double aMaxDistance, const double aWindowSize);

#endif //SCHLAM_MATCHER_H
