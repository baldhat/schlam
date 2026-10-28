//
// Created by jonathan_michel on 10/28/26.
//

#include "CovisibilityHelpers.h"

#include "Map.h"

std::vector<Neighbor> getCovisibilityNeighbors(Map* map,
                                               KeyFrameId frameId,
                                               std::size_t numNeighbors,
                                               std::uint32_t minSharedObservedCount) {
    std::vector<Neighbor> neighbors;
    auto& keyframe = map->mKeyFrames.at(frameId);
    for (const auto& [frameId, sharedObservedCount] : keyframe.mSharedPointCount) {
        if (frameId == keyframe.mId || sharedObservedCount < minSharedObservedCount)
            continue;

        neighbors.emplace_back(frameId, sharedObservedCount);
    }

    const std::size_t keep =
      std::min(neighbors.size(), numNeighbors); // current keyframe uses one slot
    std::partial_sort(
      neighbors.begin(), neighbors.begin() + keep, neighbors.end(),
      [](const Neighbor& a, const Neighbor& b) {
          if (a.second != b.second) return a.second > b.second;
          return a.first < b.first;
      });
    neighbors.resize(keep);
    return neighbors;
}