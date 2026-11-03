//
// Created by jonathan_michel on 10/28/26.
//

#ifndef SCHLAM_COVISIBILITYHELPERS_H
#define SCHLAM_COVISIBILITYHELPERS_H

#include <cstdint>
#include <utility>
#include <vector>
#include <algorithm>
#include <set>

#include "MapTypes.h"

class Map;
using Neighbor = std::pair<KeyFrameId, std::uint32_t>;
std::vector<Neighbor> getCovisibilityNeighbors(
    Map* map,
    KeyFrameId frameId,
    std::size_t numNeighbors,
    std::uint32_t minSharedObservedCount
);

#endif //SCHLAM_COVISIBILITYHELPERS_H
