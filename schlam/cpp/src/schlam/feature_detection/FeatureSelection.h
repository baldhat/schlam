#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

class KeyPoint;

namespace schlam::detail {

// Returns the spatially selected features in quadtree traversal order.
std::vector<std::size_t> selectFeatureIndices(const std::vector<KeyPoint>& features,
                                            std::uint32_t width, std::uint32_t height,
                                            std::uint32_t count);

}
