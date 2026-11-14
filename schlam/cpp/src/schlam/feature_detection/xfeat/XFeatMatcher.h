#pragma once

#include "schlam/feature_detection/IDescriptor.h"
#include <array>
#include <vector>

// Upstream XFeat sparse matching: global cosine similarity, mutual argmax,
// then a strict similarity threshold. Indices follow input order (first wins ties).
// maximumDistance is 1-min_cossim; upstream disables filtering when min_cossim <= 0.
std::vector<std::array<std::size_t, 2>> matchXFeat(
    const std::vector<DescriptorPtr>& first, const std::vector<DescriptorPtr>& second,
    double maximumDistance);
