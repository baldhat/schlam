#include "XFeatMatcher.h"
#include "XFeatDescriptor.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
using Descriptors = Eigen::Matrix<float, Eigen::Dynamic, 64, Eigen::RowMajor>;
Descriptors matrix(const std::vector<DescriptorPtr>& descriptors) {
    Descriptors result(descriptors.size(), 64);
    for (std::size_t i = 0; i < descriptors.size(); ++i) {
        const auto descriptor = std::dynamic_pointer_cast<XFeatDescriptor>(descriptors[i]);
        if (!descriptor) throw std::invalid_argument("XFeat matching requires XFeat descriptors");
        result.row(i) = Eigen::Map<const Eigen::Matrix<float, 1, 64>>(descriptor->values().data());
    }
    return result;
}
}

std::vector<std::array<std::size_t, 2>> matchXFeat(
    const std::vector<DescriptorPtr>& first, const std::vector<DescriptorPtr>& second,
    double maximumDistance) {
    if (!std::isfinite(maximumDistance) || maximumDistance <= 0 || maximumDistance > 2)
        throw std::invalid_argument("XFeat match distance must be in (0, 2]");
    if (first.empty() || second.empty()) return {};
    const auto a = matrix(first), b = matrix(second);
    std::vector<std::size_t> forward(first.size()), reverse(second.size());
    std::vector<float> bestForward(first.size(), -std::numeric_limits<float>::infinity());
    std::vector<float> bestReverse(second.size(), -std::numeric_limits<float>::infinity());
    // Block the dot product to bound temporary memory for large maps. Both
    // directions use the same scores, rather than casting descriptors per pair.
    for (Eigen::Index start = 0; start < a.rows(); start += 128) {
        const Eigen::Index count = std::min<Eigen::Index>(128, a.rows()-start);
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> scores =
            a.middleRows(start, count) * b.transpose();
        for (Eigen::Index row = 0; row < count; ++row) {
            const auto i = static_cast<std::size_t>(start+row);
            for (std::size_t j = 0; j < second.size(); ++j) {
                const float similarity = scores(row,j);
                if (similarity > bestForward[i]) { bestForward[i] = similarity; forward[i] = j; }
                if (similarity > bestReverse[j]) { bestReverse[j] = similarity; reverse[j] = i; }
            }
        }
    }
    const float minimumSimilarity = static_cast<float>(1.0-maximumDistance);
    std::vector<std::array<std::size_t, 2>> result;
    for (std::size_t i = 0; i < first.size(); ++i)
        if (reverse[forward[i]] == i && (minimumSimilarity <= 0 || bestForward[i] > minimumSimilarity))
            result.push_back({i, forward[i]});
    return result;
}
