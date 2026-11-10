#include "FeatureSelection.h"

#include "../KeyPoint.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace schlam::detail {
namespace {

constexpr auto noChild = std::numeric_limits<std::size_t>::max();

struct SelectionNode {
    double left, right, top, bottom;
    std::size_t begin, end;
    double score;
    std::array<std::size_t, 4> children{noChild, noChild, noChild, noChild};

    bool isLeaf() const {
        return std::all_of(children.begin(), children.end(), [](auto child) { return child == noChild; });
    }
};

}

std::vector<std::size_t> selectFeatureIndices(const std::vector<KeyPoint>& features,
                                            const std::uint32_t width, const std::uint32_t height,
                                            const std::uint32_t count) {
    if (count == 0 || features.empty()) return {};

    std::vector<std::size_t> indices(features.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](auto a, auto b) {
        return features[a].getScore() > features[b].getScore();
    });
    if (count >= indices.size()) return indices;

    // All nodes share two index buffers. Stable partitioning keeps each node's
    // highest-scoring candidate first without copying KeyPoints or shared_ptrs.
    std::vector<std::size_t> scratch(indices.size());
    std::vector<SelectionNode> nodes;
    nodes.reserve(count);
    nodes.push_back({0, double(width), 0, double(height), 0, indices.size(),
                     features[indices.front()].getScore()});
    const auto divide = [&](std::size_t nodeIndex) {
        const auto node = nodes[nodeIndex];
        if (node.end - node.begin < 2 ||
            (node.right - node.left <= 1 && node.bottom - node.top <= 1)) return false;

        const double midX = node.left + std::floor((node.right - node.left) / 2.0);
        const double midY = node.top + std::floor((node.bottom - node.top) / 2.0);
        const auto quadrant = [&](std::size_t index) {
            return (features[index].getImgX() < midX ? 0 : 1) +
                   (features[index].getImgY() < midY ? 0 : 2);
        };
        std::array<std::size_t, 4> sizes{};
        for (auto i = node.begin; i < node.end; ++i) ++sizes[quadrant(indices[i])];
        std::array<std::size_t, 4> starts{node.begin};
        for (std::size_t q = 1; q < starts.size(); ++q) starts[q] = starts[q - 1] + sizes[q - 1];
        auto positions = starts;
        for (auto i = node.begin; i < node.end; ++i) {
            const auto index = indices[i];
            scratch[positions[quadrant(index)]++] = index;
        }
        std::copy(scratch.begin() + node.begin, scratch.begin() + node.end, indices.begin() + node.begin);
        for (std::size_t q = 0; q < sizes.size(); ++q) {
            if (sizes[q] == 0) continue;
            nodes[nodeIndex].children[q] = nodes.size();
            nodes.push_back({q % 2 == 0 ? node.left : midX, q % 2 == 0 ? midX : node.right,
                             q < 2 ? node.top : midY, q < 2 ? midY : node.bottom,
                             starts[q], starts[q] + sizes[q], features[indices[starts[q]]].getScore()});
        }
        return true;
    };

    std::vector<std::size_t> frontier{0}, next;
    frontier.reserve(count);
    next.reserve(count);
    while (frontier.size() < count) {
        next.clear();
        bool divided = false;
        for (auto index : frontier) {
            if (divide(index)) {
                divided = true;
                for (auto child : nodes[index].children) {
                    if (child != noChild) next.push_back(child);
                }
            } else {
                next.push_back(index);
            }
        }
        frontier.swap(next);
        if (!divided) break;
    }

    // Preserve the original breadth-first branch pruning, including score ties.
    frontier.assign(1, 0);
    for (;;) {
        if (frontier.size() > count) {
            std::nth_element(frontier.begin(), frontier.begin() + count, frontier.end(), [&](auto a, auto b) {
                return nodes[a].score > nodes[b].score;
            });
            frontier.resize(count);
        } else {
            next.clear();
            bool allLeaves = true;
            for (auto index : frontier) {
                if (nodes[index].isLeaf()) {
                    next.push_back(index);
                } else {
                    allLeaves = false;
                    for (auto child : nodes[index].children) {
                        if (child != noChild) next.push_back(child);
                    }
                }
            }
            if (allLeaves) break;
            frontier.swap(next);
        }
    }
    std::sort(frontier.begin(), frontier.end(), [&](auto a, auto b) { return nodes[a].begin < nodes[b].begin; });
    for (auto& index : frontier) index = indices[nodes[index].begin];
    return frontier;
}

}
