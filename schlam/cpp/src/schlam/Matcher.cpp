//
// Created by baldhat on 2/8/26.
//

#include "Matcher.h"

#include <vector>
#include <array>
#include <cstdint>
#include <limits>
#include <unordered_set>

std::vector<std::array<std::uint32_t, 2>> match(const std::vector<KeyPoint> &aKeypoints1, const std::vector<KeyPoint> &aKeypoints2, const double aMinDistance) {
    std::vector<std::array<std::uint32_t, 2>> matches;
    std::vector<bool> matchedIndices2(aKeypoints2.size(), false);

    for (std::uint32_t j = 0; j < aKeypoints1.size(); ++j) {
        auto descriptor = aKeypoints1[j].getDescriptor();
        std::uint32_t bestMatchIndex = -1;
        double bestMatchValue = std::numeric_limits<double>::max();

        for (std::uint32_t i = 0; i < aKeypoints2.size(); i++) {
            if (matchedIndices2[i]) continue;

            auto distance = (descriptor ^ aKeypoints2[i].getDescriptor()).count();
            if (distance < bestMatchValue) {
                bestMatchIndex = i;
                bestMatchValue = distance;
            }
        }

        if (bestMatchIndex != -1 && bestMatchValue < aMinDistance) {
            matches.push_back({j, bestMatchIndex});
            matchedIndices2[bestMatchIndex] = true;
        }
    }
    return matches;
}