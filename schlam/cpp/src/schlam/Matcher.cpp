//
// Created by baldhat on 2/8/26.
//

#include "Matcher.h"

#include <vector>
#include <array>
#include <cstdint>
#include <limits>

inline int hamming256(const uint8_t* a, const uint8_t* b)
{
    const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a);
    const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b);

    return  __builtin_popcountll(a64[0] ^ b64[0]) +
            __builtin_popcountll(a64[1] ^ b64[1]) +
            __builtin_popcountll(a64[2] ^ b64[2]) +
            __builtin_popcountll(a64[3] ^ b64[3]);
}

std::vector<std::array<std::uint32_t, 2>> match(const std::vector<KeyPoint> &aKeypoints1, const std::vector<KeyPoint> &aKeypoints2, const double aMinDistance) {
    std::vector<std::array<std::uint32_t, 2>> matches;
    std::vector<bool> matchedIndices2(aKeypoints2.size(), false);

    for (std::uint32_t j = 0; j < aKeypoints1.size(); ++j) {
        auto descriptor = aKeypoints1[j].getDescriptor();
        std::uint32_t bestMatchIndex = -1;
        double bestMatchValue = std::numeric_limits<double>::max();

        for (std::uint32_t i = 0; i < aKeypoints2.size(); i++) {
            if (matchedIndices2[i]) continue;

            auto distance = hamming256(descriptor.data(), aKeypoints2[i].getDescriptor().data());
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

std::array<std::vector<KeyPoint>, 2> getMatched(const std::vector<KeyPoint> &aKeypoints1,
                                                 const std::vector<KeyPoint> &aKeypoints2,
                                                 std::vector<std::array<std::uint32_t, 2>>& aMatches) {
  std::vector<KeyPoint> out1, out2;
  for (const auto& [id1, id2]: aMatches) {
    out1.push_back(aKeypoints1[id1]);
    out2.push_back(aKeypoints2[id2]);
  }
  return {out1, out2};
}
