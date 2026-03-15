//
// Created by baldhat on 2/8/26.
//

#include "Matcher.h"

#include <vector>
#include <array>
#include <cstdint>
#include <limits>

inline int hamming256(const uint8_t *a, const uint8_t *b) {
    const uint64_t *a64 = reinterpret_cast<const uint64_t *>(a);
    const uint64_t *b64 = reinterpret_cast<const uint64_t *>(b);

    return __builtin_popcountll(a64[0] ^ b64[0]) +
           __builtin_popcountll(a64[1] ^ b64[1]) +
           __builtin_popcountll(a64[2] ^ b64[2]) +
           __builtin_popcountll(a64[3] ^ b64[3]);
}

std::array<std::vector<KeyPoint *>, 2> matchGlobally(QuadTreeNode *aTree1,
                                                     QuadTreeNode *aTree2,
                                                     const double aMaxDistance) {
    auto keypoints1 = aTree1->getFeatures();
    auto keypoints2 = aTree2->getFeatures();
    std::vector<std::array<std::uint32_t, 2> > matches;
    std::vector<bool> matchedIndices2(keypoints2.size(), false);

    for (std::uint32_t j = 0; j < keypoints1.size(); ++j) {
        auto descriptor = keypoints1[j]->getDescriptor();
        std::uint32_t bestMatchIndex = -1;
        double bestMatchValue = std::numeric_limits<double>::max();

        for (std::uint32_t i = 0; i < keypoints2.size(); i++) {
            if (matchedIndices2[i]) continue;

            auto distance = hamming256(descriptor.data(), keypoints2[i]->getDescriptor().data());
            if (distance < bestMatchValue) {
                bestMatchIndex = i;
                bestMatchValue = distance;
            }
        }

        if (bestMatchIndex != -1 && bestMatchValue < aMaxDistance) {
            matches.push_back({j, bestMatchIndex});
            matchedIndices2[bestMatchIndex] = true;
        }
    }
    return getMatched(keypoints1, keypoints2, matches);
}

std::array<std::vector<KeyPoint *>, 2> matchWindow(const QuadTreeNode *aTree1,
                                                   const QuadTreeNode *aTree2,
                                                   const double aMaxDistance, const double aWindowSize) {
    std::array<std::vector<KeyPoint *>, 2> matches;
    const auto features1 = aTree1->getFeatures();

    for (const auto feature: features1) {
        const auto descriptor = feature->getDescriptor();
        KeyPoint *bestMatch = nullptr;
        double bestMatchValue = std::numeric_limits<double>::max();

        const auto x{feature->getImgX()}, y{feature->getImgY()};
        const auto x1{x - aWindowSize}, x2{x + aWindowSize}, y1{y - aWindowSize}, y2{y + aWindowSize};
        const auto candidates = aTree2->getFeatures(x1, x2, y1, y2);

        for (const auto candidate: candidates) {
            if (candidate->isMatched()) continue;

            const auto distance = hamming256(descriptor.data(), candidate->getDescriptor().data());
            if (distance < bestMatchValue) {
                bestMatch = candidate;
                bestMatchValue = distance;
            }
        }

        if (bestMatch && bestMatchValue < aMaxDistance) {
            matches[0].push_back(feature);
            matches[1].push_back(bestMatch);
            bestMatch->setMatched(true);
        }
    }
    return matches;
}

std::array<std::vector<KeyPoint *>, 2> getMatched(const std::vector<KeyPoint *> &aKeypoints1,
                                                  const std::vector<KeyPoint *> &aKeypoints2,
                                                  std::vector<std::array<std::uint32_t, 2> > &aMatches) {
    std::vector<KeyPoint *> out1, out2;
    for (const auto &[id1, id2]: aMatches) {
        out1.push_back(aKeypoints1[id1]);
        out2.push_back(aKeypoints2[id2]);
    }
    return {out1, out2};
}
