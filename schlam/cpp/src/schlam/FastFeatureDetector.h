//
// Created by baldhat on 1/4/26.
//

#ifndef SCHLAM_FASTFEATUREDETECTOR_H
#define SCHLAM_FASTFEATUREDETECTOR_H

#include <array>
#include <cstdint>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "KeyPoint.h"

namespace schlam {
    class FastFeatureDetector {
    public:
        // Threshold: 0..255 intensity units; contiguous arc length: 1..16 samples.
        explicit FastFeatureDetector(int aFastThreshold = 7, int aNumContinuous = 9);

        std::vector<KeyPoint> calculateFastFeatures(const cv::Mat &aImage) const;

    private:
        std::array<std::array<std::int8_t, 2>, 16> mFastIndices = {
            std::array<std::int8_t, 2>{-3, 0},
            {-3, 1},
            {-2, 2},
            {-1, 3},
            {0, 3},
            {1, 3},
            {2, 2},
            {3, 1},
            {3, 0},
            {3, -1},
            {2, -2},
            {1, -3},
            {0, -3},
            {-1, -3},
            {-2, -2},
            {-3, -1}
        };
        int mFastThreshold;
        int mNumContinuous;
    };
} // schlam

#endif //SCHLAM_FASTFEATUREDETECTOR_H
