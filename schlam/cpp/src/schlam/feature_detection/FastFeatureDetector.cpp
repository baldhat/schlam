//
// Created by baldhat on 1/4/26.
//

#include "FastFeatureDetector.h"

#include <stdexcept>

namespace schlam {
    FastFeatureDetector::FastFeatureDetector(const int aFastThreshold, const int aNumContinuous)
        : mFastThreshold(aFastThreshold), mNumContinuous(aNumContinuous) {
        if (aFastThreshold < 0 || aFastThreshold > 255) {
            throw std::invalid_argument("FAST threshold must be between 0 and 255");
        }
        if (aNumContinuous < 1 || aNumContinuous > 16) {
            throw std::invalid_argument("FAST contiguous arc length must be between 1 and 16");
        }
    }

    std::vector<KeyPoint> FastFeatureDetector::calculateFastFeatures(const cv::Mat &aImage) const {
        std::vector<KeyPoint> keypoints;
        keypoints.reserve(aImage.rows * aImage.cols / 10);

        const int rows = aImage.rows;
        const int cols = aImage.cols;
        const int step = aImage.step;

        const uint8_t *imgData = aImage.data;
        // Every four consecutive circle samples contain one cardinal sample.
        const int minCardinalSamples = mNumContinuous / 4;

        std::array<int, 16> offsets;
        for (int k = 0; k < 16; ++k)
            offsets[k] = mFastIndices[k][0] * step + mFastIndices[k][1];

        for (int i = 3; i < rows - 3; ++i) {
            const uint8_t *rowPtr = imgData + i * step;

            for (int j = 3; j < cols - 3; ++j) {
                const uint8_t *centerPtr = rowPtr + j;
                const int center = *centerPtr;

                const int t_high = center + mFastThreshold;
                const int t_low = center - mFastThreshold;

                int brighter = 0, darker = 0;

                const int idx0 = 0;
                const int idx4 = 4;
                const int idx8 = 8;
                const int idx12 = 12;

                int p;

                p = *(centerPtr + offsets[idx0]);
                brighter += (p > t_high);
                darker += (p < t_low);

                p = *(centerPtr + offsets[idx4]);
                brighter += (p > t_high);
                darker += (p < t_low);

                p = *(centerPtr + offsets[idx8]);
                brighter += (p > t_high);
                darker += (p < t_low);

                p = *(centerPtr + offsets[idx12]);
                brighter += (p > t_high);
                darker += (p < t_low);

                if (brighter < minCardinalSamples && darker < minCardinalSamples)
                    continue;

                int pos_count = 0;
                int neg_count = 0;

                // Repeat the first samples to detect arcs crossing the end of the circle.
                for (int k = 0; k < 16 + mNumContinuous - 1; ++k) {
                    const int idx = k < 16 ? k : k - 16;

                    p = *(centerPtr + offsets[idx]);

                    const bool isBright = (p > t_high);
                    const bool isDark = (p < t_low);

                    pos_count = isBright ? pos_count + 1 : 0;
                    neg_count = isDark ? neg_count + 1 : 0;

                    if (pos_count >= mNumContinuous || neg_count >= mNumContinuous) {
                        keypoints.emplace_back(static_cast<uint32_t>(j),
                                               static_cast<uint32_t>(i));
                        break;
                    }
                }
            }
        }

        return keypoints;
    }
} // schlam
