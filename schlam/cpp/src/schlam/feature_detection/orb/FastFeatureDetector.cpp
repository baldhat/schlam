//
// Created by baldhat on 1/4/26.
//

#include "FastFeatureDetector.h"

#include <stdexcept>
#include <opencv2/core/utility.hpp>
#include <tracy/Tracy.hpp>

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
    std::vector<KeyPoint> FastFeatureDetector::calculateFastFeatures(
        const cv::Mat& aImage) const
    {
        ZoneScoped;
        const int rows = aImage.rows;
        const int cols = aImage.cols;

        if (rows < 7 || cols < 7)
            return {};

        const auto step = static_cast<std::ptrdiff_t>(aImage.step);
        const uint8_t* imgData = aImage.data;

        const int minCardinalSamples = mNumContinuous / 4;

        std::array<std::ptrdiff_t, 16> offsets;
        for (int k = 0; k < 16; ++k) {
            offsets[k] =
                mFastIndices[k][0] * step + mFastIndices[k][1];
        }

        // Partition the image into blocks of rows.
        constexpr int blockHeight = 32;
        const int numBlocks =
            (rows - 6 + blockHeight - 1) / blockHeight;

        // Each block writes to its own vector.
        std::vector<std::vector<KeyPoint>> blockResults(numBlocks);

        cv::parallel_for_(
            cv::Range(0, numBlocks),
            [&](const cv::Range& range) {
                for (int block = range.start;
                     block < range.end; ++block) {

                    auto& localFeatures = blockResults[block];

                    const int rowBegin = 3 + block * blockHeight;
                    const int rowEnd =
                        std::min(rowBegin + blockHeight, rows - 3);

                    for (int i = rowBegin; i < rowEnd; ++i) {
                        const uint8_t* rowPtr = imgData + i * step;

                        for (int j = 3; j < cols - 3; ++j) {
                            const uint8_t* centerPtr = rowPtr + j;
                            const int center = *centerPtr;

                            const int t_high = center + mFastThreshold;
                            const int t_low = center - mFastThreshold;

                            int brighter = 0;
                            int darker = 0;

                            for (int idx : {0, 4, 8, 12}) {
                                const int p = *(centerPtr + offsets[idx]);
                                brighter += (p > t_high);
                                darker += (p < t_low);
                            }

                            if (brighter < minCardinalSamples &&
                                darker < minCardinalSamples)
                                continue;

                            int pos_count = 0;
                            int neg_count = 0;

                            for (int k = 0;
                                 k < 16 + mNumContinuous - 1; ++k) {

                                const int idx = k < 16 ? k : k - 16;
                                const int p = *(centerPtr + offsets[idx]);

                                pos_count = p > t_high ? pos_count + 1 : 0;
                                neg_count = p < t_low ? neg_count + 1 : 0;

                                if (pos_count >= mNumContinuous ||
                                    neg_count >= mNumContinuous) {
                                    localFeatures.emplace_back(
                                        static_cast<uint32_t>(j),
                                        static_cast<uint32_t>(i));
                                    break;
                                }
                            }
                        }
                    }
                }
            });

        // Merge results in image-row order.
        std::size_t total = 0;
        for (const auto& block : blockResults)
            total += block.size();

        std::vector<KeyPoint> keypoints;
        keypoints.reserve(total);

        for (auto& block : blockResults) {
            keypoints.insert(
                keypoints.end(),
                std::make_move_iterator(block.begin()),
                std::make_move_iterator(block.end()));
        }

        return keypoints;
    }
} // schlam
