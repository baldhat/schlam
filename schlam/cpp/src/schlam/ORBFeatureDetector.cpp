//
// Created by baldhat on 2/7/26.
//

#include "ORBFeatureDetector.h"

#include <opencv2/opencv.hpp>

#include "src/plotting/plotter.hpp"

#include <algorithm>
#include <bitset>
#include <execution>
#include <vector>

#include "orb_pattern.h"
#include "QuadTreeNode.h"

ORBFeatureDetector::ORBFeatureDetector(const std::uint32_t aNumFeatures,
                                       std::shared_ptr<Plotter> apPlotter,
                                       const std::uint8_t aNumLevels,
                                       const double aLevelFactor) : mNumFeatures(aNumFeatures), mNumLevels(aNumLevels),
                                                                    mLevelFactor(aLevelFactor), mpPlotter(apPlotter) {
    for (std::uint32_t i = 0; i < mNumLevels; i++) {
        mLevelFactors.push_back(std::pow(mLevelFactor, i));
    }
    mOrientationIndices = getPointsInRadius(31);

    distributeFeaturesToLevels();
}

std::vector<KeyPoint> ORBFeatureDetector::getFeatures(const cv::Mat &aImage) {
    auto pyramid = buildPyramid(aImage, mNumLevels, mLevelFactor);
    auto features = calcFeatures(pyramid);
    addOrientation(pyramid, features);
    addDescriptors(pyramid, features);
    return features;
}

void ORBFeatureDetector::distributeFeaturesToLevels() {
    auto featuresPerScale = mNumFeatures * (1 - mLevelFactor) / (1 - std::pow(mLevelFactor, mNumLevels));
    std::int32_t numFeatures{0};
    for (std::uint8_t level = 0; level < mNumLevels - 1; level++) {
        mLevels.push_back(level);
        mNumFeaturesPerLevel.push_back(static_cast<std::uint32_t>(std::round(featuresPerScale * mLevelFactors[level])));
        numFeatures += mNumFeaturesPerLevel[level];
    }
    mLevels.push_back(mNumLevels - 1);
    mNumFeaturesPerLevel.push_back(std::max(static_cast<std::int32_t>(mNumFeatures) - numFeatures, 0));
}

std::vector<KeyPoint> ORBFeatureDetector::calcFeatures(const std::vector<cv::Mat> &aPyramid) {
    std::vector<KeyPoint> features;
    std::mutex mut;
    std::for_each(std::execution::par, mLevels.begin(), mLevels.end(), [&](std::uint8_t level) {
        const auto levelImageHeight{aPyramid[level].rows}, levelImageWidth{aPyramid[level].cols};
        auto levelFeatures = calculateFastFeatures(aPyramid[level]);
        removeAtImageBorder(levelFeatures, levelImageWidth, levelImageHeight, 3);
        computeHarrisResponse(aPyramid[level], levelFeatures, 7);
        levelFeatures = filterWithOctree(levelFeatures, levelImageWidth, levelImageHeight, mNumFeaturesPerLevel[level]);
        rescaleFeatures(levelFeatures, 1.0 / mLevelFactors[level], level);
        std::lock_guard<std::mutex> guard(mut);
        features.insert(features.end(), levelFeatures.begin(), levelFeatures.end());
    });
    std::cout << "Returning " << features.size() << " features" << std::endl;
    mpPlotter->plotFeatures(aPyramid[0], features);
    return features;
}

void ORBFeatureDetector::addDescriptors(
    const std::vector<cv::Mat> &aPyramid,
    std::vector<KeyPoint> &aFeatures)
{
    const size_t nLevels = aPyramid.size();

    std::vector<cv::Mat> blurredPyramid(nLevels);
    for (size_t lvl = 0; lvl < nLevels; ++lvl)
    {
        cv::GaussianBlur(aPyramid[lvl],
                         blurredPyramid[lvl],
                         cv::Size(7, 7),
                         2, 2,
                         cv::BORDER_REPLICATE);
    }

    for (auto &feature : aFeatures)
    {
        const int level = feature.getLevel();
        const double factor = mLevelFactors[level];

        const cv::Mat &img = blurredPyramid[level];
        const uint8_t* imgData = img.data;
        const int step = img.step;

        // Faster than std::round
        const int centerX = int(feature.getImgX() * factor + 0.5);
        const int centerY = int(feature.getImgY() * factor + 0.5);

        const float angle = static_cast<float>(feature.getAngle());
        const float sinA = std::sin(angle);
        const float cosA = std::cos(angle);

        uint8_t desc[32] = {0};

        const uint8_t* centerPtr = imgData + centerY * step + centerX;

        for (int i = 0; i < 256; ++i)
        {
            const int idx = i * 4;

            const int x0 = gOrbBitPattern31[idx];
            const int y0 = gOrbBitPattern31[idx + 1];
            const int x1 = gOrbBitPattern31[idx + 2];
            const int y1 = gOrbBitPattern31[idx + 3];

            const int rx0 = int(x0 * cosA - y0 * sinA + 0.5f);
            const int ry0 = int(x0 * sinA + y0 * cosA + 0.5f);
            const int rx1 = int(x1 * cosA - y1 * sinA + 0.5f);
            const int ry1 = int(x1 * sinA + y1 * cosA + 0.5f);

            const uint8_t p0 = *(centerPtr + ry0 * step + rx0);
            const uint8_t p1 = *(centerPtr + ry1 * step + rx1);

            desc[i >> 3] |= (p0 < p1) << (i & 7);
        }

        feature.setDescriptor(std::to_array(desc)); 
    }
}

// void ORBFeatureDetector::addDescriptors(const std::vector<cv::Mat> &aPyramid, std::vector<KeyPoint> &aFeatures) {
//     std::vector<cv::Mat> blurredPyramid{aPyramid.size()};
//     for (std::uint32_t i = 0; i < aPyramid.size(); i++) {
//         cv::GaussianBlur(aPyramid[i], blurredPyramid[i], cv::Size(7, 7), 2, 2);
//     }
//     for (auto &feature: aFeatures) {
//         const double factor = mLevelFactors[feature.getLevel()];
//         const std::uint8_t *data = blurredPyramid[feature.getLevel()].data;
//         const int step = blurredPyramid[feature.getLevel()].step;
//
//         std::int32_t centerX{static_cast<std::int32_t>(std::round(feature.getImgX() * factor))};
//         std::int32_t centerY{static_cast<std::int32_t>(std::round(feature.getImgY() * factor))};
//         double sin{std::sin(feature.getAngle())}, cos{std::cos(feature.getAngle())};
//
//         std::bitset<256> descriptor;
//         for (std::uint32_t i = 0; i < 256 * 4; i += 4) {
//             std::int32_t x0{gOrbBitPattern31[i]}, y0{gOrbBitPattern31[i + 1]};
//             std::int32_t x1{gOrbBitPattern31[i + 2]}, y1{gOrbBitPattern31[i + 3]};
//             auto x0r = static_cast<std::int32_t>(std::round(x0 * cos - y0 * sin));
//             auto y0r = static_cast<std::int32_t>(std::round(x0 * sin + y0 * cos));
//             auto x1r = static_cast<std::int32_t>(std::round(x1 * cos - y1 * sin));
//             auto y1r = static_cast<std::int32_t>(std::round(x1 * sin + y1 * cos));
//             auto p0 = data[(y0r + centerY) * step + (x0r + centerX)];
//             auto p1 = data[(y1r + centerY) * step + (x1r + centerX)];
//             descriptor[i/4] = p0 < p1;
//         }
//         feature.setDescriptor(descriptor);
//     }
// }

std::vector<KeyPoint> ORBFeatureDetector::filterWithOctree(std::vector<KeyPoint> &aFeatures, const std::uint32_t aWidth,
                                                           const std::uint32_t aHeight,
                                                           const std::uint32_t aNumFeatures) {
    const auto root = std::make_shared<QuadTreeNode>(0, aWidth, 0, aHeight, aFeatures);
    if (aNumFeatures > aFeatures.size()) {
        return aFeatures;
    }
    std::vector<std::shared_ptr<QuadTreeNode> > nodes;
    nodes.push_back(root);
    while (nodes.size() < aNumFeatures) {
        std::vector<std::shared_ptr<QuadTreeNode> > newNodes;
        for (auto &node: nodes) {
            if (node->isLeaf()) {
                newNodes.push_back(node);
                continue;
            }
            for (auto children = node->divide(); auto &child: children) {
                if (!child->isLeaf()) {
                    newNodes.push_back(child);
                }
            }
        }
        nodes = newNodes;
    }

    std::vector<KeyPoint> features;
    for (const auto &node: nodes) {
        features.push_back(node->getMaxFeature());
    }
    return features;
}


std::vector<KeyPoint>
ORBFeatureDetector::calculateFastFeatures(const cv::Mat &aImage)
{
    std::vector<KeyPoint> keypoints;
    keypoints.reserve(aImage.rows * aImage.cols / 10);

    const int rows = aImage.rows;
    const int cols = aImage.cols;
    const int step = aImage.step;

    const uint8_t* imgData = aImage.data;

    std::array<int, 16> offsets;
    for (int k = 0; k < 16; ++k)
        offsets[k] = mFastIndices[k][0] * step + mFastIndices[k][1];

    for (int i = 3; i < rows - 3; ++i)
    {
        const uint8_t* rowPtr = imgData + i * step;

        for (int j = 3; j < cols - 3; ++j)
        {
            const uint8_t* centerPtr = rowPtr + j;
            const int center = *centerPtr;

            const int t_high = center + mFastThreshold;
            const int t_low  = center - mFastThreshold;

            int brighter = 0, darker = 0;

            const int idx0  = 0;
            const int idx4  = 4;
            const int idx8  = 8;
            const int idx12 = 12;

            int p;

            p = *(centerPtr + offsets[idx0]);
            brighter += (p > t_high);
            darker   += (p < t_low);

            p = *(centerPtr + offsets[idx4]);
            brighter += (p > t_high);
            darker   += (p < t_low);

            p = *(centerPtr + offsets[idx8]);
            brighter += (p > t_high);
            darker   += (p < t_low);

            p = *(centerPtr + offsets[idx12]);
            brighter += (p > t_high);
            darker   += (p < t_low);

            if (brighter < 3 && darker < 3)
                continue;

            int pos_count = 0;
            int neg_count = 0;

            // Loop 16 + (nContinuous - 1) for wrap-around
            for (int k = 0; k < 16 + nContinuous - 1; ++k)
            {
                const int idx = k < 16 ? k : k - 16;

                p = *(centerPtr + offsets[idx]);

                const bool isBright = (p > t_high);
                const bool isDark   = (p < t_low);

                pos_count = isBright ? pos_count + 1 : 0;
                neg_count = isDark   ? neg_count + 1 : 0;

                if (pos_count >= nContinuous || neg_count >= nContinuous)
                {
                    keypoints.emplace_back(static_cast<uint32_t>(j),
                                           static_cast<uint32_t>(i));
                    break;
                }
            }
        }
    }

    return keypoints;
}
// std::vector<KeyPoint>
// ORBFeatureDetector::calculateFastFeatures(const cv::Mat &aImage) {
//     std::vector<KeyPoint> keypoints;
//     cv::Mat paddedImage;
//     cv::copyMakeBorder(aImage, paddedImage, 3, 3, 3, 3, cv::BORDER_REPLICATE);
//     const uint8_t *data = paddedImage.data;
//     const int step = paddedImage.step;
//     for (int i = 3; i < aImage.rows - 3; ++i) {
//         const uchar *rowPtr = aImage.ptr<uchar>(i);
//         for (int j = 3; j < aImage.cols - 3; ++j) {
//             auto pixel = rowPtr[j];
//             std::uint8_t neg_count{0};
//             std::uint8_t pos_count{0};
//             for (const auto &indice: mFastIndices) {
//                 auto diff = pixel - data[(i + indice[0]) * step + (j + indice[1])];
//                 if (diff < -mFastThreshold) {
//                     neg_count++;
//                     pos_count = 0;
//                 } else if (diff > mFastThreshold) {
//                     neg_count = 0;
//                     pos_count++;
//                 }
//                 if (neg_count == nContinuous || pos_count == nContinuous) {
//                     keypoints.emplace_back(static_cast<std::uint32_t>(j), static_cast<std::uint32_t>(i));
//                     break;
//                 }
//             }
//             for (int q = 0; q < nContinuous - 1; ++q) {
//                 auto indice = mFastIndices[q];
//                 auto diff = pixel - data[(i + indice[0]) * step + (j + indice[1])];
//                 if (diff < -mFastThreshold) {
//                     neg_count++;
//                     pos_count = 0;
//                 } else if (diff > mFastThreshold) {
//                     neg_count = 0;
//                     pos_count++;
//                 }
//                 if (neg_count == nContinuous || pos_count == nContinuous) {
//                     keypoints.emplace_back(static_cast<std::uint32_t>(j), static_cast<std::uint32_t>(i));
//                     break;
//                 }
//             }
//         }
//     }
//     return keypoints;
// }

void ORBFeatureDetector::computeHarrisResponse(const cv::Mat &aImage, std::vector<KeyPoint> &aFeatures,
                                               const std::uint8_t blockSize, const double aHarrisK) {
    cv::Mat Ix, Iy;
    cv::filter2D(aImage, Ix, CV_32F, mGradKernelX);
    cv::filter2D(aImage, Iy, CV_32F, mGradKernelY);

    cv::Mat Ix2, Iy2, Ixy;
    cv::multiply(Ix, Ix, Ix2);
    cv::multiply(Iy, Iy, Iy2);
    cv::multiply(Ix, Iy, Ixy);

    const double scale = std::pow((1.0 / (4.0 * blockSize * 255.0)), 4);

    for (auto &feature: aFeatures) {
        int fx = feature.getImgX();
        int fy = feature.getImgY();

        float sumA{0}, sumB{0}, sumC{0};

        // 7x7 window
        for (int y = -3; y <= 3; ++y) {
            const float *pIx2 = Ix2.ptr<float>(fy + y);
            const float *pIy2 = Iy2.ptr<float>(fy + y);
            const float *pIxy = Ixy.ptr<float>(fy + y);

            for (int x = -3; x <= 3; ++x) {
                if (fx + x < 0 || fx + x >= aImage.cols) continue;

                sumA += pIx2[fx + x];
                sumB += pIy2[fx + x];
                sumC += pIxy[fx + x];
            }
        }

        double score = ((sumA * sumB - sumC * sumC) - aHarrisK * std::pow(sumA + sumB, 2)) * scale;
        feature.setScore(score);
    }
}


void ORBFeatureDetector::addOrientation(const std::vector<cv::Mat> &aPyramid,
                                        std::vector<KeyPoint> &aFeatures) {
    std::vector<std::vector<KeyPoint *> > featuresByLevel(aPyramid.size());
    for (auto &f: aFeatures) {
        featuresByLevel[f.getLevel()].push_back(&f);
    }

    for (int level = 0; level < aPyramid.size(); ++level) {
        const cv::Mat &img = aPyramid[level];
        const auto &levelFeatures = featuresByLevel[level];
        if (levelFeatures.empty()) continue;

        const uint8_t *dataPtr = img.data;
        const size_t step = img.step;
        const int rows = img.rows;
        const int cols = img.cols;

        const int radius = 15;

        for (KeyPoint *f: levelFeatures) {
            int centerX = static_cast<int>(std::round(f->getImgX() * mLevelFactors[level]));
            int centerY = static_cast<int>(std::round(f->getImgY() * mLevelFactors[level]));

            if (centerX < radius || centerX >= cols - radius ||
                centerY < radius || centerY >= rows - radius) {
                continue;
            }

            double m10{0}, m01{0};
            for (const auto &[offX, offY]: mOrientationIndices) {
                // Direct pointer access: base + y_offset + x_offset
                uint8_t pixelVal = dataPtr[(centerY + offY) * step + (centerX + offX)];

                m10 += offX * pixelVal;
                m01 += offY * pixelVal;
            }
            f->setAngle(std::atan2(static_cast<float>(m01), static_cast<float>(m10)));
        }
    }
}

void ORBFeatureDetector::rescaleFeatures(std::vector<KeyPoint> &aFeatures, const double aScale,
                                         const std::uint32_t aLevel) {
    for (auto &feature: aFeatures) {
        feature.scaleBy(aScale, aScale);
        feature.setLevel(aLevel);
    }
}
