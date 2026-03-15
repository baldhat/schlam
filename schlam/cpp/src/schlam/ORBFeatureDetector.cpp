//
// Created by baldhat on 2/7/26.
//

#include "ORBFeatureDetector.h"

#include <opencv2/opencv.hpp>

#include "plotting/plotter.hpp"

#include <algorithm>
#include <bitset>
#include <execution>
#include <vector>

#include "orb_pattern.h"

ORBFeatureDetector::ORBFeatureDetector(const std::uint32_t aNumFeatures,
                                       std::shared_ptr<Plotter> apPlotter,
                                       const std::uint8_t aNumLevels,
                                       const double aLevelFactor) : mNumFeatures(aNumFeatures), mNumLevels(aNumLevels),
                                                                    mLevelFactor(aLevelFactor), mpPlotter(apPlotter) {
    for (std::uint32_t i = 0; i < mNumLevels; i++) {
        mLevelFactors.push_back(std::pow(mLevelFactor, i));
    }
    mOrientationIndices = getPointsInRadius(mOrientationRadius);

    distributeFeaturesToLevels();
}

void ORBFeatureDetector::detectFeatures(Frame &aFrame) {
    auto pyramid = buildPyramid(aFrame.mImage, mNumLevels, mLevelFactor);
    std::unique_ptr<QuadTreeNode> tree = calcFeatures(pyramid);
    addOrientation(pyramid, tree.get());
    addDescriptors(pyramid, tree.get());
    aFrame.mKeypointTree = std::move(tree);
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

std::unique_ptr<QuadTreeNode> ORBFeatureDetector::calcFeatures(const std::vector<cv::Mat> &aPyramid) {
    std::mutex mut;
    std::vector<KeyPoint> features;
    std::for_each(std::execution::par, mLevels.begin(), mLevels.end(), [&](std::uint8_t level) {
        const auto levelImageHeight{aPyramid[level].rows}, levelImageWidth{aPyramid[level].cols};
        auto levelFeatures = calculateFastFeatures(aPyramid[level]);
        levelFeatures = removeAtImageBorder(levelFeatures, levelImageWidth, levelImageHeight, 16);
        computeHarrisResponse(aPyramid[level], levelFeatures, 7);
        //levelFeatures = filterWithOctree(levelFeatures, levelImageWidth, levelImageHeight, mNumFeaturesPerLevel[level]);
        rescaleFeatures(levelFeatures, 1.0 / mLevelFactors[level], level);
        std::lock_guard guard(mut);
        features.insert(features.end(), levelFeatures.begin(), levelFeatures.end());
    });
    auto tree = filterWithOctree(features, aPyramid[0].cols, aPyramid[0].rows, mNumFeatures);
    auto treeFeatures = tree->getFeatures();
    std::cout << "Returning " << treeFeatures.size() << " features" << std::endl;
    mpPlotter->plotFeatures(aPyramid[0], treeFeatures);
    return std::move(tree);
}

void ORBFeatureDetector::addDescriptors(
    const std::vector<cv::Mat> &aPyramid,
    QuadTreeNode* aTree) {
    const size_t nLevels = aPyramid.size();

    std::vector<cv::Mat> blurredPyramid(nLevels);
    for (size_t lvl = 0; lvl < nLevels; ++lvl) {
        cv::GaussianBlur(aPyramid[lvl],
                         blurredPyramid[lvl],
                         cv::Size(7, 7),
                         2, 2,
                         cv::BORDER_REPLICATE);
    }

    for (auto &feature: aTree->getFeatures()) {
        const int level = feature->getLevel();
        const double factor = mLevelFactors[level];

        const cv::Mat &img = blurredPyramid[level];
        const uint8_t *imgData = img.data;
        const int step = img.step;

        // Faster than std::round
        const int centerX = int(feature->getImgX() * factor + 0.5);
        const int centerY = int(feature->getImgY() * factor + 0.5);

        const float angle = static_cast<float>(feature->getAngle());
        const float sinA = std::sin(angle);
        const float cosA = std::cos(angle);

        uint8_t desc[32] = {0};

        const uint8_t *centerPtr = imgData + centerY * step + centerX;

        for (int i = 0; i < 256; ++i) {
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

        feature->setDescriptor(std::to_array(desc));
    }
}


std::unique_ptr<QuadTreeNode> ORBFeatureDetector::filterWithOctree(std::vector<KeyPoint> &aFeatures,
                                                                   const std::uint32_t aWidth,
                                                                   const std::uint32_t aHeight,
                                                                   const std::uint32_t aNumFeatures) {
    std::sort(aFeatures.begin(), aFeatures.end(),
        [](KeyPoint &a, KeyPoint &b) { return a.getScore() > b.getScore(); }
    );
    auto root = std::make_unique<QuadTreeNode>(0, aWidth, 0, aHeight, aFeatures);
    if (aNumFeatures > aFeatures.size()) {
        return std::move(root);
    }
    std::vector<QuadTreeNode *> nodes;
    nodes.push_back(root.get());
    while (nodes.size() < aNumFeatures) {
        std::vector<QuadTreeNode *> newNodes;
        for (auto &node: nodes) {
            if (node->isLeaf()) {
                newNodes.push_back(node);
                continue;
            }
            node->divide();
            for (auto children = node->getChildren(); auto &child: children) {
                if (!child->isLeaf()) {
                    newNodes.push_back(child);
                }
            }
        }
        nodes = newNodes;
    }

    // Each leaf might have multiple key points right now, so only keep the best feature per leaf
    root->retainBestFeaturesPerLeaf();

    // We might have more leaves than we need, so prune the worst ones
    retainTopN(root.get(), aNumFeatures);
    return std::move(root);
}

void ORBFeatureDetector::retainTopN(QuadTreeNode *aTree, std::uint32_t aTopN) {
    std::vector<QuadTreeNode *> nodes;
    nodes.push_back(aTree);
    bool finished = false;
    while (!finished) {
        if (nodes.size() > aTopN) {
            std::nth_element(nodes.begin(), nodes.begin() + aTopN, nodes.end(),
                [](auto a, auto b) { return a->getMaxScore() > b->getMaxScore(); });
            for (auto it = nodes.begin() + aTopN; it != nodes.end(); ++it) {
                auto node = *it;
                auto parent = node->getParent();
                parent->removeChild(node);
            }
            nodes.erase(nodes.begin() + aTopN, nodes.end());
        } else {
            std::size_t numLeaves = 0;
            std::vector<QuadTreeNode *> newNodes;
            for (auto &node: nodes) {
                if (node->isLeaf()) {
                    numLeaves++;
                    newNodes.push_back(node);
                    continue;
                }
                for (auto children = node->getChildren(); auto &child: children) {
                    newNodes.push_back(child);
                }
            }
            if (numLeaves == nodes.size()) {
                finished = true;
            }
            nodes = newNodes;
        }
    }
}


std::vector<KeyPoint>
ORBFeatureDetector::calculateFastFeatures(const cv::Mat &aImage) {
    std::vector<KeyPoint> keypoints;
    keypoints.reserve(aImage.rows * aImage.cols / 10);

    const int rows = aImage.rows;
    const int cols = aImage.cols;
    const int step = aImage.step;

    const uint8_t *imgData = aImage.data;

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

            if (brighter < 3 && darker < 3)
                continue;

            int pos_count = 0;
            int neg_count = 0;

            // Loop 16 + (nContinuous - 1) for wrap-around
            for (int k = 0; k < 16 + nContinuous - 1; ++k) {
                const int idx = k < 16 ? k : k - 16;

                p = *(centerPtr + offsets[idx]);

                const bool isBright = (p > t_high);
                const bool isDark = (p < t_low);

                pos_count = isBright ? pos_count + 1 : 0;
                neg_count = isDark ? neg_count + 1 : 0;

                if (pos_count >= nContinuous || neg_count >= nContinuous) {
                    keypoints.emplace_back(static_cast<uint32_t>(j),
                                           static_cast<uint32_t>(i));
                    break;
                }
            }
        }
    }

    return keypoints;
}

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
                                        QuadTreeNode* aTree) {
    std::vector<std::vector<KeyPoint *> > featuresByLevel(aPyramid.size());
    for (auto &f: aTree->getFeatures()) {
        featuresByLevel[f->getLevel()].push_back(f);
    }

    for (int level = 0; level < aPyramid.size(); ++level) {
        const cv::Mat &img = aPyramid[level];
        const auto &levelFeatures = featuresByLevel[level];
        if (levelFeatures.empty()) continue;

        const uint8_t *dataPtr = img.data;
        const size_t step = img.step;
        const int rows = img.rows;
        const int cols = img.cols;

        for (KeyPoint *f: levelFeatures) {
            int centerX = static_cast<int>(std::round(f->getImgX() * mLevelFactors[level]));
            int centerY = static_cast<int>(std::round(f->getImgY() * mLevelFactors[level]));

            if (centerX < mOrientationRadius || centerX > (cols - mOrientationRadius) ||
                centerY < mOrientationRadius || centerY > (rows - mOrientationRadius)) {
                continue;
            }

            double m10{0}, m01{0};
            for (const auto &[offX, offY]: mOrientationIndices) {
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
