//
// Created by baldhat on 2/7/26.
//

#include "ORBFeatureDetector.h"

#include <opencv2/opencv.hpp>

#include "plotting/IPlotter.hpp"
#include "FastFeatureDetector.h"
#include "../../KeyPoint.h"
#include "../../helpers/utils.h"
#include "../../Frame.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <execution>
#include <numeric>
#include <vector>
#include <tracy/Tracy.hpp>

#include "orb_pattern.h"
#include "OrbDescriptor.h"

namespace {

constexpr auto descriptorPattern = [] {
    constexpr auto sampleCount = gOrbBitPattern31.size() / 2;
    struct Pattern {
        std::array<std::array<std::int8_t, 2>, sampleCount> offsets{};
        std::array<std::uint16_t, sampleCount> sampleIndices{};
        std::size_t uniqueCount = 0;
    } pattern;
    // Index signed byte coordinate pairs by their unsigned representations.
    std::array<std::uint16_t, 256 * 256> indexByOffset{};
    for (std::size_t i = 0; i < sampleCount; ++i) {
        const auto x = gOrbBitPattern31[2 * i];
        const auto y = gOrbBitPattern31[2 * i + 1];
        const auto key = 256 * static_cast<std::uint8_t>(x) + static_cast<std::uint8_t>(y);
        auto& index = indexByOffset[key];
        if (index == 0) {
            pattern.offsets[pattern.uniqueCount++] = {x, y};
            index = static_cast<std::uint16_t>(pattern.uniqueCount);
        }
        pattern.sampleIndices[i] = index - 1;
    }
    return pattern;
}();

} // namespace

ORBFeatureDetector::ORBFeatureDetector(const std::uint32_t aNumFeatures,
                                       std::shared_ptr<IPlotter> apPlotter,
                                       const std::uint8_t aNumLevels,
                                       const double aLevelFactor) : mNumFeatures(aNumFeatures), mNumLevels(aNumLevels),
                                                                    mLevelFactor(aLevelFactor), mpPlotter(apPlotter) {
    mSettings.count = aNumFeatures;
    mSettings.levels = aNumLevels;
    mSettings.scaleFactor = 1.0 / aLevelFactor;
    for (std::uint32_t i = 0; i < mNumLevels; i++) {
        mLevelFactors.push_back(std::pow(mLevelFactor, i));
    }
    mOrientationIndices = getPointsInRadius(mOrientationRadius);

    distributeFeaturesToLevels();
}

ORBFeatureDetector::ORBFeatureDetector(const schlam::config::Features& settings,
                                       std::shared_ptr<IPlotter> plotter)
    : ORBFeatureDetector(settings.count, std::move(plotter), settings.levels, 1.0 / settings.scaleFactor) {
    mSettings = settings;
    mFastFeatureDetector = schlam::FastFeatureDetector(settings.fastThreshold, settings.fastArcLength);
    mOrientationRadius = settings.orientationRadius;
    mOrientationIndices = getPointsInRadius(mOrientationRadius);
}

void ORBFeatureDetector::extractFeatures(Frame &aFrame) {
    ZoneScoped;
    auto pyramid = buildPyramid(aFrame.mImage, mNumLevels, mLevelFactor);
    std::unique_ptr<QuadTreeNode> tree = calcFeatures(pyramid);
    addOrientation(pyramid, tree.get());
    addDescriptors(pyramid, tree.get());
    aFrame.mKeypointTree = std::move(tree);
}

void ORBFeatureDetector::distributeFeaturesToLevels() {
    auto featuresPerScale = mNumFeatures * (1 - mLevelFactor) / (1 - std::pow(mLevelFactor, mNumLevels));
    std::uint32_t numFeatures{0};
    for (std::uint8_t level = 0; level < mNumLevels - 1; level++) {
        mLevels.push_back(level);
        const auto levelBudget = static_cast<std::uint32_t>(std::round(featuresPerScale * mLevelFactors[level]));
        mNumFeaturesPerLevel.push_back(std::min(levelBudget, mNumFeatures - numFeatures));
        numFeatures += mNumFeaturesPerLevel[level];
    }
    mLevels.push_back(mNumLevels - 1);
    mNumFeaturesPerLevel.push_back(mNumFeatures - numFeatures);
}

std::unique_ptr<QuadTreeNode> ORBFeatureDetector::calcFeatures(const std::vector<cv::Mat> &aPyramid) {
    ZoneScoped;
    std::vector<std::vector<std::shared_ptr<KeyPoint>>> featuresByLevel(mNumLevels);
    std::for_each(std::execution::par, mLevels.begin(), mLevels.end(), [&](std::uint8_t level) {
        const auto levelImageHeight{aPyramid[level].rows}, levelImageWidth{aPyramid[level].cols};
        auto levelFeatures = mFastFeatureDetector.calculateFastFeatures(aPyramid[level]);
        levelFeatures = removeAtImageBorder(levelFeatures, levelImageWidth, levelImageHeight, mSettings.border);
        computeHarrisResponse(aPyramid[level], levelFeatures, mSettings.harrisBlockSize, mSettings.harrisK);
        featuresByLevel[level] = selectLevelFeatures(levelFeatures, levelImageWidth, levelImageHeight,
            mNumFeaturesPerLevel[level]);
        rescaleFeatures(featuresByLevel[level], 1.0 / mLevelFactors[level], level);
    });
    return mergeFeatureLevels(featuresByLevel, aPyramid[0]);
}

std::unique_ptr<QuadTreeNode> ORBFeatureDetector::mergeFeatureLevels(
    const std::vector<std::vector<std::shared_ptr<KeyPoint>>>& featuresByLevel,
    const cv::Mat& image) {
    ZoneScoped;
    std::vector<std::shared_ptr<KeyPoint>> features;
    features.reserve(mNumFeatures);
    for (const auto& levelFeatures : featuresByLevel) {
        features.insert(features.end(), levelFeatures.begin(), levelFeatures.end());
    }
    std::sort(features.begin(), features.end(), [](const auto& a, const auto& b) {
        return a->getScore() > b->getScore();
    });
    auto tree = std::make_unique<QuadTreeNode>(0, image.cols, 0, image.rows, std::move(features), nullptr);
    // Index the selected features without pruning another levels quota
    // Different levels can contribute features at the same full-resolution position
    std::vector<QuadTreeNode*> pending{tree.get()};
    while (!pending.empty()) {
        auto* node = pending.back();
        pending.pop_back();
        node->divide();
        const auto children = node->getChildren();
        pending.insert(pending.end(), children.begin(), children.end());
    }
    auto treeFeatures = tree->getFeatures();
    if (mpPlotter) {
        mpPlotter->plotFeatures(image, treeFeatures, *tree);
    }
    return tree;
}

void ORBFeatureDetector::addDescriptors(
    const std::vector<cv::Mat> &aPyramid,
    QuadTreeNode* aTree) {
    ZoneScoped;
    const size_t nLevels = aPyramid.size();

    std::vector<cv::Mat> blurredPyramid(nLevels);
    for (size_t lvl = 0; lvl < nLevels; ++lvl) {
        ZoneScoped;
        cv::GaussianBlur(aPyramid[lvl],
                         blurredPyramid[lvl],
                         cv::Size(mSettings.descriptorBlurSize, mSettings.descriptorBlurSize),
                         mSettings.descriptorBlurSigma, mSettings.descriptorBlurSigma,
                         cv::BORDER_REPLICATE);
    }


    auto features = aTree->getFeatures();
    std::for_each(std::execution::par_unseq,  features.begin(),  features.end(), [&](auto& feature) {
        const int level = feature->getLevel();
        const double factor = mLevelFactors[level];

        const cv::Mat &img = blurredPyramid[level];
        const uint8_t *imgData = img.data;
        const int step = img.step;

        const int centerX = int(feature->getImgX() * factor + 0.5);
        const int centerY = int(feature->getImgY() * factor + 0.5);

        const double angle = static_cast<double>(feature->getAngle());
        const double sinA = std::sin(angle);
        const double cosA = std::cos(angle);

        uint8_t desc[32] = {0};

        const uint8_t *centerPtr = imgData + centerY * step + centerX;
        std::array<std::uint8_t, descriptorPattern.uniqueCount> samples;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const auto [x, y] = descriptorPattern.offsets[i];
            const int rx = std::lround(x * cosA - y * sinA);
            const int ry = std::lround(x * sinA + y * cosA);
            samples[i] = *(centerPtr + ry * step + rx);
        }

        for (int i = 0; i < 256; ++i) {
            const auto p0 = samples[descriptorPattern.sampleIndices[2 * i]];
            const auto p1 = samples[descriptorPattern.sampleIndices[2 * i + 1]];
            desc[i >> 3] |= (p0 < p1) << (i & 7);
        }

        feature->setDescriptor(std::make_shared<OrbDescriptor>(std::to_array(desc)));
    });
}


std::vector<std::shared_ptr<KeyPoint>> ORBFeatureDetector::selectLevelFeatures(
    std::vector<KeyPoint>& aFeatures, std::uint32_t aWidth, std::uint32_t aHeight,
    std::uint32_t aNumFeatures) {
    ZoneScoped;
    if (aNumFeatures == 0 || aFeatures.empty()) return {};

    std::sort(aFeatures.begin(), aFeatures.end(), [](const KeyPoint& a, const KeyPoint& b) {
        return a.getScore() > b.getScore();
    });

    std::vector<std::shared_ptr<KeyPoint>> selected;
    const auto limit = std::min<std::size_t>(aNumFeatures, aFeatures.size());
    selected.reserve(limit);
    if (aFeatures.size() <= limit || limit == 1) {
        for (std::size_t i = 0; i < limit; ++i) {
            selected.push_back(std::make_shared<KeyPoint>(aFeatures[i]));
        }
        return selected;
    }

    struct Leaf {
        QuadTreeNode::Bounds bounds;
        std::size_t begin, end;
    };
    std::vector<std::size_t> indices(aFeatures.size());
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    std::vector<std::size_t> scratch(indices.size());
    std::vector<std::uint8_t> quadrants(indices.size());
    std::vector<Leaf> leaves{{{0, static_cast<double>(aWidth), 0, static_cast<double>(aHeight)},
        0, indices.size()}};
    std::vector<Leaf> nextLeaves;
    while (leaves.size() < limit) {
        nextLeaves.clear();
        bool divided = false;
        for (const auto& leaf : leaves) {
            if (leaf.end - leaf.begin < 2 || !QuadTreeNode::canDivide(leaf.bounds)) {
                nextLeaves.push_back(leaf);
                continue;
            }
            divided = true;
            const QuadTreeNode::Split split(leaf.bounds);
            std::array<std::size_t, 5> offsets{leaf.begin};
            for (auto i = leaf.begin; i < leaf.end; ++i) {
                const auto& feature = aFeatures[indices[i]];
                const auto quadrant = split.quadrant(feature.getImgX(), feature.getImgY());
                quadrants[i] = static_cast<std::uint8_t>(quadrant);
                ++offsets[quadrant + 1];
            }
            std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());

            // Preserve score order within each quadrant so its first candidate wins.
            auto write = offsets;
            for (auto i = leaf.begin; i < leaf.end; ++i) {
                scratch[write[quadrants[i]]++] = indices[i];
            }
            std::copy_n(scratch.begin() + leaf.begin, leaf.end - leaf.begin, indices.begin() + leaf.begin);

            for (std::size_t quadrant = 0; quadrant < split.quadrants.size(); ++quadrant) {
                if (offsets[quadrant] == offsets[quadrant + 1]) continue;
                nextLeaves.push_back({split.quadrants[quadrant], offsets[quadrant], offsets[quadrant + 1]});
            }
        }
        if (!divided) break;
        leaves.swap(nextLeaves);
    }

    if (leaves.size() > limit) {
        std::nth_element(leaves.begin(), leaves.begin() + limit, leaves.end(), [&](const Leaf& a, const Leaf& b) {
            return aFeatures[indices[a.begin]].getScore() > aFeatures[indices[b.begin]].getScore();
        });
        leaves.resize(limit);
        // Subdivision laid out the ranges in traversal order before nth_element reordered them.
        std::sort(leaves.begin(), leaves.end(), [](const Leaf& a, const Leaf& b) { return a.begin < b.begin; });
    }
    for (const auto& leaf : leaves) {
        selected.push_back(std::make_shared<KeyPoint>(aFeatures[indices[leaf.begin]]));
    }
    return selected;
}


void ORBFeatureDetector::computeHarrisResponse(const cv::Mat &aImage, std::vector<KeyPoint> &aFeatures,
                                               const int blockSize, const double aHarrisK) {
    ZoneScoped;
    const double scale = std::pow((1.0 / (4.0 * blockSize * 255.0)), 4);
    const int radius = blockSize / 2;
    const auto scoreFeatures = [&](const auto& rowProducts) {
        for (auto& feature : aFeatures) {
            const int fx = static_cast<int>(std::lround(feature.getImgX()));
            const int fy = static_cast<int>(std::lround(feature.getImgY()));
            const int firstX = std::max(0, fx - radius);
            const int lastX = std::min(aImage.cols - 1, fx + radius);
            double sumA = 0, sumB = 0, sumC = 0;

            for (int y = -radius; y <= radius; ++y) {
                const auto products = rowProducts(fy + y);
                for (int x = firstX; x <= lastX; ++x) {
                    const auto [xx, yy, xy] = products(x);
                    sumA += xx;
                    sumB += yy;
                    sumC += xy;
                }
            }
            const double score = ((sumA * sumB - sumC * sumC) - aHarrisK * std::pow(sumA + sumB, 2)) * scale;
            feature.setScore(score);
        }
    };

    cv::Mat Ix, Iy;
    if (aImage.type() == CV_8UC1 && !aImage.empty() && !aImage.isSubmatrix()) {
        // Byte Sobel gradients and their products fit exactly in int16 and float32, respectively.
        cv::spatialGradient(aImage, Ix, Iy);
        scoreFeatures([&](int row) {
            const auto* dx = Ix.ptr<std::int16_t>(row);
            const auto* dy = Iy.ptr<std::int16_t>(row);
            return [=](int x) {
                const int horizontal = dx[x], vertical = dy[x];
                return std::array{horizontal * horizontal, vertical * vertical, horizontal * vertical};
            };
        });
        return;
    }

    cv::filter2D(aImage, Ix, CV_32F, mGradKernelX);
    cv::filter2D(aImage, Iy, CV_32F, mGradKernelY);
    cv::Mat Ix2, Iy2, Ixy;
    cv::multiply(Ix, Ix, Ix2);
    cv::multiply(Iy, Iy, Iy2);
    cv::multiply(Ix, Iy, Ixy);
    scoreFeatures([&](int row) {
        const auto* xx = Ix2.ptr<float>(row);
        const auto* yy = Iy2.ptr<float>(row);
        const auto* xy = Ixy.ptr<float>(row);
        return [=](int x) { return std::array{xx[x], yy[x], xy[x]}; };
    });
}

void ORBFeatureDetector::addOrientation(const std::vector<cv::Mat> &aPyramid,
                                        QuadTreeNode* aTree) {
    ZoneScoped;
    std::vector<std::vector<std::shared_ptr<KeyPoint>>> featuresByLevel(aPyramid.size());
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

        for (const auto& f: levelFeatures) {
            int centerX = static_cast<int>(std::round(f->getImgX() * mLevelFactors[level]));
            int centerY = static_cast<int>(std::round(f->getImgY() * mLevelFactors[level]));

            const int radius = static_cast<int>(mOrientationRadius);
            if (centerX < radius || centerX >= cols - radius ||
                centerY < radius || centerY >= rows - radius) {
                continue;
            }

            double m10{0}, m01{0};
            for (const auto &[offX, offY]: mOrientationIndices) {
                uint8_t pixelVal = dataPtr[(centerY + offY) * step + (centerX + offX)];
                m10 += offX * pixelVal;
                m01 += offY * pixelVal;
            }
            f->setAngle(std::atan2(static_cast<double>(m01), static_cast<double>(m10)));
        }
    }
}

void ORBFeatureDetector::rescaleFeatures(std::vector<std::shared_ptr<KeyPoint>> &aFeatures, const double aScale,
                                         const std::uint32_t aLevel) {
    for (auto &feature: aFeatures) {
        feature->scaleBy(aScale, aScale);
        feature->setLevel(aLevel);
    }
}
