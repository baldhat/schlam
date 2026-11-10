//
// Created by baldhat on 2/7/26.
//

#include "ORBFeatureDetector.h"

#include <opencv2/opencv.hpp>

#include "plotting/IPlotter.hpp"
#include "FastFeatureDetector.h"
#include "FeatureSelection.h"
#include "ORBDescriptor.h"
#include "../KeyPoint.h"
#include "../helpers/utils.h"
#include "../Frame.h"

#include <algorithm>
#include <cstddef>
#include <execution>
#include <vector>
#include <tracy/Tracy.hpp>

namespace {

std::vector<int> orientationHalfWidths(std::uint32_t radius) {
    std::vector<int> widths(radius + 1);
    auto x = static_cast<std::int64_t>(radius);
    const auto radiusSquared = x * x;
    for (std::size_t y = 0; y < widths.size(); ++y) {
        const auto dy = static_cast<std::int64_t>(y);
        while (x * x + dy * dy > radiusSquared) --x;
        widths[y] = static_cast<int>(x);
    }
    return widths;
}

}

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
    mOrientationHalfWidths = orientationHalfWidths(mOrientationRadius);

    distributeFeaturesToLevels();
}

ORBFeatureDetector::ORBFeatureDetector(const schlam::config::Features& settings,
                                       std::shared_ptr<IPlotter> plotter)
    : ORBFeatureDetector(settings.count, std::move(plotter), settings.levels, 1.0 / settings.scaleFactor) {
    mSettings = settings;
    mFastFeatureDetector = schlam::FastFeatureDetector(settings.fastThreshold, settings.fastArcLength);
    mOrientationRadius = settings.orientationRadius;
    mOrientationHalfWidths = orientationHalfWidths(mOrientationRadius);
}

void ORBFeatureDetector::detectFeatures(Frame &aFrame) {
    ZoneScoped;
    auto pyramid = buildPyramid(aFrame.mImage, mNumLevels, mLevelFactor);
    std::unique_ptr<QuadTreeNode> tree = calcFeatures(pyramid);
    const auto features = tree->getFeatures();
    addOrientation(pyramid, features);
    addDescriptors(pyramid, features);
    if (mpPlotter) mpPlotter->plotFeatures(aFrame.mImage, features, *tree);
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
        if (mNumFeaturesPerLevel[level] == 0) return;
        const auto levelImageHeight{aPyramid[level].rows}, levelImageWidth{aPyramid[level].cols};
        auto levelFeatures = mFastFeatureDetector.calculateFastFeatures(aPyramid[level]);
        levelFeatures = removeAtImageBorder(levelFeatures, levelImageWidth, levelImageHeight, mSettings.border);
        computeHarrisResponse(aPyramid[level], levelFeatures, mSettings.harrisBlockSize, mSettings.harrisK);
        auto levelTree = filterWithOctree(levelFeatures, levelImageWidth, levelImageHeight,
            mNumFeaturesPerLevel[level]);
        featuresByLevel[level] = levelTree->getFeatures();
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
    auto tree = std::make_unique<QuadTreeNode>(0, image.cols, 0, image.rows,
        std::move(features), nullptr);
    // Index the selected features without pruning another levels quota
    // Different levels can contribute features at the same full-resolution position
    std::vector<QuadTreeNode*> pending{tree.get()};
    while (!pending.empty()) {
        auto* node = pending.back();
        pending.pop_back();
        node->divide();
        node->forEachChild([&](auto* child) { pending.push_back(child); });
    }
    return tree;
}

void ORBFeatureDetector::addDescriptors(
    const std::vector<cv::Mat> &aPyramid,
    const std::vector<std::shared_ptr<KeyPoint>>& features) {
    ZoneScoped;
    if (features.empty()) return;
    const size_t nLevels = aPyramid.size();

    std::vector<bool> usedLevels(nLevels);
    for (const auto& feature : features) usedLevels[feature->getLevel()] = true;
    std::vector<cv::Mat> blurredPyramid(nLevels);
    for (size_t lvl = 0; lvl < nLevels; ++lvl) {
        if (!usedLevels[lvl]) continue;
        ZoneScoped;
        cv::GaussianBlur(aPyramid[lvl],
                         blurredPyramid[lvl],
                         cv::Size(mSettings.descriptorBlurSize, mSettings.descriptorBlurSize),
                         mSettings.descriptorBlurSigma, mSettings.descriptorBlurSigma,
                         cv::BORDER_REPLICATE);
    }


    std::for_each(std::execution::par_unseq, features.begin(), features.end(), [&](const auto& feature) {
        const int level = feature->getLevel();
        const double factor = mLevelFactors[level];

        const cv::Mat &img = blurredPyramid[level];
        const uint8_t *imgData = img.data;
        const auto step = static_cast<std::ptrdiff_t>(img.step);

        const int centerX = int(feature->getImgX() * factor + 0.5);
        const int centerY = int(feature->getImgY() * factor + 0.5);

        const double angle = static_cast<double>(feature->getAngle());
        const double sinA = std::sin(angle);
        const double cosA = std::cos(angle);

        Descriptor desc{};
        const uint8_t *centerPtr = imgData + centerY * step + centerX;
        std::array<int, 1024> rotated;
        schlam::detail::rotateOrbPattern(sinA, cosA, rotated);
        for (std::size_t byte = 0; byte < desc.size(); ++byte) {
            unsigned bits = 0;
            for (int bit = 0; bit < 8; ++bit) {
                const auto idx = byte * 32 + bit * 4;
                const auto p0 = centerPtr[rotated[idx + 1] * step + rotated[idx]];
                const auto p1 = centerPtr[rotated[idx + 3] * step + rotated[idx + 2]];
                bits |= (p0 < p1) << bit;
            }
            desc[byte] = static_cast<std::uint8_t>(bits);
        }

        feature->setDescriptor(desc);
    });
}


std::unique_ptr<QuadTreeNode> ORBFeatureDetector::filterWithOctree(std::vector<KeyPoint>& aFeatures,
                                                                   const std::uint32_t aWidth,
                                                                   const std::uint32_t aHeight,
                                                                   const std::uint32_t aNumFeatures) {
    ZoneScoped;
    const auto selected = schlam::detail::selectFeatureIndices(aFeatures, aWidth, aHeight, aNumFeatures);
    std::vector<std::shared_ptr<KeyPoint>> features;
    features.reserve(selected.size());
    for (auto index : selected) features.push_back(std::make_shared<KeyPoint>(aFeatures[index]));
    // Only the merged, full-resolution tree needs a spatial index.
    return std::make_unique<QuadTreeNode>(0, aWidth, 0, aHeight, std::move(features), nullptr);
}

void ORBFeatureDetector::computeHarrisResponse(const cv::Mat& aImage, std::vector<KeyPoint>& aFeatures,
                                               const int blockSize, const double aHarrisK) {
    ZoneScoped;
    if (aFeatures.empty()) return;
    cv::Mat Ix, Iy;
    cv::filter2D(aImage, Ix, CV_32F, mGradKernelX);
    cv::filter2D(aImage, Iy, CV_32F, mGradKernelY);

    // Fuse gradient products and prefix sums; each candidate then needs four
    // lookups, independent of block size. Double sums preserve integer gradients.
    cv::Mat integral(aImage.rows + 1, aImage.cols + 1, CV_64FC3);
    integral.row(0).setTo(cv::Scalar(0, 0, 0));
    for (int y = 0; y < aImage.rows; ++y) {
        const auto* dx = Ix.ptr<float>(y);
        const auto* dy = Iy.ptr<float>(y);
        const auto* previous = integral.ptr<cv::Vec3d>(y);
        auto* row = integral.ptr<cv::Vec3d>(y + 1);
        cv::Vec3d sum(0, 0, 0);
        row[0] = sum;
        for (int x = 0; x < aImage.cols; ++x) {
            // Match the original float products before accumulating in double.
            sum += cv::Vec3d(dx[x] * dx[x], dy[x] * dy[x], dx[x] * dy[x]);
            row[x + 1] = previous[x + 1] + sum;
        }
    }
    const double scale = std::pow(1.0 / (4.0 * blockSize * 255.0), 4);
    const int radius = blockSize / 2;
    for (auto& feature : aFeatures) {
        const int fx = static_cast<int>(std::lround(feature.getImgX()));
        const int fy = static_cast<int>(std::lround(feature.getImgY()));
        const int left = std::clamp(fx - radius, 0, aImage.cols);
        const int right = std::clamp(fx + radius + 1, 0, aImage.cols);
        const int top = std::clamp(fy - radius, 0, aImage.rows);
        const int bottom = std::clamp(fy + radius + 1, 0, aImage.rows);
        const auto* upper = integral.ptr<cv::Vec3d>(top);
        const auto* lower = integral.ptr<cv::Vec3d>(bottom);
        const auto sum = (lower[right] - lower[left]) - (upper[right] - upper[left]);
        const double trace = sum[0] + sum[1];
        feature.setScore(((sum[0] * sum[1] - sum[2] * sum[2]) - aHarrisK * (trace * trace)) * scale);
    }
}

void ORBFeatureDetector::addOrientation(const std::vector<cv::Mat>& aPyramid,
                                        const std::vector<std::shared_ptr<KeyPoint>>& features) {
    ZoneScoped;
    if (features.empty()) return;
    std::vector<std::vector<std::shared_ptr<KeyPoint>>> featuresByLevel(aPyramid.size());
    for (const auto& feature : features) featuresByLevel[feature->getLevel()].push_back(feature);
    const int radius = static_cast<int>(mOrientationRadius);
    std::vector<std::ptrdiff_t> rowOffsets(2 * radius + 1);
    for (std::size_t level = 0; level < aPyramid.size(); ++level) {
        if (featuresByLevel[level].empty()) continue;
        const auto& image = aPyramid[level];
        const auto step = static_cast<std::ptrdiff_t>(image.step);
        for (int y = -radius; y <= radius; ++y) rowOffsets[y + radius] = y * step;
        for (const auto& feature : featuresByLevel[level]) {
            const int centerX = static_cast<int>(std::round(feature->getImgX() * mLevelFactors[level]));
            const int centerY = static_cast<int>(std::round(feature->getImgY() * mLevelFactors[level]));
            if (centerX < radius || centerX >= image.cols - radius ||
                centerY < radius || centerY >= image.rows - radius) continue;

            const auto* center = image.data + centerY * step + centerX;
            std::int64_t m10 = 0, m01 = 0;
            for (int y = -radius; y <= radius; ++y) {
                const auto* row = center + rowOffsets[y + radius];
                const int halfWidth = mOrientationHalfWidths[std::abs(y)];
                std::int64_t rowSum = 0;
                for (int x = -halfWidth; x <= halfWidth; ++x) {
                    const auto pixel = row[x];
                    rowSum += pixel;
                    m10 += x * pixel;
                }
                m01 += y * rowSum;
            }
            feature->setAngle(std::atan2(static_cast<double>(m01), static_cast<double>(m10)));
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
