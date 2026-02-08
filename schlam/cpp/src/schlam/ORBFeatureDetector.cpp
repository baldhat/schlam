//
// Created by baldhat on 2/7/26.
//

#include "ORBFeatureDetector.h"

#include <opencv2/opencv.hpp>

#include "src/plotting/plotter.hpp"

#include <algorithm>
#include <execution>
#include <vector>

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
    std::for_each(std::execution::seq, mLevels.begin(), mLevels.end(), [&](std::uint8_t level) {
        auto levelImageWidth{aPyramid[level].cols}, levelImageHeight{aPyramid[level].rows};
        auto levelFeatures = calculateFastFeatures(aPyramid[level]);
        removeAtImageBorder(levelFeatures, levelImageWidth, levelImageHeight, 3);
        computeHarrisResponse(aPyramid[level], levelFeatures, 7);
        levelFeatures = filterWithOctree(levelFeatures, levelImageWidth, levelImageHeight, mNumFeaturesPerLevel[level]);
        //mpPlotter->plotFeatures(aPyramid[level], levelFeatures);
        rescaleFeatures(levelFeatures, 1.0/mLevelFactors[level]);
        features.insert(features.end(), levelFeatures.begin(), levelFeatures.end());
    });
    std::cout << "Returning " << features.size() << " features" << std::endl;
    mpPlotter->plotFeatures(aPyramid[0], features);
    return features;
}

void ORBFeatureDetector::addDescriptors(const std::vector<cv::Mat>& aPyramid, std::vector<KeyPoint>& aFeatures) {

}

std::vector<KeyPoint> ORBFeatureDetector::filterWithOctree(std::vector<KeyPoint> &aFeatures, const std::uint32_t aWidth,
                                                           const std::uint32_t aHeight,
                                                           const std::uint32_t aNumFeatures) {
    auto node = std::make_shared<QuadTreeNode>(0, aWidth, 0, aHeight, aFeatures);
    if (aNumFeatures > aFeatures.size()) {
        return aFeatures;
    }
    std::vector<std::shared_ptr<QuadTreeNode>> nodes;
    nodes.push_back(node);
    while (nodes.size() < aNumFeatures) {
        std::vector<std::shared_ptr<QuadTreeNode>> newNodes;
        for (auto& node : nodes) {
            if (node->isLeaf()) {
                newNodes.push_back(node);
                continue;
            }
            auto children = node->divide();
            for (auto& child : children) {
                if (!child->isLeaf()) {
                    newNodes.push_back(child);
                }
            }
        }
        nodes = newNodes;
    }

    std::vector<KeyPoint> features;
    for (auto& node : nodes) {
        features.push_back(node->getMaxFeature());
    }
    return features;
}

std::vector<KeyPoint>
ORBFeatureDetector::calculateFastFeatures(const cv::Mat &aImage) {
    std::vector<KeyPoint> keypoints;
    cv::Mat paddedImage;
    cv::copyMakeBorder(aImage, paddedImage, 3, 3, 3, 3, cv::BORDER_REPLICATE);
    const uint8_t* data = paddedImage.data;
    const int step = paddedImage.step;
    for (int i = 3; i < aImage.rows - 3; ++i) {
        const uchar *rowPtr = aImage.ptr<uchar>(i);
        for (int j = 3; j < aImage.cols - 3; ++j) {
            auto pixel = rowPtr[j];
            std::uint8_t neg_count{0};
            std::uint8_t pos_count{0};
            for (const auto &indice: mFastIndices) {
                auto diff = pixel - data[(i + indice[0]) * step + (j + indice[1])];
                if (diff < -mFastThreshold) {
                    neg_count++;
                    pos_count = 0;
                } else if (diff > mFastThreshold) {
                    neg_count = 0;
                    pos_count++;
                }
                if (neg_count == nContinuous || pos_count == nContinuous) {
                    keypoints.push_back({static_cast<std::uint32_t>(j), static_cast<std::uint32_t>(i)});
                    break;
                }
            }
            for (int q = 0; q < nContinuous - 1; ++q) {
                auto indice = mFastIndices[q];
                auto diff = pixel - data[(i + indice[0]) * step + (j + indice[1])];
                if (diff < -mFastThreshold) {
                    neg_count++;
                    pos_count = 0;
                } else if (diff > mFastThreshold) {
                    neg_count = 0;
                    pos_count++;
                }
                if (neg_count == nContinuous || pos_count == nContinuous) {
                    keypoints.push_back({static_cast<std::uint32_t>(j), static_cast<std::uint32_t>(i)});
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

    for (auto &feature : aFeatures) {
        int fx = feature.getImgX();
        int fy = feature.getImgY();

        float sumA{0}, sumB{0}, sumC{0};

        // 7x7 window
        for (int y = -3; y <= 3; ++y) {

            const float* pIx2 = Ix2.ptr<float>(fy + y);
            const float* pIy2 = Iy2.ptr<float>(fy + y);
            const float* pIxy = Ixy.ptr<float>(fy + y);

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
    std::vector<std::vector<KeyPoint*>> featuresByLevel(aPyramid.size());
    for (auto &f : aFeatures) {
        featuresByLevel[f.getLevel()].push_back(&f);
    }

    for (int level = 0; level < aPyramid.size(); ++level) {
        const cv::Mat &img = aPyramid[level];
        const auto &levelFeatures = featuresByLevel[level];
        if (levelFeatures.empty()) continue;

        const uint8_t* dataPtr = img.data;
        const size_t step = img.step;
        const int rows = img.rows;
        const int cols = img.cols;

        const int radius = 15;

        for (KeyPoint* f : levelFeatures) {
            int centerX = static_cast<int>(std::round(f->getImgX() * mLevelFactors[level]));
            int centerY = static_cast<int>(std::round(f->getImgY() * mLevelFactors[level]));

            if (centerX < radius || centerX >= cols - radius ||
                centerY < radius || centerY >= rows - radius) {
                continue;
            }

            double m10{0}, m01{0};
            for (const auto &[offX, offY] : mOrientationIndices) {
                // Direct pointer access: base + y_offset + x_offset
                uint8_t pixelVal = dataPtr[(centerY + offY) * step + (centerX + offX)];

                m10 += offX * pixelVal;
                m01 += offY * pixelVal;
            }
            f->setAngle(std::atan2(static_cast<float>(m01), static_cast<float>(m10)));
        }
    }
}

void ORBFeatureDetector::rescaleFeatures(std::vector<KeyPoint>& aFeatures, double aScale) {
    for (auto& feature : aFeatures) {
        feature.scaleBy(aScale, aScale);
    }
}
