//
// Created by baldhat on 2/7/26.
//

#ifndef SCHLAM_ORBFEATUREDETECTOR_H
#define SCHLAM_ORBFEATUREDETECTOR_H
#include <opencv2/core/mat.hpp>

#include "FastFeatureDetector.h"
#include "config/Config.hpp"
#include "../helpers/utils.h"

class Frame;
class IPlotter;
class QuadTreeNode;
class KeyPoint;

class ORBFeatureDetector {
public:
    ORBFeatureDetector(const std::uint32_t aNumFeatures, std::shared_ptr<IPlotter> apPlotter,
                       const std::uint8_t aNumLevels,
                       const double aLevelFactor = 1 / 1.2);
    ORBFeatureDetector(const schlam::config::Features& settings, std::shared_ptr<IPlotter> plotter);

    void detectFeatures(Frame& frame);

    void distributeFeaturesToLevels();

private:
    friend class ORBFeatureDetectorTest;
    std::uint32_t mNumFeatures{0};
    std::uint8_t mNumLevels{0};
    std::uint32_t mOrientationRadius{15};
    schlam::config::Features mSettings;
    double mLevelFactor{0};
    std::vector<double> mLevelFactors;
    std::vector<int> mOrientationHalfWidths;
    std::shared_ptr<IPlotter> mpPlotter;

    schlam::FastFeatureDetector mFastFeatureDetector;
    std::vector<std::uint32_t> mNumFeaturesPerLevel;
    std::vector<std::uint8_t> mLevels;

    std::uint32_t mGradKernelSize{9};
    cv::Mat mGradKernelX = (cv::Mat_<double>(3, 3) <<
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1);

    cv::Mat mGradKernelY = (cv::Mat_<double>(3, 3) <<
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1);

    std::unique_ptr<QuadTreeNode> calcFeatures(const std::vector<cv::Mat> &aPyramid);

    std::unique_ptr<QuadTreeNode> mergeFeatureLevels(
        const std::vector<std::vector<std::shared_ptr<KeyPoint>>>& featuresByLevel,
        const cv::Mat& image);

    void addDescriptors(const std::vector<cv::Mat>& aPyramid,
                        const std::vector<std::shared_ptr<KeyPoint>>& features);

    void addOrientation(const std::vector<cv::Mat> &aPyramid,
                        const std::vector<std::shared_ptr<KeyPoint>>& features);

    void computeHarrisResponse(const cv::Mat &aImage, std::vector<KeyPoint> &aFeatures,
                               const int blockSize = 7, const double aHarrisK = 0.04);

    std::tuple<double, double> applyGradKernels(const cv::Mat &aImage, int x, int y);

    std::unique_ptr<QuadTreeNode> filterWithOctree(std::vector<KeyPoint> &aFeatures, const std::uint32_t aWidth,
                                                           const std::uint32_t aHeight,
                                                           const std::uint32_t aNumFeatures);

    void rescaleFeatures(std::vector<std::shared_ptr<KeyPoint>>& aFeatures, const double aScale,
                         const std::uint32_t aLevel);

};


#endif //SCHLAM_ORBFEATUREDETECTOR_H
