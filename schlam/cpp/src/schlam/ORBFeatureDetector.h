//
// Created by baldhat on 2/7/26.
//

#ifndef SCHLAM_ORBFEATUREDETECTOR_H
#define SCHLAM_ORBFEATUREDETECTOR_H
#include <opencv2/core/mat.hpp>

#include "KeyPoint.h"
#include "utils.h"
#include "src/plotting/plotter.hpp"

class ORBFeatureDetector {
public:
    ORBFeatureDetector(const std::uint32_t aNumFeatures, std::shared_ptr<Plotter> apPlotter,
                       const std::uint8_t aNumLevels,
                       const double aLevelFactor = 1 / 1.2);

    std::vector<KeyPoint> getFeatures(const cv::Mat &aImage);

    void distributeFeaturesToLevels();

private:
    std::uint32_t mNumFeatures{0};
    std::uint8_t mNumLevels{0};
    double mLevelFactor{0};
    std::vector<double> mLevelFactors;
    std::vector<Point> mOrientationIndices;
    std::shared_ptr<Plotter> mpPlotter;

    std::array<std::array<std::int8_t, 2>, 16> mFastIndices = {
        std::array<std::int8_t, 2>{-3, 0},
        {-3, 1},
        {-2, 2},
        {-1, 3},
        {0, 3},
        {1, 3},
        {2, 2},
        {3, 0},
        {3, -1},
        {3, -2},
        {2, -3},
        {1, -3},
        {0, -3},
        {-1, -2},
        {-2, -1},
        {-3, -1}
    };
    double mFastThreshold = 7;
    std::uint8_t nContinuous = 9;
    std::vector<std::uint32_t> mNumFeaturesPerLevel;
    std::vector<std::uint8_t> mLevels;

    std::uint32_t mGradKernelSize{9};
    cv::Mat mGradKernelX = (cv::Mat_<float>(3, 3) <<
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1);

    cv::Mat mGradKernelY = (cv::Mat_<float>(3, 3) <<
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1);

    std::vector<KeyPoint> calcFeatures(const std::vector<cv::Mat> &aPyramid);

    void addDescriptors(const std::vector<cv::Mat>& aPyramid, std::vector<KeyPoint>& aFeatures);

    std::vector<KeyPoint> calculateFastFeatures(const cv::Mat &aImage);

    void addOrientation(const std::vector<cv::Mat> &aPyramid,
                        std::vector<KeyPoint> &aFeatures);

    void computeHarrisResponse(const cv::Mat &aImage, std::vector<KeyPoint> &aFeatures,
                               const std::uint8_t blockSize = 7, const double aHarrisK = 0.04);

    std::tuple<double, double> applyGradKernels(const cv::Mat &aImage, int x, int y);

    std::vector<KeyPoint> filterWithOctree(std::vector<KeyPoint> &aFeatures, const std::uint32_t aWidth,
                                                           const std::uint32_t aHeight,
                                                           const std::uint32_t aNumFeatures);

    void rescaleFeatures(std::vector<KeyPoint>& aFeatures, double aScale);
};


#endif //SCHLAM_ORBFEATUREDETECTOR_H
