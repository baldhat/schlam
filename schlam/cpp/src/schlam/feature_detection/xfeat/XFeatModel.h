#ifndef SCHLAM_XFEATMODEL_H
#define SCHLAM_XFEATMODEL_H

#include <filesystem>
#include <memory>
#include <opencv2/core/mat.hpp>

// Loads the pretrained XFeat backbone once and exposes its dense outputs.
// Sparse keypoint selection and descriptor sampling belong to the extractor.
class XFeatModel {
public:
    struct Output {
        // Owned float32 NCHW tensors; spatial dimensions are input H/8, W/8.
        cv::Mat features;       // [1, 64, H/8, W/8], unnormalized descriptors
        cv::Mat keypointLogits; // [1, 65, H/8, W/8], includes the dustbin channel
        cv::Mat reliability;   // [1, 1, H/8, W/8], probabilities
    };

    explicit XFeatModel(const std::filesystem::path& modelPath, int inferenceThreads = 4);
    ~XFeatModel();
    XFeatModel(const XFeatModel&) = delete;
    XFeatModel& operator=(const XFeatModel&) = delete;

    // CV_8UC1 grayscale, H/W >= 32 and divisible by 32. Converts pixels to [0,1].
    // Resizing and coordinate remapping remain explicit responsibilities of the extractor.
    [[nodiscard]] Output run(const cv::Mat& image) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
