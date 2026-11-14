#include "XFeatFeatureExtractor.h"
#include "XFeatDescriptor.h"
#include "plotting/IPlotter.hpp"
#include "schlam/Frame.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <tracy/Tracy.hpp>

namespace {
float cubicWeight(float distance) {
    constexpr float a = -0.75f; // PyTorch grid_sample bicubic kernel.
    distance = std::abs(distance);
    if (distance <= 1.0f) return ((a + 2) * distance - (a + 3)) * distance * distance + 1;
    if (distance < 2.0f) return ((a * distance - 5 * a) * distance + 8 * a) * distance - 4 * a;
    return 0.0f;
}

// Match InterpolateSparse2d: normalize by full-resolution W-1/H-1, then
// grid_sample with align_corners=false and zero padding on the coarse field.
float interpolate(const cv::Mat& tensor, int channel, float x, float y,
                  int fullWidth, int fullHeight, bool bicubic) {
    const int height = tensor.size[2], width = tensor.size[3];
    const float gridX = 2.0f * (x / (fullWidth - 1)) - 1.0f;
    const float gridY = 2.0f * (y / (fullHeight - 1)) - 1.0f;
    const float sx = ((gridX + 1.0f) * width - 1.0f) * 0.5f;
    const float sy = ((gridY + 1.0f) * height - 1.0f) * 0.5f;
    const int baseX = static_cast<int>(std::floor(sx)), baseY = static_cast<int>(std::floor(sy));
    const float* values = tensor.ptr<float>() + channel * height * width;
    float result = 0.0f;
    const int first = bicubic ? -1 : 0, last = bicubic ? 2 : 1;
    for (int dy = first; dy <= last; ++dy) {
        const int iy = baseY + dy;
        if (iy < 0 || iy >= height) continue;
        const float wy = bicubic ? cubicWeight(sy - iy) : std::max(0.0f, 1.0f - std::abs(sy - iy));
        for (int dx = first; dx <= last; ++dx) {
            const int ix = baseX + dx;
            if (ix < 0 || ix >= width) continue;
            const float wx = bicubic ? cubicWeight(sx - ix) : std::max(0.0f, 1.0f - std::abs(sx - ix));
            result += values[iy * width + ix] * wx * wy;
        }
    }
    return result;
}

std::filesystem::path modelPath(const schlam::config::XFeat& settings) {
    return settings.modelPath.empty() ? std::filesystem::path(SCHLAM_DEFAULT_XFEAT_MODEL_PATH) :
                                       std::filesystem::path(settings.modelPath);
}
}

XFeatFeatureExtractor::XFeatFeatureExtractor(const schlam::config::Features& settings,
        std::shared_ptr<IPlotter> plotter)
    : settings_(settings), plotter_(std::move(plotter)), model_(modelPath(settings.xfeat), settings.xfeat.inferenceThreads) {
    if (!std::isfinite(settings.xfeat.detectionThreshold) || settings.xfeat.detectionThreshold < 0.0 ||
        settings.xfeat.detectionThreshold > 1.0) {
        throw std::invalid_argument("XFeat detectionThreshold must be in [0,1]");
    }
    if (settings.xfeat.imageSizing != schlam::config::XFeatImageSizing::RESIZE &&
        settings.xfeat.imageSizing != schlam::config::XFeatImageSizing::PAD) {
        throw std::invalid_argument("Invalid XFeat image sizing policy");
    }
}

void XFeatFeatureExtractor::extractFeatures(Frame& frame) {
    ZoneScoped;
    const auto& image = frame.mImage;
    if (image.empty() || image.dims != 2 || image.type() != CV_8UC1) {
        throw std::invalid_argument("XFeat extraction requires nonempty CV_8UC1 images");
    }
    std::vector<KeyPoint> features;
    if (settings_.count > 0) {
        // Resize in float, matching upstream bilinear preprocessing without
        // quantizing the interpolated intensities back to uint8.
        cv::Mat normalized, resized;
        image.convertTo(normalized, CV_32FC1, 1.0 / 255.0);
        const bool pad = settings_.xfeat.imageSizing == schlam::config::XFeatImageSizing::PAD;
        const int height = pad ? ((image.rows + 31) / 32) * 32 : std::max(32, (image.rows / 32) * 32);
        const int width = pad ? ((image.cols + 31) / 32) * 32 : std::max(32, (image.cols / 32) * 32);
        if (pad) cv::copyMakeBorder(normalized, resized, 0, height - image.rows, 0, width - image.cols, cv::BORDER_REFLECT_101);
        else cv::resize(normalized, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
        auto output = model_.runNormalized(resized);
        const int coarseHeight = height / 8, coarseWidth = width / 8, cells = coarseHeight * coarseWidth;
        float* dense = output.features.ptr<float>();
        for (int cell = 0; cell < cells; ++cell) {
            double squaredNorm = 0.0;
            for (int channel = 0; channel < 64; ++channel) {
                const double value = dense[channel * cells + cell];
                squaredNorm += value * value;
            }
            const double norm = std::max(std::sqrt(squaredNorm), 1e-12);
            for (int channel = 0; channel < 64; ++channel)
                dense[channel * cells + cell] = static_cast<float>(dense[channel * cells + cell] / norm);
        }

        cv::Mat heatmap(height, width, CV_32FC1);
        const float* logits = output.keypointLogits.ptr<float>();
        for (int cell = 0; cell < cells; ++cell) {
            float maximum = logits[cell];
            for (int channel = 1; channel < 65; ++channel) maximum = std::max(maximum, logits[channel * cells + cell]);
            float exponentials[65], sum = 0.0f;
            for (int channel = 0; channel < 65; ++channel) {
                exponentials[channel] = std::exp(logits[channel * cells + cell] - maximum);
                sum += exponentials[channel];
            }
            // The 65th dustbin contributes to softmax but not the 8x8 heatmap.
            for (int channel = 0; channel < 64; ++channel) {
                heatmap.at<float>((cell / coarseWidth) * 8 + channel / 8,
                                  (cell % coarseWidth) * 8 + channel % 8) = exponentials[channel] / sum;
            }
        }
        cv::Mat maxima;
        cv::dilate(heatmap, maxima, cv::Mat::ones(5, 5, CV_8UC1), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, 0);
        struct Candidate { int x, y; float score; };
        std::vector<Candidate> candidates;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (pad && (x >= image.cols || y >= image.rows)) continue;
                const float heat = heatmap.at<float>(y, x);
                if (heat <= settings_.xfeat.detectionThreshold || heat != maxima.at<float>(y, x) || (x == 0 && y == 0)) continue;
                // Upstream nearest grid sampling maps the last row/column to
                // a half-pixel outside an even-sized tensor (zero padding).
                const int nx = static_cast<int>(std::nearbyint(((2.0f * (float(x) / (width - 1)) - 1.0f + 1.0f) * width - 1.0f) * 0.5f));
                const int ny = static_cast<int>(std::nearbyint(((2.0f * (float(y) / (height - 1)) - 1.0f + 1.0f) * height - 1.0f) * 0.5f));
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                const float score = heatmap.at<float>(ny, nx) * interpolate(output.reliability, 0, x, y, width, height, false);
                if (score > 0.0f && std::isfinite(score)) candidates.push_back({x, y, score});
            }
        }
        const auto count = std::min<std::size_t>(settings_.count, candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + count, candidates.end(), [](const auto& a, const auto& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.y != b.y ? a.y < b.y : a.x < b.x;
        });
        for (std::size_t i = 0; i < count; ++i) {
            const auto& candidate = candidates[i];
            XFeatDescriptor::Values descriptor;
            double squaredNorm = 0.0;
            for (int channel = 0; channel < 64; ++channel) {
                descriptor[channel] = interpolate(output.features, channel, candidate.x, candidate.y, width, height, true);
                squaredNorm += static_cast<double>(descriptor[channel]) * descriptor[channel];
            }
            if (!std::isfinite(squaredNorm) || squaredNorm <= 1e-24) continue;
            features.emplace_back(pad ? candidate.x : candidate.x * (double(image.cols) / width),
                pad ? candidate.y : candidate.y * (double(image.rows) / height),
                0, candidate.score, 0.0, std::make_shared<XFeatDescriptor>(descriptor));
        }
    }
    auto tree = std::make_shared<QuadTreeNode>(0, image.cols, 0, image.rows, features);
    std::vector<QuadTreeNode*> pending{tree.get()};
    while (!pending.empty()) {
        auto* node = pending.back();
        pending.pop_back();
        node->divide();
        const auto children = node->getChildren();
        pending.insert(pending.end(), children.begin(), children.end());
    }
    frame.mKeypointTree = tree;
    const auto selected = tree->getFeatures(); // Assign stable feature indices.
    if (plotter_) plotter_->plotFeatures(image, selected, *tree);
}
