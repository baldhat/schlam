#pragma once

#include "XFeatModel.h"
#include "config/Config.hpp"
#include "schlam/feature_detection/IFeatureExtractor.h"

class IPlotter;

class XFeatFeatureExtractor final : public IFeatureExtractor {
public:
    explicit XFeatFeatureExtractor(const schlam::config::Features& settings,
                                   std::shared_ptr<IPlotter> plotter = nullptr);
    void extractFeatures(Frame& frame) override;

private:
    schlam::config::Features settings_;
    std::shared_ptr<IPlotter> plotter_;
    XFeatModel model_;
};
