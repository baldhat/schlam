#include "FeatureExtractorFactory.h"
#include "orb/ORBFeatureDetector.h"
#include "xfeat/XFeatFeatureExtractor.h"

#include <stdexcept>
#include <utility>

std::shared_ptr<IFeatureExtractor> createFeatureExtractor(
        const schlam::config::Features& settings, std::shared_ptr<IPlotter> plotter) {
    switch (settings.type) {
        case schlam::config::FeatureType::ORB:
            return std::make_shared<ORBFeatureDetector>(settings, std::move(plotter));
        case schlam::config::FeatureType::XFEAT:
            return std::make_shared<XFeatFeatureExtractor>(settings, std::move(plotter));
    }
    throw std::invalid_argument("Invalid feature extractor type");
}
