#include "XFeatDescriptor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

XFeatDescriptor::XFeatDescriptor(Values values) : values_(values) {
    double squaredNorm = 0.0;
    for (const float value : values_) {
        if (!std::isfinite(value)) throw std::invalid_argument("XFeat descriptors must be finite");
        squaredNorm += static_cast<double>(value) * value;
    }
    if (squaredNorm <= 0.0) throw std::invalid_argument("XFeat descriptors must be nonzero");
    const double norm = std::sqrt(squaredNorm);
    for (auto& value : values_) value = static_cast<float>(value / norm);
}

double XFeatDescriptor::getDistance(const DescriptorPtr& other) const {
    const auto* xfeat = dynamic_cast<const XFeatDescriptor*>(other.get());
    if (!xfeat) throw std::invalid_argument("Incompatible descriptor types");
    double similarity = 0.0;
    for (std::size_t i = 0; i < values_.size(); ++i)
        similarity += static_cast<double>(values_[i]) * xfeat->values_[i];
    return std::clamp(1.0 - similarity, 0.0, 2.0);
}
