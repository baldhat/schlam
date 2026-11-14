#pragma once

#include "schlam/feature_detection/IDescriptor.h"
#include <array>

class XFeatDescriptor final : public IDescriptor {
public:
    using Values = std::array<float, 64>;
    explicit XFeatDescriptor(Values values);

    // Cosine distance in [0,2]. Inputs are normalized once at construction.
    double getDistance(const DescriptorPtr& other) const override;
    const Values& values() const { return values_; }

private:
    Values values_;
};
