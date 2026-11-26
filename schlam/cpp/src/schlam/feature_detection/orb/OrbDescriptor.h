//
// Created by baldhat on 11/14/26.
//

#ifndef SCHLAM_ORBDESCRIPTOR_H
#define SCHLAM_ORBDESCRIPTOR_H
#include <array>
#include <cstdint>

#include "schlam/feature_detection/IDescriptor.h"

using TDescriptor = std::array<std::uint8_t, 32>;

class OrbDescriptor : public IDescriptor {
public:
    using Words = std::array<std::uint64_t, 4>;

    OrbDescriptor() = default;
    OrbDescriptor(const TDescriptor& aDescriptor);

    double getDistance(const DescriptorPtr& other) const;
    Words packedWords() const;

    // Distances at or above the exclusive limit are returned as the limit.
    static int boundedDistance(const Words& source, const Words& target, int limit);

   TDescriptor mDescriptor;
};


#endif //SCHLAM_ORBDESCRIPTOR_H
