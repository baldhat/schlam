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
    OrbDescriptor() = default;
    OrbDescriptor(const TDescriptor& aDescriptor);

    double getDistance(const DescriptorPtr& other) const;

   TDescriptor mDescriptor;
};


#endif //SCHLAM_ORBDESCRIPTOR_H
