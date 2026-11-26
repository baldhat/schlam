//
// Created by baldhat on 11/14/26.
//

#include "OrbDescriptor.h"

#include <bit>
#include <cstring>

namespace {

#if defined(__x86_64__) && defined(__ELF__) && __has_cpp_attribute(gnu::target_clones)
[[gnu::target_clones("popcnt", "default")]]
#endif
int hamming256(const std::uint8_t* a, const std::uint8_t* b) {
    int distance = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t x, y;
        std::memcpy(&x, a + i * 8, sizeof(x));
        std::memcpy(&y, b + i * 8, sizeof(y));
        distance += std::popcount(x ^ y);
    }
    return distance;
}

} // namespace

OrbDescriptor::OrbDescriptor(const TDescriptor& descriptor) : mDescriptor(descriptor) {}

double OrbDescriptor::getDistance(const DescriptorPtr &other) const {
    const auto* orb = dynamic_cast<const OrbDescriptor*>(other.get());
    if (!orb) {
        throw std::invalid_argument("Incompatible descriptor types");
    }

    return hamming256(mDescriptor.data(), orb->mDescriptor.data());
}
