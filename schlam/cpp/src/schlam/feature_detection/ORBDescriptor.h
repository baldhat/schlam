#pragma once

#include "orb_pattern.h"

#include <array>

namespace schlam::detail {

// Pattern coordinates are small and finite. Truncation followed by a fractional
// correction preserves lround's ties-away-from-zero rule and allows vectorization.
inline int roundPatternCoordinate(double value) {
    const int whole = static_cast<int>(value);
    const double fraction = value - whole;
    return whole + (fraction >= 0.5) - (fraction <= -0.5);
}

inline void rotateOrbPattern(double sinAngle, double cosAngle, std::array<int, 1024>& rotated) {
    for (std::size_t i = 0; i < gOrbBitPattern31.size(); i += 2) {
        const double x = gOrbBitPattern31[i], y = gOrbBitPattern31[i + 1];
        rotated[i] = roundPatternCoordinate(x * cosAngle - y * sinAngle);
        rotated[i + 1] = roundPatternCoordinate(x * sinAngle + y * cosAngle);
    }
}

}
