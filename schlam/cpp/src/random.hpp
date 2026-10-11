#pragma once

#include <random>

namespace utils {
    inline constexpr std::mt19937::result_type kRandomSeed = 12344321;

    std::mt19937& randomEngine();
}
