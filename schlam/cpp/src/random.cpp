#include "random.hpp"

namespace utils {
    std::mt19937& randomEngine() {
        static std::mt19937 engine{kRandomSeed};
        return engine;
    }
}
