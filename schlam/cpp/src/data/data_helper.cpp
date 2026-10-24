#include "data_helper.hpp"

#include <chrono>

namespace helpers {
    std::chrono::system_clock::time_point stringNanoToTimePoint(const std::string &nano_str) {
        long long nanos = std::stoll(nano_str);
        std::chrono::nanoseconds duration(nanos);
        return std::chrono::system_clock::time_point(
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                duration));
    }

    std::chrono::system_clock::time_point longNanoToTimePoint(const std::uint64_t &aNanos) {
        std::chrono::nanoseconds duration(aNanos);
        return std::chrono::system_clock::time_point(
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                duration));
    }

    std::chrono::system_clock::time_point stringSecondsToTimePoint(const std::string &aSeconds) {
        double seconds = std::stod(aSeconds);
        std::chrono::nanoseconds duration(static_cast<long long>(std::round(seconds * 1e9)));
        return std::chrono::system_clock::time_point(
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                duration));
    }
}