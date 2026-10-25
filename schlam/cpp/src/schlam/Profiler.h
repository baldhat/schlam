//
// Created by jonathan_michel on 10/25/26.
//

#ifndef SCHLAM_PROFILER_H
#define SCHLAM_PROFILER_H

#include <chrono>

class IPlotter;

class Profiler {
public:
    Profiler(IPlotter& plotter, std::chrono::steady_clock::time_point start);
    ~Profiler();
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    double finish();

private:
    IPlotter& mPlotter;
    std::chrono::steady_clock::time_point mStart;
    double mElapsedMs{0.0};
    bool mFinished{false};
};


#endif //SCHLAM_PROFILER_H
