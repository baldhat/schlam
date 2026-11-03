//
// Created by jonathan_michel on 10/25/26.
//

#include "Profiler.h"

#include "plotting/IPlotter.hpp"


Profiler::Profiler(IPlotter& plotter, std::chrono::steady_clock::time_point start)
    : mPlotter(plotter), mStart(start) {}

Profiler::~Profiler() { finish(); }

double Profiler::finish() {
    if (!mFinished) {
        const auto elapsed = std::chrono::steady_clock::now() - mStart;
        mElapsedMs = std::chrono::duration<double, std::milli>(elapsed).count();
        mPlotter.recordFrameTime(elapsed);
        mFinished = true;
    }
    return mElapsedMs;
}
