#pragma once

#include "IFeatureExtractor.h"
#include "config/Config.hpp"
#include <memory>

class IPlotter;

std::shared_ptr<IFeatureExtractor> createFeatureExtractor(
    const schlam::config::Features& settings, std::shared_ptr<IPlotter> plotter = nullptr);
