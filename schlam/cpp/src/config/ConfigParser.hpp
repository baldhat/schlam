#pragma once

#include "Config.hpp"

#include <filesystem>

namespace schlam::config {

class ConfigParser {
public:
    static Config load(const std::filesystem::path& path);
};

}  // namespace schlam::config
