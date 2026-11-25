from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, cmake_layout


class SchlamConan(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    options = {"with_pangolin": [True, False]}
    default_options = {
        "with_pangolin": True,
        "tracy/*:timer_fallback": True,
        "tracy/*:on_demand": True,
    }
    requires = (
        "fmt/10.2.1",
        "eigen/3.4.0",
        "opencv/4.12.0",
        "ffmpeg/7.1.3",
        "boost/1.85.0",
        "yaml-cpp/0.8.0",
        "spdlog/1.13.0",
        "fast-cpp-csv-parser/20191004",
        "g2o/1.0.0.fd656273",
        "tracy/0.14.1",
    )
    generators = "PkgConfigDeps", "CMakeDeps", "VirtualRunEnv"

    def requirements(self):
        if self.options.with_pangolin:
            self.requires("pangolin/0.9.4")

    def build_requirements(self):
        self.test_requires("gtest/1.15.0")

    def layout(self):
        cmake_layout(self, build_folder="build" if self.options.with_pangolin else "build/headless")

    def generate(self):
        toolchain = CMakeToolchain(self)
        toolchain.variables["SCHLAM_WITH_PANGOLIN"] = bool(self.options.with_pangolin)
        if not self.options.with_pangolin:
            toolchain.presets_prefix = "conan-headless"
        toolchain.generate()
