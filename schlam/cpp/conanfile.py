from conan import ConanFile
from conan.tools.cmake import cmake_layout


class SchlamConan(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    generators = "PkgConfigDeps", "CMakeDeps", "CMakeToolchain", "VirtualRunEnv"
    default_options = {
        "tracy/*:timer_fallback": True,
        "tracy/*:on_demand": True,
        "onnxruntime/*:shared": True,
        "onnxruntime/*:with_cuda": False,
        "onnx/*:shared": False,
        "onnx/*:disable_static_registration": True,
    }

    def requirements(self):
        # OpenCV 4.12 pins older versions. Keep one consistent Eigen/Protobuf
        # version across OpenCV, ONNX Runtime, g2o and the application.
        self.requires("eigen/5.0.1", force=True)
        self.requires("protobuf/6.33.5", force=True)
        self.requires("cpuinfo/cci.20251210", force=True)
        for requirement in (
            "onnxruntime/1.24.4", "fmt/10.2.1", "opencv/4.12.0",
            "boost/1.85.0", "yaml-cpp/0.8.0", "spdlog/1.13.0",
            "fast-cpp-csv-parser/20191004", "g2o/1.0.0.fd656273",
            "pangolin/0.9.4", "tracy/0.14.1",
        ):
            self.requires(requirement)

    def build_requirements(self):
        self.test_requires("gtest/1.15.0")

    def layout(self):
        cmake_layout(self)
