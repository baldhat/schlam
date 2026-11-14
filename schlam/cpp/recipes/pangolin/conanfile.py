from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy, get, replace_in_file
from os.path import join


class PangolinConan(ConanFile):
    name = "pangolin"
    version = "0.9.4"
    package_type = "shared-library"
    settings = "os", "arch", "compiler", "build_type"

    def requirements(self):
        self.requires("eigen/5.0.1")

    def source(self):
        get(
            self,
            "https://codeload.github.com/stevenlovegrove/Pangolin/tar.gz/"
            "b2346d6ba383342c835ef72e43abc3478434d16c",
            sha256="c65fa303b348b9cd74519f3f2f98f4a17758a9ddcc2a9a26a9b25f412098d0df",
            filename="pangolin-v0.9.4.tar.gz",
            strip_root=True,
        )
        replace_in_file(
            self,
            join(self.source_folder, "components/pango_opengl/CMakeLists.txt"),
            "PUBLIC ${epoxy_LIBRARIES}",
            "PUBLIC epoxy::epoxy",
        )
        replace_in_file(
            self,
            join(self.source_folder, "cmake/PangolinConfig.cmake.in"),
            "SET( Pangolin_CMAKEMODULES @CMAKE_CURRENT_SOURCE_DIR@/cmake )",
            'SET( Pangolin_CMAKEMODULES "${PROJECT_CMAKE_DIR}" )',
        )
        replace_in_file(
            self,
            join(self.source_folder, "cmake/PangolinConfig.cmake.in"),
            "include(CMakeFindDependencyMacro)",
            'list(APPEND CMAKE_MODULE_PATH "${PROJECT_CMAKE_DIR}")\n'
            "include(CMakeFindDependencyMacro)\n"
            "find_dependency(epoxy REQUIRED)",
        )

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        toolchain = CMakeToolchain(self)
        toolchain.variables.update({
            "BUILD_SHARED_LIBS": True,
            "BUILD_TOOLS": False,
            "BUILD_EXAMPLES": False,
            "BUILD_TESTS": False,
            "BUILD_PANGOLIN_PYTHON": False,
            "BUILD_PANGOLIN_LIBPNG": False,
            "BUILD_PANGOLIN_LIBJPEG": False,
            "BUILD_PANGOLIN_LIBTIFF": False,
            "BUILD_PANGOLIN_LIBOPENEXR": False,
            "BUILD_PANGOLIN_LZ4": False,
            "BUILD_PANGOLIN_ZSTD": False,
            "BUILD_PANGOLIN_LIBRAW": False,
            "BUILD_PANGOLIN_LIBDC1394": False,
            "BUILD_PANGOLIN_FFMPEG": False,
            "BUILD_PANGOLIN_REALSENSE": False,
            "BUILD_PANGOLIN_REALSENSE2": False,
            "BUILD_PANGOLIN_OPENNI": False,
            "BUILD_PANGOLIN_OPENNI2": False,
            "BUILD_PANGOLIN_LIBUVC": False,
            "BUILD_PANGOLIN_DEPTHSENSE": False,
            "BUILD_PANGOLIN_TELICAM": False,
            "BUILD_PANGOLIN_PLEORA": False,
        })
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        copy(
            self,
            "Findepoxy.cmake",
            src=join(self.source_folder, "cmake"),
            dst=join(self.package_folder, "lib/cmake/Pangolin"),
        )

    def package_info(self):
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs = ["lib/cmake/Pangolin"]
        self.cpp_info.libs = [
            "pango_plot", "pango_display", "pango_windowing", "pango_vars",
            "pango_geometry", "pango_video", "pango_packetstream",
            "pango_opengl", "pango_image", "pango_core", "tinyobj",
        ]
        self.cpp_info.system_libs = ["epoxy", "OpenGL", "EGL", "X11"]
