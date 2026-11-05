from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import get, replace_in_file


class G2OConan(ConanFile):
    name = "g2o"
    version = "1.0.0.fd656273"
    package_type = "shared-library"
    settings = "os", "arch", "compiler", "build_type"

    def requirements(self):
        self.requires("eigen/3.4.0")
        self.requires("spdlog/1.13.0")

    def source(self):
        get(
            self,
            "https://codeload.github.com/RainerKuemmerle/g2o/tar.gz/"
            "fd65627348213d56866b24c0b7d054bcf105f0a5",
            sha256="f99797552a1a7ea1481e38c32fa16cca87ff2fa8ecba4d6a65d6434269b750c9",
            filename="g2o-fd656273.tar.gz",
            strip_root=True,
        )
        for component in ("core", "stuff"):
            replace_in_file(
                self,
                f"{self.source_folder}/g2o/{component}/CMakeLists.txt",
                f"$<INSTALL_INTERFACE:include/g2o/{component}>",
                "$<INSTALL_INTERFACE:include>",
            )

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        toolchain = CMakeToolchain(self)
        toolchain.variables.update({
            "BUILD_SHARED_LIBS": True,
            "G2O_USE_LOGGING": True,
            "G2O_USE_OPENGL": False,
            "G2O_USE_CHOLMOD": False,
            "G2O_USE_CSPARSE": False,
            "G2O_BUILD_APPS": False,
            "G2O_BUILD_EXAMPLES": False,
            "G2O_BUILD_SLAM2D_TYPES": False,
            "G2O_BUILD_SLAM3D_ADDON_TYPES": False,
            "G2O_BUILD_ICP_TYPES": False,
            "BUILD_UNITTESTS": False,
            "DO_SSE_AUTODETECT": False,
            "DISABLE_SSE3": True,
            "DISABLE_SSE4_1": True,
            "DISABLE_SSE4_2": True,
            "DISABLE_SSE4_A": True,
        })
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs = ["lib/cmake/g2o"]
        self.cpp_info.libs = [
            "g2o_types_sim3", "g2o_types_sba", "g2o_types_slam3d",
            "g2o_solver_eigen", "g2o_solver_dense", "g2o_solver_structure_only",
            "g2o_core", "g2o_stuff",
        ]
