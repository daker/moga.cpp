from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout


class MogaConan(ConanFile):
    name = "moga"
    version = "0.1.0"
    license = "BSD-3-Clause"
    description = "Backend neutral shader DSL compiling to GLSL ES 3.0 and WGSL"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}
    exports_sources = "CMakeLists.txt", "include/*", "src/*", "examples/*", "tests/*"
    generators = "CMakeDeps"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def validate(self):
        check_min_cppstd(self, 23)

    def build_requirements(self):
        self.test_requires("doctest/2.4.11")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["MOGA_BUILD_TESTS"] = not self.conf.get("tools.build:skip_test", default=False)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        if not self.conf.get("tools.build:skip_test", default=False):
            cmake.test()

    def package(self):
        CMake(self).install()

    def package_info(self):
        self.cpp_info.libs = ["moga"]
        self.cpp_info.set_property("cmake_target_name", "moga::moga")
