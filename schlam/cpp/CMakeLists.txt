cmake_minimum_required(VERSION 3.23)
project(Schlam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Find Conan packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

# 2. Find System Pangolin
set(Pangolin_DIR "/home/baldhat/dev/Pangolin/build")
find_package(Pangolin REQUIRED CONFIG)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(schlam 
    src/main.cpp
    src/plotting/plotter.cpp
    src/tft/rigid_transform_3d.cpp
    src/tft/transform.cpp
    src/tft/transformable_3d.cpp
    src/tft/transformer.cpp

    src/plotting/plotter.hpp
    src/tft/rigid_transform_3d.hpp
    src/tft/transform.hpp
    src/tft/transformable_3d.hpp
    src/tft/transformer.hpp
)


# 3. Link everything
target_link_libraries(schlam PRIVATE 
    fmt::fmt 
    ${OpenCV_LIBS} 
    Eigen3::Eigen 
    pango_core pango_display pango_plot pango_image pango_geometry pango_opengl
)