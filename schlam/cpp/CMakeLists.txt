cmake_minimum_required(VERSION 3.23)
project(Schlam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-fno-omit-frame-pointer)

# 1. Find Conan packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(yaml-cpp REQUIRED)
find_package(fast-cpp-csv-parser REQUIRED)
find_package(TBB REQUIRED)

# 2. Find System Pangolin
set(Pangolin_DIR "/home/baldhat/dev/Pangolin/build")
find_package(Pangolin REQUIRED CONFIG)
include_directories(${Pangolin_INCLUDE_DIRS})

include_directories(
  ${PROJECT_SOURCE_DIR}
)

# LIBRARY
add_library(${PROJECT_NAME} SHARED 
    src/plotting/plotter.cpp
    src/tft/rigid_transform_3d.cpp
    src/tft/transformable_3d.cpp
    src/tft/transformer.cpp
    src/data/image_data.cpp
    src/data/imu_data.cpp
    src/data/mav_dataloader.cpp
    src/data/gt_data.cpp

    src/plotting/plotter.hpp
    src/tft/rigid_transform_3d.hpp
    src/tft/transformable_3d.hpp
    src/tft/transformer.hpp
    src/data/image_data.hpp
    src/data/imu_data.hpp
    src/data/mav_dataloader.hpp
    src/data/gt_data.hpp
    src/schlam/FastFeatureDetector.cpp
    src/schlam/FastFeatureDetector.h
    src/schlam/ORBFeatureDetector.cpp
    src/schlam/ORBFeatureDetector.h
    src/schlam/KeyPoint.cpp
    src/schlam/KeyPoint.h
    src/schlam/utils.cpp
    src/schlam/utils.h
    src/schlam/QuadTreeNode.cpp
    src/schlam/QuadTreeNode.h
    src/schlam/orb_pattern.h
    src/schlam/Matcher.cpp
    src/schlam/Matcher.h
    src/schlam/Ransac.cpp
    src/schlam/Ransac.h
)

target_link_libraries(${PROJECT_NAME} 
    fmt::fmt 
    ${OpenCV_LIBS} 
    Eigen3::Eigen 
    pango_core pango_display pango_plot pango_image pango_geometry pango_opengl pango_video
    yaml-cpp::yaml-cpp
    Boost::boost Boost::thread
    fast-cpp-csv-parser::fast-cpp-csv-parser
    TBB::tbb
)


add_executable(app 
    src/main.cpp
)
target_link_libraries(app ${PROJECT_NAME})

add_executable(tft 
    src/tft/main.cpp
)
target_link_libraries(tft ${PROJECT_NAME})

# ----------------------------------------
# -------------- TESTING -----------------
# ----------------------------------------
enable_testing()
find_package(GTest REQUIRED CONFIG)

add_executable(unit_tests tests/test_ransac.cpp)

# Ensure GTest targets are linked correctly
target_link_libraries(unit_tests PRIVATE 
    ${PROJECT_NAME}
    GTest::gtest_main
    GTest::gtest
)

include(GoogleTest)
# PRE_TEST avoids trying to run the binary during the 'make' phase
gtest_discover_tests(unit_tests)


