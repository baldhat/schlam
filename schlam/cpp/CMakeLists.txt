cmake_minimum_required(VERSION 3.23)
project(Schlam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SCHLAM_USE_OPENCV_RECONSTRUCTION
    "Use OpenCV essential-matrix estimation and triangulation for reconstruction" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SPDLOG_FMT_EXTERNAL)
add_definitions(-DSPDLOG_FMT_EXTERNAL)

add_compile_options(-fno-omit-frame-pointer)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(onnxruntime REQUIRED CONFIG)
find_package(fmt REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(yaml-cpp REQUIRED)
find_package(fast-cpp-csv-parser REQUIRED)
find_package(TBB REQUIRED)
find_package(spdlog REQUIRED)
find_package(Tracy REQUIRED)

find_package(Pangolin REQUIRED CONFIG)
find_package(g2o REQUIRED CONFIG)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
)

# LIBRARY
add_library(${PROJECT_NAME} SHARED 
    src/config/ConfigParser.cpp
    src/config/ConfigParser.hpp
    src/config/Config.hpp
    src/random.cpp
    src/random.hpp

    src/plotting/PangolinPlotter.hpp
    src/plotting/PangolinPlotter.cpp

    src/tft/rigid_transform_3d.cpp
    src/tft/transformable_3d.cpp
    src/tft/transformer.cpp
    src/tft/rigid_transform_3d.hpp
    src/tft/transformable_3d.hpp
    src/tft/transformer.hpp

    src/data/image_data.cpp
    src/data/imu_data.cpp
    src/data/datasets/MAVDataset.cpp
    src/data/gt_data.cpp
    src/data/DatasetInterface.hpp
    src/data/datasets/KittiDataset.cpp
    src/data/datasets/KittiDataset.hpp
    src/data/data_helper.hpp
    src/data/data_helper.cpp
    src/data/image_data.hpp
    src/data/imu_data.hpp
    src/data/datasets/MAVDataset.hpp
    src/data/gt_data.hpp

    src/schlam/feature_detection/orb/FastFeatureDetector.cpp
    src/schlam/feature_detection/orb/FastFeatureDetector.h
    src/schlam/feature_detection/orb/ORBFeatureDetector.cpp
    src/schlam/feature_detection/orb/ORBFeatureDetector.h
    src/schlam/KeyPoint.cpp
    src/schlam/KeyPoint.h
    src/schlam/helpers/utils.cpp
    src/schlam/helpers/utils.h
    src/schlam/feature_detection/orb/QuadTreeNode.cpp
    src/schlam/feature_detection/orb/QuadTreeNode.h
    src/schlam/feature_detection/orb/orb_pattern.h
    src/schlam/Matcher.cpp
    src/schlam/Matcher.h
    src/schlam/Ransac.cpp
    src/schlam/Ransac.h
    src/schlam/optimization/Optimizer.cpp
    src/schlam/optimization/Optimizer.h
    src/schlam/Frame.cpp
    src/schlam/Frame.h
    src/schlam/optimization/Optimizer.cpp
    src/schlam/MapPoint.cpp
    src/schlam/MapPoint.hpp
    src/schlam/Map.cpp
    src/schlam/Map.h
    src/schlam/MotionModel.cpp
    src/schlam/MotionModel.h
    src/schlam/KeyFrame.cpp
    src/schlam/KeyFrame.h
    src/plotting/IPlotter.hpp
    src/plotting/DummyPlotter.cpp
    src/plotting/DummyPlotter.h
    src/schlam/SchlamSystem.cpp
    src/schlam/SchlamSystem.h
    src/schlam/SchlamValidator.cpp
    src/schlam/SchlamValidator.h
    src/schlam/helpers/Profiler.cpp
    src/schlam/helpers/Profiler.h
    src/schlam/optimization/CovisibilityHelpers.cpp
    src/schlam/optimization/CovisibilityHelpers.h
    src/schlam/feature_detection/IFeatureExtractor.h
    src/schlam/feature_detection/IDescriptor.h
    src/schlam/feature_detection/orb/OrbDescriptor.cpp
    src/schlam/feature_detection/orb/OrbDescriptor.h
    src/schlam/feature_detection/xfeat/XFeatModel.cpp
    src/schlam/feature_detection/xfeat/XFeatModel.h
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    SCHLAM_USE_OPENCV_RECONSTRUCTION=$<BOOL:${SCHLAM_USE_OPENCV_RECONSTRUCTION}>)

target_link_libraries(${PROJECT_NAME} 
    fmt::fmt
    spdlog::spdlog
    ${OpenCV_LIBS} 
    Eigen3::Eigen 
    onnxruntime::onnxruntime
    pango_core pango_display pango_plot pango_image pango_geometry pango_opengl pango_video
    yaml-cpp::yaml-cpp
    Boost::boost Boost::thread
    fast-cpp-csv-parser::fast-cpp-csv-parser
    TBB::tbb
    g2o::core g2o::solver_dense g2o::solver_eigen g2o::types_sim3 g2o::types_slam3d
    Tracy::TracyClient
)


add_executable(app
    src/main.cpp
    src/tracy_setup.cpp
)
target_link_libraries(app ${PROJECT_NAME})
target_compile_definitions(app PRIVATE SCHLAM_DEFAULT_CONFIG_PATH="${PROJECT_SOURCE_DIR}/config/mav.yaml")

add_executable(tft 
    src/tft/main.cpp
)
target_link_libraries(tft ${PROJECT_NAME})

add_executable(xfeat_model src/xfeat_model.cpp)
target_link_libraries(xfeat_model ${PROJECT_NAME})
target_compile_definitions(xfeat_model PRIVATE
    SCHLAM_DEFAULT_XFEAT_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/xfeat/xfeat.onnx")

# ----------------------------------------
# -------------- TESTING -----------------
# ----------------------------------------
enable_testing()
find_package(GTest REQUIRED CONFIG)

add_executable(unit_tests
    tests/test_ransac.cpp
    tests/test_reconstruction.cpp
    tests/test_features.cpp
    tests/test_mav_dataloader.cpp
    tests/test_motion_model.cpp
    tests/test_optimizer.cpp
    tests/test_schlam_system.cpp
    tests/test_schlam_validator.cpp
    tests/test_xfeat_model.cpp
)

# Ensure GTest targets are linked correctly
target_link_libraries(unit_tests PRIVATE 
    # Resolve the test entry point before transitive Boost.Test archives.
    GTest::gtest_main
    GTest::gtest
    ${PROJECT_NAME}
)
target_compile_definitions(unit_tests PRIVATE
    SCHLAM_XFEAT_ASSET_DIR="${PROJECT_SOURCE_DIR}/models/xfeat")

include(GoogleTest)
# Discovery runs during the build and requires the Conan runtime environment.
gtest_discover_tests(unit_tests)


