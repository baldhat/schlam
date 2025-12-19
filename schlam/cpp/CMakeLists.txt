cmake_minimum_required(VERSION 3.23)
project(Schlam LANGUAGES CXX)

# Force C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Conan dependencies
find_package(fmt CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)


add_executable(app 
    src/main.cpp
)

target_link_libraries(app 
    PRIVATE 
    fmt::fmt
    Eigen3::Eigen
    opencv::opencv
)