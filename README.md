# schlam

A from-scratch visual(-inertial) SLAM playground. The core building blocks of a SLAM frontend — feature detection, matching, RANSAC-based pose estimation, and bundle adjustment — are implemented by hand rather than taken off the shelf, in two parallel implementations:

- **C++** (`schlam/cpp`): the main pipeline, running on the [EuRoC MAV dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) with Pangolin-based 3D visualization.
- **Python** (`schlam/python`): a PyTorch-based prototyping ground with IMU preintegration, pyceres bundle adjustment, and ROS 2 visualization. Supports EuRoC MAV and KITTI Odometry.

## C++ pipeline

The main executable (`src/main.cpp`) runs a frame-to-frame visual odometry loop:

1. **Data loading** (`src/data`) — threaded EuRoC MAV loader for images, IMU samples, and ground truth, configured via YAML sensor configs.
2. **Feature detection** (`src/schlam`) — custom FAST and ORB detectors with a quadtree (`QuadTreeNode`) for spatially even keypoint distribution.
3. **Matching** (`Matcher`) — windowed descriptor matching between consecutive frames.
4. **Initial reconstruction** (`Ransac`) — essential matrix estimation with RANSAC, pose recovery, and point triangulation.
5. **Optimization** (`Optimizer`) — two-view bundle adjustment over poses and landmarks using [g2o](https://github.com/RainerKuemmerle/g2o) (Levenberg-Marquardt with Huber kernels).
6. **Visualization** (`src/plotting`) — Pangolin viewer showing the camera frustum, point cloud, and coordinate frames.

`src/tft` is a small transform-tree library ("tf for the rest of us") for registering and chaining rigid 3D transforms between named coordinate frames (`imu`, `cam0`, `world`, ...).

### Building

Dependencies can be resolved with Conan (see `conanfile.txt`) or installed on the system; [Pangolin](https://github.com/stevenlovegrove/Pangolin) must be installed on the system either way. g2o is vendored as a git submodule.

```bash
git submodule update --init --recursive

cd schlam/cpp
conan install . --build=missing
cmake --preset conan-release
cmake --build --preset conan-release
```

### Running

Point `MAV_PATH` at the `mav0` folder of an extracted EuRoC MAV sequence (the directory containing `cam0/`, `imu0/`, and `state_groundtruth_estimate0/`):

```bash
export MAV_PATH=/path/to/euroc/MH_01_easy/mav0
./app   # from the build directory
```

There is also a `tft` executable for testing the transform-tree visualization on its own.


## Python pipeline

The Python implementation (`schlam/python/main.py`) mirrors the C++ pipeline but adds:

- **IMU preintegration** (`imu_calc.py`) between camera frames.
- **Sliding-window local bundle adjustment** (`local_bundle_adjustment.py`) over the last N frames using pyceres.
- **Matching** via FLANN or pyramidal Lucas-Kanade optical flow (`matcher.py`).
- **ROS 2 visualization** (`visualizer.py`) publishing camera path, point cloud, and TF frames (view with RViz).

Install dependencies with Poetry (a working ROS 2 installation with `rclpy`/`tf2_ros` is required separately):

```bash
poetry install
```

Set the dataset path and run:

```bash
export KITTI_ODOMETRY_PATH=/path/to/KittiOdometry
python schlam/python/main.py
```

## Repository layout

```
schlam/
├── cpp/
│   ├── src/
│   │   ├── schlam/     # Feature detection, matching, RANSAC, optimization
│   │   ├── data/       # EuRoC MAV dataset loading
│   │   ├── tft/        # Rigid transform tree
│   │   └── plotting/   # Pangolin visualization
│   ├── tests/          # GTest unit tests
│   └── external/g2o    # g2o submodule
└── python/             # PyTorch/pyceres prototype pipeline
```
