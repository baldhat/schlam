# Local Conan packages

The application requires two packages built from pinned upstream source:

| Package | Upstream revision | Purpose |
| --- | --- | --- |
| `g2o/1.0.0.fd656273` | `fd65627348213d56866b24c0b7d054bcf105f0a5` | Keeps the revision previously used through `external/g2o` |
| `pangolin/0.9.4` | `b2346d6ba383342c835ef72e43abc3478434d16c` (`v0.9.4`) | Replaces the developer specific Pangolin build path |

The recipes verify SHA-256 checksums of both source archives. They build shared libraries and install each project's native CMake package config. The g2o recipe keeps logging support but disables unused solver families and GUI support. The Pangolin recipe keeps its OpenGL viewer and disables optional image and video backends. `conan.lock` pins the Conan recipe revisions of the application dependency graph.

On the self hosted Ubuntu runner, install GCC/G++ 14, CMake 3.23 or newer, Git, Python 3, `ninja-build`, `pkg-config`, `libepoxy-dev`, `libegl-dev`, `libgl-dev`, `libglu1-mesa-dev`, `libx11-dev`, `libglew-dev`, and `libtbb-dev`:

```sh
sudo apt-get update
sudo apt-get install -y gcc-14 g++-14 cmake git python3 python3-venv ninja-build pkg-config \
  libepoxy-dev libegl-dev libgl-dev libglu1-mesa-dev libx11-dev libglew-dev libtbb-dev
```

From `schlam/cpp`, run the following for either `Debug` or `Release`:

```sh
conan profile detect --force
conan create recipes/g2o -s build_type=Release --lockfile=conan.lock --build=missing
conan create recipes/pangolin -s build_type=Release --lockfile=conan.lock --build=missing
conan install . -s build_type=Release --lockfile=conan.lock --build=missing
cmake -S . -B build/Release -DCMAKE_TOOLCHAIN_FILE="$PWD/build/Release/generators/conan_toolchain.cmake" -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release --parallel 2
ctest --test-dir build/Release --output-on-failure
```

The GitHub Actions workflow runs Debug and Release builds and tests for pull requests on GitHub hosted runners. Every push runs build, tests, and `config/mav_headless.yaml` on a self hosted Linux x64 runner. That runner must be able to read `/home/baldhat/data/mav/vicon_room1/V1_02_medium/mav0`; the workflow sets `MAV_PATH` to its parent path. Pull request code never runs on the dataset host. No private Conan remote or secrets are required. Both packages are rebuilt in each fresh CI job, so builds take longer than using a binary remote.
