# Local Conan packages

The application requires two packages built from pinned upstream source:

| Package | Upstream revision | Purpose |
| --- | --- | --- |
| `g2o/1.0.0.fd656273` | `fd65627348213d56866b24c0b7d054bcf105f0a5` | Keeps the revision previously used through `external/g2o` |
| `pangolin/0.9.4` | `b2346d6ba383342c835ef72e43abc3478434d16c` (`v0.9.4`) | Replaces the developer specific Pangolin build path |

The recipes verify SHA-256 checksums of both source archives. They build shared libraries and install each project's native CMake package config. The g2o recipe keeps logging support but disables unused solver families and GUI support. The Pangolin recipe keeps its OpenGL viewer and disables optional image and video backends. `conan.lock` pins the Conan recipe revisions of the application dependency graph.

Application dependencies are declared in `conanfile.py`. Eigen is pinned to 5.0.1
in the application and both local recipes, and ONNX Runtime 1.24.4 is supplied by
ConanCenter. OpenCV 4.12 still requests Eigen 3.4 and Protobuf 3.21, so the application
forces a consistent Eigen 5.0.1 and Protobuf 6.33.5 across the graph. cpuinfo is also
unified at `cci.20251210` for ONNX Runtime and the video codec dependency. The
Python recipe is needed to express these overrides; it replaces `conanfile.txt`.
Dependent libraries must be rebuilt with the new headers and Protobuf ABI.
When reusing an existing CMake build directory after the upgrade, add
`-U g2o_DIR -U Pangolin_DIR` to the configure command below so cached native
package paths are rediscovered, or use `cmake --fresh` with CMake 3.24 or newer.

The pretrained [XFeat model](../models/xfeat/README.md) is included in the source
tree with its license, provenance and export checksums. C++ model loading and
inference tests run as part of `unit_tests`. Python/PyTorch is required only to
regenerate the model using `recipes/export-xfeat.py`.

See the [main build instructions](../../../README.md#building) for Conan installation. Install the complete Ubuntu 24.04 package list from the repository root:

```sh
sudo apt-get update
xargs -r -a schlam/cpp/recipes/system-packages-ubuntu.txt sudo apt-get install -y
```

[`system-packages-ubuntu.txt`](system-packages-ubuntu.txt) includes the project's build and graphics dependencies, the apt requirements of the locked `xorg/system`, `xkeyboard-config/system`, `vaapi/system`, and `vdpau/system` recipes, and headers for Pangolin's Wayland backend autodetection. OpenCV's current options enable FFmpeg and Wayland, so these system libraries are needed even when the application uses `DummyPlotter`.

Conan checks these packages without installing them on the self hosted runner. The GitHub hosted PR job installs the same list. If the lockfile or dependency options change, review the list against the new recipes' `system_requirements()` methods.

From `schlam/cpp`, use GCC/G++ 14 and C++23 to match CI:

```sh
export CC=gcc-14
export CXX=g++-14
conan profile detect --force
conan create recipes/g2o -s build_type=Release -s compiler.cppstd=23 --lockfile=conan.lock --build=missing -c tools.build:jobs="$(nproc)"
conan create recipes/pangolin -s build_type=Release -s compiler.cppstd=23 --lockfile=conan.lock --build=missing -c tools.build:jobs="$(nproc)"
conan install . -s build_type=Release -s compiler.cppstd=23 --lockfile=conan.lock --build=missing -c tools.build:jobs="$(nproc)"
cmake -S . -B build/Release -DCMAKE_TOOLCHAIN_FILE="$PWD/build/Release/generators/conan_toolchain.cmake" -DCMAKE_BUILD_TYPE=Release
source build/Release/generators/conanrun.sh
cmake --build build/Release --parallel "$(nproc)"
ctest --test-dir build/Release --output-on-failure
```

Replace `Release` with `Debug` throughout for a Debug build. Keep your existing profile if you use a custom one; `conan profile detect --force` replaces the default profile.

Activate `conanrun.sh` in each new shell before building, testing, or running the app. `VirtualRunEnv` supplies the Conan shared-library paths, including Pangolin's transitive libraries. Activation is needed during the build because GoogleTest executes the test binary to discover tests. CI activates it separately in the build, test, and headless steps because steps do not share shell environments.

`conan create --build=missing` exports the recipe and reuses a matching binary from the runner account's Conan cache, normally `~/.conan2` or the configured `CONAN_HOME`. Each build type needs its own binary. Recipe, dependency, and compiler changes can require a new build. Keep the cache between jobs to reuse packages on later pushes.

The [GitHub Actions workflow](../../../.github/workflows/cpp.yml) builds and tests both Debug and Release on every push, then runs `config/mav_headless.yaml` and `config/kitti_headless.yaml` for Release. The runner defaults are `MAV_PATH=/home/baldhat/data/mav/vicon_room1` and `KITTI_PATH=/home/baldhat/data/kitti`; repository Actions variables with the same names can override them. The KITTI root must contain `data_odometry_gray`, `data_odometry_calib`, and `data_odometry_poses`.

Each headless run has a 30-minute timeout, and both run even if the first fails. The job summary reports each config's validation status, matched pose count, tracking coverage, alignment scale, ATE and RPE RMSEs, FPS, and dataset percentage processed. Full application output is available in the headless step's job log; log files and copies of the configs used are also kept locally under `build/validation` on the runner. An application error, timeout, or missing/unsuccessful validation fails the step after both runs finish. No error thresholds are imposed.

To run the same evaluations locally, activate `conanrun.sh`, set the dataset paths, then run `bash recipes/run-headless-validation.sh` from `schlam/cpp`. Logs, config copies and a Markdown summary are written under `build/validation`. `SCHLAM_BUILD_DIR`, `SCHLAM_VALIDATION_DIR`, and `SCHLAM_HEADLESS_TIMEOUT` override the build directory, output directory, and per-dataset timeout.

The workflow uses a self hosted Linux x64 runner, its persistent Conan cache, and all processors reported by `nproc`. Pull requests build and test on fresh GitHub hosted runners with two build jobs; those runners build the local recipes from source. No private Conan remote or credentials are required.

On pushes, the `memory_profile` job runs after the dataset jobs succeed. It builds the Release app with `-O3 -DNDEBUG -g` so Heaptrack can attribute allocations to source locations, then profiles `config/mav_headless.yaml` using the same MAV dataset. The runner must already provide `heaptrack` and `heaptrack_print`; the job does not install them. Profiling is limited to 60 minutes, and an application failure or timeout fails the job.

The job summary shows Heaptrack's final statistics. The `heaptrack-release-<commit>-<attempt>` artifact contains the compressed capture, the full `heaptrack-report.txt`, the application log, the configuration, and the Heaptrack version. Artifacts are retained for 30 days; available captures and logs are uploaded even when profiling fails. Download the capture and open it with `heaptrack_gui` to inspect allocation call stacks, peak heap consumers, and temporary allocations. Heaptrack's allocation totals describe heap usage; they do not include every part of the process's resident memory.

See the [runner setup](../../../README.md#github-actions) for host dependencies and the dataset path.

CI allows five source download retries with ten seconds between attempts using `tools.files.download:retry` and `tools.files.download:retry_wait`. This helps with transient HTTP 504 responses from upstream source hosts. Rerunning a failed job reuses completed packages on the self hosted runner; a persistent download failure still requires the upstream URL to become reachable.
