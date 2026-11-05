# Local Conan packages

The application requires two packages built from pinned upstream source:

| Package | Upstream revision | Purpose |
| --- | --- | --- |
| `g2o/1.0.0.fd656273` | `fd65627348213d56866b24c0b7d054bcf105f0a5` | Keeps the revision previously used through `external/g2o` |
| `pangolin/0.9.4` | `b2346d6ba383342c835ef72e43abc3478434d16c` (`v0.9.4`) | Replaces the developer specific Pangolin build path |

The recipes verify SHA-256 checksums of both source archives. They build shared libraries and install each project's native CMake package config. The g2o recipe keeps logging support but disables unused solver families and GUI support. The Pangolin recipe keeps its OpenGL viewer and disables optional image and video backends. `conan.lock` pins the Conan recipe revisions of the application dependency graph.

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
cmake --build build/Release --parallel "$(nproc)"
ctest --test-dir build/Release --output-on-failure
```

Replace `Release` with `Debug` throughout for a Debug build. Keep your existing profile if you use a custom one; `conan profile detect --force` replaces the default profile.

`conan create --build=missing` exports the recipe and reuses a matching binary from the runner account's Conan cache, normally `~/.conan2` or the configured `CONAN_HOME`. Each build type needs its own binary. Recipe, dependency, and compiler changes can require a new build. Keep the cache between jobs to reuse packages on later pushes.

The [GitHub Actions workflow](../../../.github/workflows/cpp.yml) runs build, tests, and the installed MAV dataset headlessly on every push, in both Debug and Release. It uses a self hosted Linux x64 runner, its persistent Conan cache, and all processors reported by `nproc`. Pull requests build and test on fresh GitHub hosted runners with two build jobs; those runners build the local recipes from source. No private Conan remote or credentials are required.

See the [runner setup](../../../README.md#github-actions) for host dependencies and the dataset path.

CI allows five source download retries with ten seconds between attempts using `tools.files.download:retry` and `tools.files.download:retry_wait`. This helps with transient HTTP 504 responses from upstream source hosts. Rerunning a failed job reuses completed packages on the self hosted runner; a persistent download failure still requires the upstream URL to become reachable.
