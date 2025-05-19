
[//]: # "SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"
[//]: # ""
[//]: # "Licensed under the Apache License, Version 2.0 (the 'License');"
[//]: # "you may not use this file except in compliance with the License."
[//]: # "You may obtain a copy of the License at"
[//]: # "http://www.apache.org/licenses/LICENSE-2.0"
[//]: # ""
[//]: # "Unless required by applicable law or agreed to in writing, software"
[//]: # "distributed under the License is distributed on an 'AS IS' BASIS"
[//]: # "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied."
[//]: # "See the License for the specific language governing permissions and"
[//]: # "limitations under the License."

# CV-CUDA


[![License](https://img.shields.io/badge/License-Apache_2.0-yellogreen.svg)](https://opensource.org/licenses/Apache-2.0)

![Version](https://img.shields.io/badge/Version-v0.15.0--beta-blue)

![Platform](https://img.shields.io/badge/Platform-linux--64_%7C_win--64_wsl2%7C_aarch64-gray)

[![CUDA](https://img.shields.io/badge/CUDA-v11.7-%2376B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit-archive)
[![GCC](https://img.shields.io/badge/GCC-v11.0-yellow)](https://gcc.gnu.org/gcc-11/changes.html)
[![Python](https://img.shields.io/badge/python-v3.8_%7c_v3.9_%7c_v3.10%7c_v3.11%7c_v3.12%7c_v3.13-blue?logo=python)](https://www.python.org/)
[![CMake](https://img.shields.io/badge/CMake-v3.20-%23008FBA?logo=cmake)](https://cmake.org/)

CV-CUDA is an open-source project that enables building efficient cloud-scale
Artificial Intelligence (AI) imaging and computer vision (CV) applications. It
uses graphics processing unit (GPU) acceleration to help developers build highly
efficient pre- and post-processing pipelines. CV-CUDA originated as a
collaborative effort between [NVIDIA][NVIDIA Develop] and [ByteDance][ByteDance].

Refer to our [Developer Guide](DEVELOPER_GUIDE.md) for more information on the
operators available.

## Getting Started

To get a local copy up and running follow these steps.

### Compatibility

|CV-CUDA Build|Platform|CUDA Version|CUDA Compute Capability|Hardware Architectures|Nvidia Driver|Python Versions|Supported Compilers (build from source)|API compatibility with prebuilt binaries|OS/Linux distributions tested with prebuilt packages|
|-|-|-|-|-|-|-|-|-|-|
|x86_64_cu11|x86_64|11.7 or later|SM7 and later|Volta, Turing, Ampere, Ada Lovelace, Hopper|r525 or later*** |3.8 - 3.13|gcc>=9* <br> gcc>=11**|gcc>=9|ManyLinux2014-compliant, Ubuntu>= 20.04<br>WSL2/Ubuntu>=20.04|
|x86_64_cu12|x86_64|12.2 or later|SM7 and later|Volta, Turing, Ampere, Ada Lovelace, Hopper|r525 or later***|3.8 - 3.13|gcc>=9* <br> gcc>=11**|gcc>=9|ManyLinux2014-compliant, Ubuntu>= 20.04<br>WSL2/Ubuntu>=20.04|
|aarch64_cu11|aarch64 SBSA****|11.7 or later|SM7 and later|ARM SBSA (incl. Grace): Volta, Turing, Ampere, Ada Lovelace, Hopper|r525 or later***|3.8 - 3.13|gcc>=9* <br> gcc>=11**|gcc>=9|ManyLinux2014-compliant, Ubuntu>= 20.04|
|aarch64_cu12|aarch64 SBSA****|12.2 or later|SM7 and later|ARM SBSA (incl. Grace): Volta, Turing, Ampere, Ada Lovelace, Hopper|r525 or later***|3.8 - 3.13|gcc>=9* <br> gcc>=11**|gcc>=9|ManyLinux2014-compliant, Ubuntu>= 20.04|
|aarch64_cu11|aarch64 Jetson****|11.4|SM7 and later|Jetson AGX Orin|JetPack 5.1|3.8|gcc>=9* <br> gcc>=11**|gcc>=9|Jetson Linux 35.x|
|aarch64_cu12|aarch64 Jetson****|12.2|SM7 and later|Jetson AGX Orin, IGX Orin + Ampere RTX6000, IGX Orin + ADA RTX6000|JetPack 6.0 DP, r535 (IGX OS v0.6)|3.10|gcc>=9* <br> gcc>=11**|gcc>=9|Jetson Linux 36.2<br> IGX OS v0.6|

\* partial build, no test module (see Known Limitations) <br>
\** full build, including test module <br>
\*** [samples][CV-CUDA Samples] require driver r535 or later to run and are only officially supported with CUDA 12. <br>
\**** starting with v0.14, aarch64 packages (deb, tar.xz or wheels) distributed on Github (release "assets") or Pypi are SBSA-compatible unless noted otherwise. Jetson builds (deb, tar.xz, whl) can be found in explicitly named "Jetson" archives in Github release assets.

### Known limitations and issues

- Starting with v0.14, aarch64 packages (deb, tar.xz or wheels) distributed on Github (release "assets") and Pypi are the SBSA-compatible ones. Jetson builds (deb, tar.xz, whl) can be found in explicitly named "Jetson" archives in Github release assets.
- For GCC versions lower than 11.0, C++17 support needs to be enabled when compiling CV-CUDA.
- The C++ test module cannot build with gcc<11 (requires specific C++-20 features).  With gcc-9 or gcc-10, please build with option `-DBUILD_TESTS=0`
- [CV-CUDA Samples] require driver r535 or later to run and are only officially supported with CUDA 12.
- Only one CUDA version (CUDA 11.x or CUDA 12.x) of CV-CUDA packages (Debian packages, tarballs, Python Wheels) can be installed at a time. Please uninstall all packages from a given CUDA version before installing packages from a different version.
- Documentation built on Ubuntu 20.04 needs an up-to-date version of sphinx (`pip install --upgrade sphinx`) as well as explicitly parsing the system's default python version ` ./ci/build_docs path/to/build -DPYTHON_VERSIONS="<py_ver>"`.
- The Resize and RandomResizedCrop operators incorrectly interpolate pixel values near the boundary of an image or tensor when using cubic interpolation. This will be fixed in an upcoming release.

### Installation

For convenience, we provide pre-built packages for various combinations of CUDA versions, Python versions and architectures [here][CV-CUDA GitHub Releases].
The following steps describe how to install CV-CUDA from such pre-built packages.

We support two main alternative pathways:
- Standalone Python Wheels (containing C++/CUDA Libraries and Python bindings)
- DEB or Tar archive installation (C++/CUDA Libraries, Headers, Python bindings)

Choose the installation method that meets your environment needs.

#### Python Wheel File Installation

Check pypi.org projects for support for your platform of choice, [cvcuda-cu11][cvcuda-cu11] and [cvcuda-cu12][cvcuda-cu12] for CUDA 11 and CUDA 12, respectively.

Use the following command to install the latest available version:
   ```shell
   pip install cvcuda-<cu_ver>
   ```

where <cu_ver> is the desired CUDA version, 'cu11' or 'cu12'.

Alternatively, download the appropriate .whl file for your computer architecture, Python and CUDA version from the release assets of current CV-CUDA release. Release information of all CV-CUDA releases can be found [here][CV-CUDA GitHub Releases]. Once downloaded, execute the `pip install` command to install the Python wheel. For example:
   ```shell
   pip install ./cvcuda_<cu_ver>-<x.x.x>-cp<py_ver>-cp<py_ver>-linux_<arch>.whl
   ```

where `<cu_ver>` is the desired CUDA version, `<x.x.x>` is the CV-CUDA release version, `<py_ver>` is the desired Python version and `<arch>` is the desired architecture.

Please note that the Python wheels are standalone, they include both the C++/CUDA libraries and the Python bindings.

#### DEB File Installation

Install C++/CUDA libraries (cvcuda-lib*) and development headers (cvcuda-dev*) using `apt`:
```shell
sudo apt install -y ./cvcuda-lib-<x.x.x>-<cu_ver>-<arch>-linux.deb ./cvcuda-dev-<x.x.x>-<cu_ver>-<arch>-linux.deb
```

Install Python bindings (cvcuda-python*) using `apt`:
```shell
sudo apt install -y ./cvcuda-python<py_ver>-<x.x.x>-<cu_ver>-<arch>-linux.deb
```
where `<cu_ver>` is the desired CUDA version, `<py_ver>` is the desired Python version and `<arch>` is the desired architecture.

#### Tar File Installation

Install C++/CUDA libraries (cvcuda-lib*) and development headers (cvcuda-dev*):
```shell
tar -xvf cvcuda-lib-<x.x.x>-<cu_ver>-<arch>-linux.tar.xz
tar -xvf cvcuda-dev-<x.x.x>-<cu_ver>-<arch>-linux.tar.xz
```
Install Python bindings (cvcuda-python*)
```shell
tar -xvf cvcuda-python<py_ver>-<x.x.x>-<cu_ver>-<arch>-linux.tar.xz
```
where `<cu_ver>` is the desired CUDA version, `<py_ver>` is the desired Python version and `<arch>` is the desired architecture.


### Build from Source

Follow these instruction to build CV-CUDA from source:

#### 1. Set up your local CV-CUDA repository

Install the dependencies needed to setup up the repository:
- git
- git-lfs: to retrieve binary files from remote repository

On Ubuntu >= 20.04, install the following packages using `apt`:
```shell
sudo apt install -y git git-lfs
```

Clone the repository
```shell
git clone https://github.com/CVCUDA/CV-CUDA.git
```

Assuming the repository was cloned in `~/cvcuda`, it needs to be properly configured by running the `init_repo.sh` script only once.

```shell
cd ~/cvcuda
./init_repo.sh
```

#### 2. Build CV-CUDA

Install the dependencies required to build CV-CUDA:
- g++-11: compiler to be used
- cmake (>= 3.20), ninja-build (optional): manage build rules
- python3-dev: for python bindings
- libssl-dev: needed by the testsuite (MD5 hashing utilities)
- CUDA toolkit
- patchelf

On Ubuntu >= 20.04, install the following packages using `apt`:
```shell
sudo apt install -y g++-11 cmake ninja-build python3-dev libssl-dev patchelf
```

Any version of the 11.x or 12.x CUDA toolkit should work.
CV-CUDA was tested with 11.7 and 12.2, these versions are thus recommended.

```shell
sudo apt install -y cuda-11-7
# or
sudo apt install -y cuda-12-2
```

Build the project:
```shell
ci/build.sh [release|debug] [output build tree path] [-DBUILD_TESTS=1|0] [-DPYTHON_VERSIONS='3.8;3.9;3.10;3.11'] [-DPUBLIC_API_COMPILERS='gcc-9;gcc-11;clang-11;clang-14']
```

- The default build type is 'release'.
- If output build tree path isn't specified, it will be `build-rel` for release
      builds, and `build-deb` for debug.
- The library is in `build-rel/lib` and executables (tests, etc...) are in `build-rel/bin`.
- The `-DBUILD_TESTS` option can be used to disable/enable building the tests (enabled by default, see Known Limitations).
- The `-DPYTHON_VERSIONS` option can be used to select Python versions to build bindings and Wheels for. By default, only the default system Python3 version will be selected.
- The `-DPUBLIC_API_COMPILERS` option can be used to select the compilers used to check public API compatibility. By default, gcc-11, gcc-9, clang-11, and clang-14 is tried to be selected and checked.

#### 3. Build Documentation

Known limitation: Documentation built on Ubuntu 20.04 needs an up-to-date version of sphinx (`pip install --upgrade sphinx`) as well as explicitly parsing the system's default python version ` ./ci/build_docs path/to/build -DPYTHON_VERSIONS="<py_ver>"`.

Install the dependencies required to  build the documentation:
- doxygen: parse header files for reference documentation
- python3, python3-pip: to install some python packages needed
- sphinx, breathe, recommonmark, graphiviz: to render the documentation
- sphinx-rtd-theme: documentation theme used

On Ubuntu, install the following packages using `apt` and `pip`:
```shell
sudo apt install -y doxygen graphviz python3 python3-pip sphinx
python3 -m pip install breathe recommonmark graphviz sphinx-rtd-theme
```

Build the documentation:
```shell
ci/build_docs.sh [build folder]
```
Default build folder is 'build'.

#### 4. Build and run Samples

For instructions on how to build samples from source and run them, see the [Samples](samples/README.md) documentation.

#### 5. Run Tests

Install the dependencies required for running the tests:
- python3, python3-pip: to run python bindings tests
- torch: dependencies needed by python bindings tests

On Ubuntu >= 20.04, install the following packages using `apt` and `pip`:
```shell
sudo apt install -y python3 python3-pip
python3 -m pip install pytest torch numpy==1.26
```

The tests are in `<buildtree>/bin`. You can run the script below to run all tests at once. Here's an example when build tree is created in `build-rel`:
```shell
build-rel/bin/run_tests.sh
```

#### 6. Package installers and Python Wheels

Package installers

Installers can be generated using the following cpack command once you have successfully built the project:
```shell
cd build-rel
cpack .
```
This will generate in the build directory both Debian installers and tarballs (\*.tar.xz), needed for integration in other distros.

For a fine-grained choice of what installers to generate, the full syntax is:

```shell
cpack . -G [DEB|TXZ]
```
- DEB for Debian packages
- TXZ for \*.tar.xz tarballs.

Python Wheels

By default, during the `release` build, Python bindings and wheels are created for the available CUDA version and the specified Python version(s). The wheels are now output to the `build-rel/python3/repaired_wheels` folder (after being processed by the `auditwheel repair` command in the case of ManyLinux). The single generated python wheel is compatible with all versions of python specified during the cmake build step. Here, `build-rel` is the build directory used to build the release build.

The new Python wheels for PyPI compliance must be built within the ManyLinux 2014 Docker environment. The Docker images can be generated using the `docker/manylinux/docker_buildx.sh` script. These images ensure the wheels meet ManyLinux 2014 and PyPI standards.

The built wheels can still be installed using `pip`. For example, to install the Python wheel built for CUDA 12.x, Python 3.10 and 3.11 on Linux x86_64 systems:
```shell
pip install ./cvcuda_cu12-<x.x.x>-cp310.cp311-cp310.cp311-linux_x86_64.whl
```

## Contributing

CV-CUDA is an open source project. As part of the Open Source Community, we are
committed to the cycle of learning, improving, and updating that makes this
community thrive. However, CV-CUDA is not yet ready
for external contributions.

To understand the process for contributing the CV-CUDA, see our
[Contributing](CONTRIBUTING.md) page. To understand our commitment to the Open
Source Community, and providing an environment that both supports and respects
the efforts of all contributors, please read our
[Code of Conduct](CODE_OF_CONDUCT.md).

### CV-CUDA Make Operator Tool

The `mkop.sh` script is a powerful tool for creating a scaffold for new operators in the CV-CUDA library. It automates several tasks, ensuring consistency and saving time.

#### Features of `mkop.sh`:

1. **Operator Stub Creation**: Generates no-op (no-operation) operator templates, which serve as a starting point for implementing new functionalities.

2. **File Customization**: Modifies template files to include the new operator's name, ensuring consistent naming conventions across the codebase.

3. **CMake Integration**: Adds the new operator files to the appropriate CMakeLists, facilitating seamless compilation and integration into the build system.

4. **Python Bindings**: Creates Python wrapper stubs for the new operator, allowing it to be used within Python environments.

5. **Test Setup**: Generates test files for both C++ and Python, enabling immediate development of unit tests for the new operator.

#### How to Use `mkop.sh`:

Run the script with the desired operator name. The script assumes it's located in `~/cvcuda/tools/mkop`.

```shell
./mkop.sh [Operator Name]
```

If the script is run from a different location, provide the path to the CV-CUDA root directory.

```shell
./mkop.sh [Operator Name] [CV-CUDA root]
```

**NOTE**: The first letter of the new operator name is captitalized where needed to match the rest of the file structures.

#### Process Details:

- **Initial Setup**: The script begins by validating the input and setting up necessary variables. It then capitalizes the first letter of the operator name to adhere to naming conventions.

- **Template Modification**: It processes various template files (`Public.h`, `PrivateImpl.cpp`, etc.), replacing placeholders with the new operator name. This includes adjusting file headers, namespaces, and function signatures.

- **CMake and Python Integration**: The script updates `CMakeLists.txt` files and Python module files to include the new operator, ensuring it's recognized by the build system and Python interface.

- **Testing Framework**: Finally, it sets up test files for both C++ and Python, allowing developers to immediately start writing tests for the new operator.

## License

CV-CUDA operates under the [Apache-2.0](LICENSE.md) license.

## Third-party software redistribution

See [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES.md) for licenses of software redistributed as part of CV-CUDA's code or binary packages.

## Security

CV-CUDA, as a NVIDIA program, is committed to secure development practices.
Please read our [Security](SECURITY.md) page to learn more.

## Acknowledgements

CV-CUDA is developed jointly by NVIDIA and ByteDance.

References:

- [Optimizing Microsoft Bing Visual Search with NVIDIA Accelerated Libraries][bing-blog]
- [Accelerating AI Pipelines: Boosting Visual Search Efficiency, GTC 2025][bing-gtc25]
- [Optimize Short-Form Video Processing Toward the Speed of Light, GTC 2025][cosmos-splitting-gtc25]
- [CV-CUDA Increasing Throughput and Reducing Costs for AI-Based Computer Vision with CV-CUDA][increased-throughput-blog]
- [NVIDIA Announces Microsoft, Tencent, Baidu Adopting CV-CUDA for Computer Vision AI][cv-cuda-announcement]
- [CV-CUDA helps Tencent Cloud audio and video PaaS platform achieve full-process GPU acceleration for video enhancement AI][tencent-blog]

[NVIDIA Develop]: https://developer.nvidia.com/
[ByteDance]: https://www.bytedance.com/
[CV-CUDA GitHub Releases]: https://github.com/CVCUDA/CV-CUDA/releases
[CV-CUDA Samples]: https://github.com/CVCUDA/CV-CUDA/blob/main/samples/README.md
[cvcuda-cu11]: https://pypi.org/project/cvcuda-cu11/
[cvcuda-cu12]: https://pypi.org/project/cvcuda-cu12/

[bing-blog]: https://developer.nvidia.com/blog/optimizing-microsoft-bing-visual-search-with-nvidia-accelerated-libraries/
[bing-gtc25]: https://www.nvidia.com/en-us/on-demand/session/gtc25-s71676/
[cosmos-splitting-gtc25]: https://www.nvidia.com/en-us/on-demand/session/gtc25-s73178/
[increased-throughput-blog]: https://developer.nvidia.com/blog/increasing-throughput-and-reducing-costs-for-computer-vision-with-cv-cuda/
[cv-cuda-announcement]: https://blogs.nvidia.com/blog/2023/03/21/cv-cuda-ai-computer-vision/
[tencent-blog]: https://developer.nvidia.com/zh-cn/blog/cv-cuda-high-performance-image-processing/
