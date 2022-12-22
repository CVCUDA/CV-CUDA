# CV-CUDA

[![License](https://img.shields.io/badge/License-Apache_2.0-yellogreen.svg)](https://opensource.org/licenses/Apache-2.0)

![Version](https://img.shields.io/badge/Version-v0.2.0--alpha-blue)

![Platform](https://img.shields.io/badge/Platform-linux--64_%7C_win--64_wsl2-gray)

[![Cuda](https://img.shields.io/badge/CUDA-v11.7-%2376B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit-archive)
[![GCC](https://img.shields.io/badge/GCC-v11.0-yellow)](https://gcc.gnu.org/gcc-11/changes.html)
[![Python](https://img.shields.io/badge/python-v3.7_%7c_v3.8_%7c_v3.9_%7c_v3.10-blue?logo=python)](https://www.python.org/)
[![CMake](https://img.shields.io/badge/CMake-v3.22-%23008FBA?logo=cmake)](https://cmake.org/)

CV-CUDA is an open-source project that enables building efficient cloud-scale
Artificial Intelligence (AI) imaging and computer vision (CV) applications. It
uses graphics processing unit (GPU) acceleration to help developers build highly
efficient pre- and post-processing pipelines. CV-CUDA originated as a
collaborative effort between [NVIDIA][NVIDIA Develop] and [ByteDance][ByteDance].

Refer to our [Developer Guide](DEVELOPER_GUIDE.md) for more information on the
operators avaliable as of release v0.2.0-alpha.

## Getting Started

To get a local copy up and running follow these steps.

### Pre-requisites

- Linux distro:
  - Ubuntu x86_64 >= 18.04
  - WSL2 with Ubuntu >= 20.04 (tested with 20.04)
- CUDA Driver >= 11.7 (Not tested on 12.0)
- GCC >= 11.0
- Python >= 3.7
- cmake >= 3.22

### Installation

The following steps describe how to install CV-CUDA from pre-built install
packages. Choose the installation method that meets your environment needs.

#### Tar File Installation

```
tar -xvf nvcv-lib-0.2.0-cuda11-x86_64-linux.tar.xz
```

#### DEB File Installation

```
sudo dpkg -i nvcv-lib-0.2.0-cuda11-x86_64-linux.deb
```

#### Python WHL File Installation

```
pip install nvcv_python-0.2.0-cp38-cp38-linux_x86_64.whl
```

### Build from Source

Follow these instruction to successfully build CV-CUDA from source:

1. Build CV-CUDA

   ```
   cd ~/cvcuda
   ci/build.sh
   ```

   This will compile a x86 release build of CV-CUDA inside `build-rel` directory.
   The library is in build-rel/lib, docs in build-rel/docs and executables
   (tests, etc...) in build-rel/bin.

   The script accepts some parameters to control the creation of the build tree:

   ```
   ci/build.sh [release|debug] [output build tree path]
   ```

   By default it builds for release.

   If output build tree path isn't specified, it'll be `build-rel` for release
   builds, and build-deb for debug.

1. Build Documentation

   ```
   ci/build_docs.sh [build folder]
   ```

   Example:
   `ci/build_docs.sh build

1. Build Samples

   ```
   ./ci/build_samples.sh [build folder]
   ```

   _(For instructions on how to compile samples outside of the CV-CUDA project,
   see the [Samples](samples/README.md) documentation)_

1. Run Tests

   The tests are in `<buildtree>/bin`. You can run the script below to run all
   tests at once. Here's an example when build tree is created in `build-rel`

   ```
   build-rel/bin/run_tests.sh
   ```

1. Run Samples

   The samples are installed in `<buildtree>/bin`. You can run the script below
   to download and serialize the model and run the sample with the test data
   provided.

   ```shell
   ./ci/run_samples.sh
   ```

1. Package installers

   From a succesfully built project, installers can be generated using cpack:

   ```shell
   cd build-rel
   cpack .
   ```

   This will generate in the build directory both Debian installers and tarballs
   (\*.tar.xz), needed for integration in other distros.

   For a fine-grained choice of what installers to generate, the full syntax is:

   ```
   cmake . -G [DEB|TXZ]
   ```

   - DEB for Debian packages
   - TXZ for \*.tar.xz tarballs.

## Contributing

CV-CUDA is an open source project. As part of the Open Source Community, we are
committed to the cycle of learning, improving, and updating that makes this
community thrive. However, as of release v0.2.0-alpha, CV-CUDA is not yet ready
for external contributions.

To understand the process for contributing the CV-CUDA, see our
[Contributing](CONTRIBUTING.md) page. To understand our committment to the Open
Source Community, and providing an environment that both supports and respects
the efforts of all contributors, please read our
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

CV-CUDA operates under the [Apache-2.0](LICENSE.md) license.

## Security

CV-CUDA, as a NVIDIA program, is committed to secure development practices.
Please read our [Security](SECURITY.md) page to learn more.

## Acknowledgements

CV-CUDA is developed jointly by NVIDIA and ByteDance.

[NVIDIA Develop]: https://developer.nvidia.com/
[ByteDance]: https://www.bytedance.com/
