# CV-CUDA

[![License](https://img.shields.io/badge/License-Apache_2.0-yellogreen.svg)](https://opensource.org/licenses/Apache-2.0)

![Version](https://img.shields.io/badge/Version-v0.5.0--beta-blue)

![Platform](https://img.shields.io/badge/Platform-linux--64_%7C_win--64_wsl2-gray)

[![Cuda](https://img.shields.io/badge/CUDA-v11.7-%2376B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit-archive)
[![GCC](https://img.shields.io/badge/GCC-v11.0-yellow)](https://gcc.gnu.org/gcc-11/changes.html)
[![Python](https://img.shields.io/badge/python-v3.8_%7c_v3.10-blue?logo=python)](https://www.python.org/)
[![CMake](https://img.shields.io/badge/CMake-v3.20-%23008FBA?logo=cmake)](https://cmake.org/)

CV-CUDA is an open-source project that enables building efficient cloud-scale
Artificial Intelligence (AI) imaging and computer vision (CV) applications. It
uses graphics processing unit (GPU) acceleration to help developers build highly
efficient pre- and post-processing pipelines. CV-CUDA originated as a
collaborative effort between [NVIDIA][NVIDIA Develop] and [ByteDance][ByteDance].

Refer to our [Developer Guide](DEVELOPER_GUIDE.md) for more information on the
operators available as of release v0.5.0-beta.

## Getting Started

To get a local copy up and running follow these steps.

### Pre-requisites

- Linux distro:
  - Ubuntu x86_64 >= 20.04
  - WSL2 with Ubuntu >= 20.04 (tested with 20.04)
- NVIDIA driver
  - Linux: Driver version 520.56.06 or higher
- CUDA Toolkit
  - Version 11.7 or above.
- GCC >= 11.0
- Python >= 3.8
- cmake >= 3.20

### Installation

The following steps describe how to install CV-CUDA from pre-built install
packages. Choose the installation method that meets your environment needs.

#### Tar File Installation

```shell
tar -xvf nvcv-lib-0.5.0-cuda11-x86_64-linux.tar.xz
tar -xvf nvcv-dev-0.5.0-cuda11-x86_64-linux.tar.xz
```

#### DEB File Installation

```shell
sudo apt-get install -y ./nvcv-lib-0.5.0-cuda11-x86_64-linux.deb ./nvcv-dev-0.5.0-cuda11-x86_64-linux.deb
```

#### Python WHL File Installation

```shell
pip install nvcv_python-0.5.0-cp38-cp38-linux_x86_64.whl
```

### Build from Source

Building CV-CUDA from source allows for customization and is essential for contributing to the project. Here are detailed steps to guide you through the process:

#### 1. Repository Setup

   Before you begin, ensure you have cloned the CV-CUDA repository to your local machine. Let's assume you've cloned it into `~/cvcuda`.

   - **Initialize the Repository**:
     After cloning, initialize the repository to configure it correctly. This setup is required only once.

     ```shell
     cd ~/cvcuda
     ./init_repo.sh
     ```

#### 2. Install Build Dependencies

   CV-CUDA requires several dependencies to build from source. The following steps are based on Ubuntu 22.04, but similar packages can be found for other distributions.

   - **Install Essential Packages**:
     These include the compiler, build system, and necessary libraries.

     ```shell
     sudo apt-get install -y g++-11 cmake ninja-build python3-dev libssl-dev
     ```

   - **CUDA Toolkit**:
     The CUDA Toolkit is essential for GPU acceleration. Although any 11.x version is compatible, 11.7 is recommended.

     ```shell
     sudo apt-get install -y cuda-minimal-build-11-7
     ```

#### 3. Build Process

   Once the dependencies are in place, you can proceed to build CV-CUDA.

   - **Run Build Script**:
     A build script is provided to simplify the compilation process. It creates a build tree and compiles the source code.

     ```shell
     ci/build.sh
     ```

     This script creates a release build by default, placing output in `build-rel`. You can specify a debug build or a different output directory:

     ```shell
     ci/build.sh [release|debug] [output build tree path]
     ```

#### 4. Build Documentation (Optional)

   If you need to build the documentation, additional dependencies are required:

   - **Install Documentation Dependencies**:
     These tools are used to generate and format the documentation.

     ```shell
     sudo apt-get install -y doxygen graphviz python3 python3-pip
     sudo python3 -m pip install sphinx==4.5.0 breathe exhale recommonmark graphviz sphinx-rtd-theme
     ```

   - **Generate Documentation**:
     Use the provided script to build the documentation.

     ```shell
     ci/build_docs.sh [build folder]
     ```

     For example:

     ```shell
     ci/build_docs.sh build_docs
     ```

#### 5. Build and Run Samples (Optional)

   CV-CUDA comes with a variety of samples to demonstrate its capabilities.

   - **See the Samples Documentation**:
     Detailed instructions for building and running samples are available in the [Samples](samples/README.md) documentation.

#### 6. Running Tests

   To ensure everything is working as expected, you can run CV-CUDA's test suite.

   - **Install Test Dependencies**:
     These are necessary to run the Python binding tests.

     ```shell
     sudo apt-get install -y python3 python3-pip
     sudo python3 -m pip install pytest torch
     ```

   - **Execute Tests**:
     Run the test scripts located in the build tree.

     ```shell
     build-rel/bin/run_tests.sh
     ```

#### 7. Packaging

   After a successful build, you can create installers using `cpack`.

   - **Generate Installers**:
     This step produces Debian packages and tarballs, suitable for distribution or installation on other systems.

     ```shell
     cd build-rel
     cpack .
     ```

     For specific installer types:

     ```shell
     cpack . -G [DEB|TXZ]
     ```

     - `DEB` for Debian packages.
     - `TXZ` for `.tar.xz` tarballs.

## Contributing

CV-CUDA is an open source project. As part of the Open Source Community, we are
committed to the cycle of learning, improving, and updating that makes this
community thrive. However, as of release v0.5.0-beta, CV-CUDA is not yet ready
for external contributions.

To understand the process for contributing the CV-CUDA, see our
[Contributing](CONTRIBUTING.md) page. To understand our committment to the Open
Source Community, and providing an environment that both supports and respects
the efforts of all contributors, please read our
[Code of Conduct](CODE_OF_CONDUCT.md).

### CV-CUDA Make Operator Tool

The `mkop.sh` script is a powerful tool for creating a scaffold for new operators in the CV-CUDA library. It automates several tasks, ensuring consistency and saving time.

#### Features of `mkop.sh`:

1. **Operator Stub Creation**: Generates no-op (no-operation) operator templates, which serve as a starting point for implementing new functionalities.

1. **File Customization**: Modifies template files to include the new operator's name, ensuring consistent naming conventions across the codebase.

1. **CMake Integration**: Adds the new operator files to the appropriate CMakeLists, facilitating seamless compilation and integration into the build system.

1. **Python Bindings**: Creates Python wrapper stubs for the new operator, allowing it to be used within Python environments.

1. **Test Setup**: Generates test files for both C++ and Python, enabling immediate development of unit tests for the new operator.

#### How to Use `mkop.sh`:

Run the script with the desired operator name. The script assumes it's located in `/cvcuda/tools/mkop`.

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

## Security

CV-CUDA, as a NVIDIA program, is committed to secure development practices.
Please read our [Security](SECURITY.md) page to learn more.

## Acknowledgements

CV-CUDA is developed jointly by NVIDIA and ByteDance.

[NVIDIA Develop]: https://developer.nvidia.com/
[ByteDance]: https://www.bytedance.com/
