..
  # SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  #
  # Licensed under the Apache License, Version 2.0 (the "License");
  # you may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
  #
  # http://www.apache.org/licenses/LICENSE-2.0
  #
  # Unless required by applicable law or agreed to in writing, software
  # distributed under the License is distributed on an "AS IS" BASIS,
  # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  # See the License for the specific language governing permissions and
  # limitations under the License.

.. _installation:

Installation
============

CV-CUDA can be installed using pre-built packages or built from source. Choose the installation method that best fits your development workflow and requirements.

.. _prerequisites:

Prerequisites
-------------

Before installing CV-CUDA, ensure your system meets the following requirements:

* **Operating System**: Ubuntu >= 22.04
* **CUDA Toolkit**: CUDA >= 12.2
* **NVIDIA Driver**: r525 or later for CUDA 12.x, r580 or later for CUDA 13.x

If you are using WSL2, follow the instructions in the :ref:`WSL2 Setup <wsl2>`.

Using Pre-built Packages
-------------------------

We provide pre-built packages for various combinations of CUDA versions, Python versions and architectures.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Package
     - Description
     - Location
   * - Python Wheels
     - Standalone packages containing C++/CUDA libraries and Python bindings, recommended for Python users
     - `cvcuda-cu12`_, `cvcuda-cu13`_
   * - Debian Packages
     - System-level installation with separate library, development, Python modules and tests
     - `CV-CUDA GitHub Releases`_
   * - Tar Archives
     - Portable installation packages for various Linux distributions
     - `CV-CUDA GitHub Releases`_

.. _python-wheels-pypi:

Python Wheels from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~

Python wheels are available on `pypi.org <https://pypi.org>`_. Check `cvcuda-cu12`_ and `cvcuda-cu13`_ for CUDA 12 and CUDA 13 support for your platform of choice.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - CUDA Version
     - Instructions
   * - CUDA 12
     - .. code-block:: shell

          pip install cvcuda-cu12
   * - CUDA 13
     - .. code-block:: shell

          pip install cvcuda-cu13


.. _debian-packages:
.. _tar-archives:

Debian Packages and Tar Archives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Debian packages and tar archives can be found in the `CV-CUDA GitHub Releases`_ in the "assets" section.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Package
     - Debian Installation
     - Tar Archive Extraction
   * - C++/CUDA libraries and development headers
     - .. code-block:: shell

          sudo apt install -y \
            ./cvcuda-lib-<x.x.x>-<cu_ver>-<arch>-linux.deb \
            ./cvcuda-dev-<x.x.x>-<cu_ver>-<arch>-linux.deb
     - .. code-block:: shell

          tar -xvf cvcuda-lib-<x.x.x>-<cu_ver>-<arch>-linux.tar.xz
          tar -xvf cvcuda-dev-<x.x.x>-<cu_ver>-<arch>-linux.tar.xz
   * - Python bindings
     - .. code-block:: shell

          sudo apt install -y \
            ./cvcuda-python<py_ver>-<x.x.x>-<cu_ver>-<arch>-linux.deb
     - .. code-block:: shell

          tar -xvf cvcuda-python<py_ver>-<x.x.x>-<cu_ver>-<arch>-linux.tar.xz
   * - Tests
     - .. code-block:: shell

          sudo apt install -y \
            ./cvcuda-tests-<x.x.x>-<cu_ver>-<arch>-linux.deb
     - .. code-block:: shell

          tar -xvf cvcuda-tests-<x.x.x>-<cu_ver>-<arch>-linux.tar.xz

where ``<cu_ver>`` is the desired CUDA version, ``<py_ver>`` is the desired Python version and ``<arch>`` is the desired architecture.

.. _building-from-source:

Building from Source
--------------------

Building CV-CUDA from source provides the most flexibility and allows you to customize the build for your specific requirements.
Follow these instructions to build CV-CUDA from source:

.. note::
   **Recommended**: For the easiest setup, we provide pre-configured Docker images with all dependencies.
   See :ref:`Docker Images <docker_images>` for development and builder images supporting x86_64 and aarch64.

   The instructions below are for manual installation on your system.

1. Set up your local CV-CUDA repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the dependencies needed to setup up the repository:

- **git**
- **git-lfs**: to retrieve binary files from remote repository
- **pre-commit**: to lint and format code
- **shellcheck**: to lint shell scripts
- **npm**: required for some pre-commit hooks

On Ubuntu >= 22.04, install the following packages using ``apt``:

.. code-block:: shell

    sudo apt install -y git git-lfs pre-commit shellcheck npm

Clone the repository:

.. code-block:: shell

    git clone https://github.com/CVCUDA/CV-CUDA.git

Assuming the repository was cloned in ``~/cvcuda``, it needs to be properly configured by running the ``init_repo.sh`` script only once.

.. code-block:: shell

    cd ~/cvcuda
    ./init_repo.sh

2. Install build and test dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::
   **Using our development Docker images? Skip this section!**

   All these dependencies are pre-configured in our development Docker images.
   See the :ref:`Docker Images <docker_images>` documentation for the recommended setup.

The following table summarizes all dependencies needed to build and test CV-CUDA.

.. list-table:: Build and Test Dependencies
   :header-rows: 1
   :widths: 20 15 20 15 15 15

   * - Package Type
     - Core C++/CUDA Libraries
     - Python Bindings and Wheels
     - Test Building
     - Test Running
     - Documentation
   * - **Debian packages**
     - g++-11, cmake, ninja-build, CUDA toolkit
     - python3-dev
     - libgtest-dev, libgmock-dev, libssl-dev, zlib1g-dev
     - fonts-dejavu
     - doxygen, graphviz, python3, python3-pip, python3-sphinx
   * - **Python packages**
     - (none)
     - setuptools, wheel, build, patchelf, auditwheel
     - (none)
     - pytest, typing-extensions, numpy, torch
     - sphinx, sphinx-rtd-theme, breathe, exhale

**Installation Notes:**

- Install the CUDA Toolkit and a compatible NVIDIA driver by following NVIDIA's official
  `CUDA Toolkit Downloads`_ and `CUDA Installation Guide for Linux`_. CUDA package names,
  repository setup, and driver requirements vary by distribution and toolkit release.
- CUDA 12.2+ and CUDA 13.x should work. CV-CUDA was tested with 12.5 and 13.3; these versions are recommended.
- For NumPy: Use ``tests/requirements.tests.numpy1.txt`` for Python 3.10-3.12 (NumPy 1.26.4) or ``tests/requirements.tests.numpy2.txt`` for Python 3.10-3.14 (NumPy 2.2.6 on Python 3.10, NumPy 2.4.4 on Python 3.11+).
- PyTorch is installed separately and not included in requirements files.

If you are using WSL2, you can follow the instructions in the :ref:`WSL2 Setup <wsl2>`.

After CUDA is installed, install the remaining Debian build and test dependencies manually.
The following command intentionally excludes CUDA packages:

.. code-block:: shell

    sudo apt install -y \
        g++-11 cmake ninja-build \
        python3-dev python3-venv python3-pip \
        libgtest-dev libgmock-dev libssl-dev zlib1g-dev \
        fonts-dejavu doxygen graphviz

Python dependencies are specified with exact versions in ``docker/requirements.build.sys_python.txt`` (build/packaging tools) and ``docs/requirements.docs.txt`` (documentation tools).
The recommended method to install these Python dependencies is to use a Python virtual environment, with **venv** or **uv**.

1. Using ``venv``

.. code-block:: shell

    python3 -m venv env
    source env/bin/activate
    python3 -m pip install -r docker/requirements.build.sys_python.txt -r docs/requirements.docs.txt

2. Using ``uv``, importantly use the ``--seed`` flag to expose pip inside the virtual environment

.. code-block:: shell

    uv venv env --seed
    source env/bin/activate
    uv pip install -r docker/requirements.build.sys_python.txt -r docs/requirements.docs.txt

.. note::
   All these dependencies are pre-configured in our Docker images.
   See the :ref:`Docker Images <docker_images>` documentation for the recommended setup.

3. Build the Project
~~~~~~~~~~~~~~~~~~~~

The central ``build.sh`` script is used to build the project, Python bindings and wheels, tests and documentation by setting the appropriate CMake arguments.

.. code-block:: shell

    build.sh [release|debug] [output build tree path] [additional cmake args]

**Build Type:**

- ``release`` (default) or ``debug``

**Output Build Tree Path:**

- If not specified, defaults to ``build-rel`` for release builds, ``build-deb`` for debug builds
- The library is in ``<build-tree>/lib`` and executables (tests, etc.) are in ``<build-tree>/bin``

**Common CMake Arguments:**

- ``-DBUILD_PYTHON=1|0``: Enable/disable Python bindings and wheels (default: ON)
- ``-DBUILD_DOCS=1|0``: Enable/disable building documentation (default: disabled). Requires ``-DBUILD_PYTHON=1``.
- ``-DBUILD_TESTS=1|0``: Enable/disable test suite build (default: enabled). When enabled, automatically enables all test sub-options unless explicitly disabled
- ``-DBUILD_TESTS_CPP=1|0``: Enable/disable building C++ tests (see Known Limitations in README for GCC-10 restrictions)
- ``-DBUILD_TESTS_WHEELS=1|0``: Enable/disable generation of the wheel testing script
- ``-DBUILD_TESTS_PYTHON=1|0``: Enable/disable building Python tests
- ``-DBUILD_BENCH=1|0``: Enable/disable benchmark builds (default: OFF). Requires `nvbench <https://github.com/NVIDIA/nvbench>`_ to be installed (``cmake --install`` or system package). See ``bench/README.md`` for details.
- ``-DPYTHON_VERSIONS='3.10;3.11;3.12;3.13;3.14'``: Select Python versions to build bindings and wheels for (default: system Python3 only)
- ``-DPUBLIC_API_COMPILERS='gcc-10;gcc-11;clang-11;clang-14'``: Select compilers for public API compatibility checks (default: gcc-11, clang-11, clang-14)
- ``-DDOC_PYTHON_VERSION='3.11'``: Override Python version for documentation build (default: system Python)
- ``-DENABLE_SANITIZER=1|0``: Enable/disable address sanitizer (default: disabled)
- ``-DCMAKE_CUDA_COMPILER=/path/to/nvcc``: Override CUDA compiler (default: /usr/local/cuda/bin/nvcc)
- ``-DCMAKE_CUDA_ARCHITECTURES=<list>``: Override the complete CUDA architecture list. On x86_64,
  the generated default builds native SM75, SM80, and SM90 code (plus SM100 and SM120 with CUDA
  12.8 or newer). Performance-sensitive SM86 and SM89 kernels are added separately without
  compiling every CUDA source for those architectures.
- ``-DCVCUDA_TARGETED_SM8X_CUBINS=AUTO|ON|OFF``: Control the supplemental SM86 and SM89 operator
  cubins. ``AUTO`` enables them whenever the effective architecture list matches the compact
  x86_64 default, including an explicitly supplied equivalent list. ``ON`` ensures selected
  kernels have native images without duplicating globally enabled real targets, and ``OFF``
  disables them (default: ``AUTO``).
- ``-DCVCUDA_AARCH64_JETSON=ON|OFF``: aarch64 only -- build for Jetson Orin platforms only, targeting only Orin-relevant GPU architectures (sm_86, sm_87, sm_89) instead of the full SBSA set; package file names carry an ``aarch64-jetson-linux`` token instead of ``aarch64-linux`` so Jetson and SBSA artifacts stay distinguishable (default: OFF)

All boolean options accept both numeric (``0``/``1``) and CMake boolean values (``ON``/``OFF``, ``YES``/``NO``, ``TRUE``/``FALSE``).

**Environment Variables:**

- ``CC``, ``CXX``: Specify C/C++ compilers (default: auto-detected gcc-11 or newer)
- ``CUDAARCHS``: Set the complete CUDA architecture list when
  ``CMAKE_CUDA_ARCHITECTURES`` is not already cached. The value is not combined with CV-CUDA's
  generated defaults.

**Examples:**

.. code-block:: shell

    # Basic release build with default Python
    build.sh

    # Build documentation
    build.sh release build-rel -DBUILD_DOCS=1 -DBUILD_PYTHON=1

    # Debug build with multiple Python versions
    build.sh debug build-debug -DPYTHON_VERSIONS='3.10;3.11;3.12'

    # Release build without tests
    build.sh release -DBUILD_TESTS=0

    # Build with specific compiler
    CC=gcc-12 CXX=g++-12 build.sh

    # Build with specific CUDA 13 version
    build.sh -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13/bin/nvcc

    # Build for Jetson Orin platforms (aarch64 only)
    build.sh release build-rel "-DCVCUDA_AARCH64_JETSON=ON"


1. Run Tests
~~~~~~~~~~~~

Prerequisite:
- project built with ``-DBUILD_TESTS=1``, ``-DBUILD_TESTS_CPP=1``, ``-DBUILD_TESTS_WHEELS=1`` or ``-DBUILD_TESTS_PYTHON=1``.
- OR tests packages installed from debian packages (see :ref:`Debian Packages <debian-packages>`) or TAR archives (see :ref:`Tar Archives <tar-archives>`).

4.1 Install dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

.. important::
   **Using our development Docker images? Skip this section!**

   See :ref:`Docker Images <docker_images>`


Prerequisites: install system Python dependencies (see step 2).
On top of that, the following dependencies are required for running the tests:

- **numpy**: required by python bindings tests
- **typing-extensions**: required by python bindings tests
- **pytest**: to run the tests
- **cupy**: required by the Python test runner and CUDA array tests

Package versions are defined in ``versions.env`` at the repository root and auto-generated
into the requirements files below. To update a version, edit ``versions.env`` and run
``bash generate_requirements.sh``.

- ``tests/requirements.tests.common.txt`` for pytest and typing-extensions (manually maintained)
- ``tests/requirements.tests.numpy1.txt`` for NumPy 1.x (Python 3.10-3.12) — auto-generated
- ``tests/requirements.tests.numpy2.txt`` for NumPy 2.x (Python 3.10-3.14) — auto-generated
- ``tests/requirements.tests.cu12.txt`` for CuPy with CUDA 12.x (compatible with CUDA 12.2+) — auto-generated
- ``tests/requirements.tests.cu12.numpy1.txt`` for the NumPy 1-compatible CuPy pin with CUDA 12.x — auto-generated
- ``tests/requirements.tests.cu13.txt`` for CuPy with CUDA 13.x (compatible with CUDA 13.3+) — auto-generated

The easiest way to install test dependencies is to use the provided script:

.. code-block:: shell

    # For NumPy 2.x + CuPy for CUDA 12.x (CUDA 12.2+)
    tests/install_test_dependencies.sh numpy2 cu12

    # For NumPy 1.x + CuPy for CUDA 12.x (Python 3.10-3.12)
    tests/install_test_dependencies.sh numpy1 cu12

    # For NumPy 2.x + CuPy for CUDA 13.x (13.3+)
    tests/install_test_dependencies.sh numpy2 cu13

Alternatively, install the dependencies manually into your virtual environment (setup in step 2):

1. Using ``venv``

.. code-block:: shell

    # For CUDA 12.2+, NumPy 2.x
    python3 -m pip install -r tests/requirements.tests.common.txt -r tests/requirements.tests.numpy2.txt -r tests/requirements.tests.cu12.txt

    # For CUDA 12.2+, NumPy 1.x
    python3 -m pip install -r tests/requirements.tests.common.txt -r tests/requirements.tests.numpy1.txt -r tests/requirements.tests.cu12.numpy1.txt

    # For CUDA 13.x (13.3+), NumPy 2.x
    python3 -m pip install -r tests/requirements.tests.common.txt -r tests/requirements.tests.numpy2.txt -r tests/requirements.tests.cu13.txt

2. Using ``uv``:

.. code-block:: shell

    # For CUDA 12.2+, NumPy 2.x
    uv pip install -r tests/requirements.tests.common.txt -r tests/requirements.tests.numpy2.txt -r tests/requirements.tests.cu12.txt

    # For CUDA 12.2+, NumPy 1.x
    uv pip install -r tests/requirements.tests.common.txt -r tests/requirements.tests.numpy1.txt -r tests/requirements.tests.cu12.numpy1.txt

    # For CUDA 13.x (13.3+), NumPy 2.x
    uv pip install -r tests/requirements.tests.common.txt -r tests/requirements.tests.numpy2.txt -r tests/requirements.tests.cu13.txt

4.2 Run the Tests
^^^^^^^^^^^^^^^^^

The test location depends on how CV-CUDA was installed:

- **Built from source**: Tests are in ``<buildtree>/bin/`` (e.g., ``build-rel/bin/``)
- **Installed from packages**: Tests are in ``/opt/nvidia/cvcuda*/bin/``

Run all tests at once using the test runner script:

.. code-block:: shell

    # If built from source (example with build-rel)
    build-rel/bin/run_tests.sh [filter1,filter2,...]

    # If installed from packages
    /opt/nvidia/cvcuda*/bin/run_tests.sh [filter1,filter2,...]

**Test Filters:**

You can optionally specify comma-separated filters to run specific test subsets:

- ``all`` - Run all tests (default if no filter specified)
- ``cvcuda`` - Run all CV-CUDA tests
- ``nvcv`` - Run all NVCV (core types library) tests
- ``cpp`` - Run all C++ tests
- ``python`` - Run all Python tests

**Examples:**

.. code-block:: shell

    # Run all tests (default)
    build-rel/bin/run_tests.sh

    # Run only CV-CUDA tests
    build-rel/bin/run_tests.sh cvcuda

    # Run only C++ tests
    build-rel/bin/run_tests.sh cpp

    # Run only C++ CV-CUDA tests (combines filters)
    build-rel/bin/run_tests.sh cvcuda,cpp

    # Run only Python NVCV tests
    build-rel/bin/run_tests.sh nvcv,python

5. Build and run Samples
~~~~~~~~~~~~~~~~~~~~~~~~
For instructions on how to build samples from source and run them, see the Samples documentation `here <https://github.com/CVCUDA/CV-CUDA/blob/main/samples/README.md>`_.

6. Packaging
~~~~~~~~~~~~~

6.1 Package installers
^^^^^^^^^^^^^^^^^^^^^^

Installers can be generated using the following cpack command once you have successfully built the project:

.. code-block:: shell

    cd build-rel
    cpack .

This will generate in the build directory both Debian installers and tarballs (\*.tar.xz), needed for integration in other distros.

For a fine-grained choice of what installers to generate, the full syntax is:

.. code-block:: shell

    cpack . -G [DEB|TXZ]

- DEB for Debian packages
- TXZ for \*.tar.xz tarballs.

6.2 Python Wheels
^^^^^^^^^^^^^^^^^

By default, during the ``release`` build, Python bindings and wheels are created for the available CUDA version and the specified Python version(s). The wheels are now output to the ``build-rel/python3/repaired_wheels`` folder (after being processed by the ``auditwheel repair`` command in the case of ManyLinux). The single generated python wheel is compatible with all versions of python specified during the cmake build step. Here, ``build-rel`` is the build directory used to build the release build.

The new Python wheels for PyPI compliance must be built within the ManyLinux_2_28 Docker environment. See :ref:`Docker Images <docker_images>` for pre-built images and build instructions. These images ensure the wheels meet ManyLinux_2_28 and PyPI standards.

The built wheels can still be installed using ``pip``. For example, to install the Python wheel built for CUDA 12.x, Python 3.10 and 3.11 on Linux x86_64 systems:


1. Method 1: Using ``venv``

.. code-block:: shell

    python3 -m pip install ./cvcuda_cu12-<x.x.x>-cp310.cp311-cp310.cp311-linux_x86_64.whl

2. Method 2: Using ``uv``:

.. code-block:: shell

    uv pip install ./cvcuda_cu12-<x.x.x>-cp310.cp311-cp310.cp311-linux_x86_64.whl


.. _CV-CUDA GitHub Releases: https://github.com/CVCUDA/CV-CUDA/releases
.. _CUDA Toolkit Downloads: https://developer.nvidia.com/cuda-downloads
.. _CUDA Installation Guide for Linux: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
.. _cvcuda-cu12: https://pypi.org/project/cvcuda-cu12/
.. _cvcuda-cu13: https://pypi.org/project/cvcuda-cu13/
