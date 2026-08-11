..
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _docker_images:

Docker Images
=============

CV-CUDA provides pre-built Docker images for development and building redistributable packages.
All images support both **x86_64 (AMD64)** and **aarch64 (ARM64)** architectures through
multi-architecture manifests.

Overview
--------

Two main categories of images:

1. **Builder Images** - Manylinux 2_28 based for creating redistributable packages (wheels, debs, tarballs)
2. **Development Images** - Ubuntu-based with complete development and testing environments

Docker automatically selects the appropriate architecture when pulling images.

Builder Images
--------------

Manylinux-based images with CUDA toolkit for building CV-CUDA packages compatible with a wide range of Linux distributions.

.. list-table:: Builder Image Variants
   :header-rows: 1
   :widths: 30 15 15 40

   * - Image Name
     - GCC Version
     - CUDA Version
     - Purpose
   * - builder_cu12.2.0_gcc10
     - 10
     - 12.2.0
     - CUDA 12.2 builds (multi-arch)
   * - builder_cu12.5.0_gcc10
     - 10
     - 12.5.0
     - CUDA 12.5 builds (multi-arch)
   * - builder_cu13.0.1_gcc10
     - 10
     - 13.0.1
     - CUDA 13.0 builds (multi-arch)
   * - builder_cu13.3.0_gcc10
     - 10
     - 13.3.0
     - CUDA 13.3 builds (multi-arch)

**Build Dependencies Hierarchy:**

.. code-block:: text

    ┌─────────────────┐    ┌─────────────────┐
    │   ManyLinux     │    │   Ubuntu 22.04  │
    └─────────┬───────┘    └─────────┬───────┘
              │                      │
              │ + GCC                │ + CUDA Toolkit
              │                      │
              ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐
    │ GCC Base Images │    │ CUDA Base Images│
    └─────────┬───────┘    └─────────┬───────┘
              │                      │
              │ BASE                 │ COPY CUDA
              └──────────┬───────────┘
                         │
                         │ Combine
                         ▼
               ┌─────────────────┐
               │ Builder Images  │
               └─────────────────┘

**Builder Image Features:**

- CMake 3.24.3
- Python 3.10-3.14 from ManyLinux
- Documentation tools (Sphinx 7.4.7/8.1.3, sphinx_rtd_theme, breathe)
- Development tools (patchelf 0.17.2, setuptools, wheel, clang 14.0)
- Full CUDA toolkit
- git-lfs for large files

Development Images
------------------

Full Ubuntu-based environments with multiple Python versions, NumPy, and PyTorch for development and testing.

Base: `NVIDIA CUDA images <https://hub.docker.com/r/nvidia/cuda/tags>`_ (nvidia/cuda:${CUDA_VER}-devel-ubuntu${UB_VER})

.. list-table:: Development Image Variants
   :header-rows: 1
   :widths: 35 15 10 10 10 20

   * - Image Name
     - Base Image
     - CUDA
     - NumPy
     - PyTorch
     - Python
   * - devel_u26.04_cu13.3.0_num2
     - ubuntu26.04
     - 13.3.0
     - 2.x
     - 2.11.0
     - 3.14
   * - devel_u22.04_cu12.5.0_num1
     - ubuntu22.04
     - 12.5.0
     - 1.26.4
     - 2.9.1
     - 3.10
   * - devel_u22.04_py310-314_cu12.5.0_num2
     - ubuntu22.04
     - 12.5.0
     - 2.x
     - 2.9.1
     - 3.10-3.14
   * - devel_u26.04_py310-314_cu13.3.0_num2
     - ubuntu26.04
     - 13.3.0
     - 2.x
     - 2.11.0
     - 3.10-3.14

The NumPy 1 image uses CuPy 13.6.0, the newest release compatible with NumPy
1.26. The NumPy 2 images use CuPy 14.0.1.

**Key Features:**

- Multiple GCC versions (10-13 on Ubuntu 22.04, 11-15 on Ubuntu 26.04)
- Multiple Clang versions (11 and 14 on Ubuntu 22.04, 18 on Ubuntu 26.04)
- CMake 3.24.3, ninja-build, ccache
- Testing frameworks (Google Test/Mock, pytest)
- ML frameworks (PyTorch, NumPy with version-specific wheels)
- Documentation tools (Doxygen, Sphinx)
- Development tools (git, git-lfs, pre-commit, shellcheck)

Version Management
------------------

All pinned Python package versions are defined in a single file at the repository root:
``versions.env``. This is the **only place** where versions should be changed.

After editing ``versions.env``, regenerate all requirements files:

.. code-block:: shell

    bash generate_requirements.sh

The generator rewrites the following files (do not edit them directly — they are
auto-generated and carry an ``AUTO-GENERATED`` header):

.. list-table:: Auto-Generated Requirements Files
   :header-rows: 1
   :widths: 45 55

   * - File
     - Contents
   * - tests/requirements.tests.cu12.txt
     - CuPy and CUDA-Python for CUDA 12.x
   * - tests/requirements.tests.cu12.numpy1.txt
     - NumPy 1-compatible CuPy and CUDA-Python for CUDA 12.x
   * - tests/requirements.tests.cu13.txt
     - CuPy and CUDA-Python for CUDA 13.x
   * - tests/requirements.tests.numpy1.txt
     - NumPy 1.x (Python 3.10-3.12)
   * - tests/requirements.tests.numpy2.txt
     - NumPy 2.x (Python 3.10-3.14)
   * - bench/python/requirements.bench.common.txt
     - Common benchmark dependencies
   * - bench/python/requirements.bench.cu12.txt
     - CUDA 12 benchmark dependencies
   * - bench/python/requirements.bench.cu13.txt
     - CUDA 13 benchmark dependencies
   * - samples/requirements.samples.common.txt
     - Common sample dependencies
   * - samples/requirements.samples.cu12.txt
     - CUDA 12 sample dependencies
   * - samples/requirements.samples.cu13.txt
     - CUDA 13 sample dependencies
   * - samples/requirements.samples.hello_world_cu12.txt
     - Minimal CUDA 12 hello-world dependencies
   * - samples/requirements.samples.hello_world_cu13.txt
     - Minimal CUDA 13 hello-world dependencies
   * - docker/requirements.build.sys_python.txt
     - System Python only: wheel building and linting tools
   * - docker/requirements.build.all_pythons.txt
     - All Python versions: pybind11 for CMake find_package
   * - tests/requirements.tests.common.txt
     - All Python versions: pytest and typing-extensions
   * - docs/requirements.docs.txt
     - System Python only: Sphinx documentation tools

The generator runs automatically in ``init_repo.sh`` (on clone) and ``docker/build_dockers.sh``
(before Docker builds). ``build.sh`` also runs the generator before each build to ensure
requirements files are always up to date. The pre-commit hook (triggered on changes to
``versions.env`` or any ``.template`` file) runs ``--check`` mode and fails if the generated
files are out of sync, forcing you to run ``bash generate_requirements.sh`` before committing.

Building the Images
-------------------

Use the ``build_dockers.sh`` script in the ``docker/`` directory.

**Usage:**

.. code-block:: shell

    # Build locally for native architecture only (default)
    ./build_dockers.sh

    # Explicitly force local build mode
    ./build_dockers.sh "" local

    # Build and push multi-arch images to registry
    ./build_dockers.sh $REGISTRY_PREFIX multiarch

**Modes:**

- ``local``: Build for native architecture only, load into local Docker (default when no registry)
- ``multiarch``: Build for both x86_64 and aarch64, push to registry (requires registry)



Using the Images
----------------

Running a  development image, mounting source code for development:

.. code-block:: shell

    docker run -it --gpus all \
      -v /path/to/cvcuda:/workspace \
      devel_u22.04_cu12.5.0_num1:v9


Using a builder image for creating manylinux-compatible wheels:

.. code-block:: shell

    docker run -it --gpus all \
      -v /path/to/cvcuda:/workspace \
      builder_cu12.5.0_gcc10:v9


Maintenance
-----------

Updating Package Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Edit ``versions.env`` at the repository root
2. Run ``bash generate_requirements.sh`` to regenerate all requirements files
3. Commit both ``versions.env`` and the regenerated files together

Updating Image Versions
^^^^^^^^^^^^^^^^^^^^^^^^

1. Increment ``VERSION`` variable in ``build_dockers.sh``
2. Run build script to create new image versions

Adding New CUDA Versions
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Create new ``Dockerfile.cuda{version}.deps`` with architecture detection
   - Use ``dpkg --print-architecture`` to detect amd64 vs arm64
   - Download appropriate CUDA installer (linux.run for x86_64, linux_sbsa.run for aarch64)
2. Add corresponding sections in ``build_dockers.sh``
3. Add the new CUDA version to ``versions.env`` and add any new package variants
4. Update development image variants

Adding New Python Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Builder images: Python versions come from base ManyLinux
- Development images: Update build arguments in ``build_dockers.sh``:

  .. code-block:: shell

      --build-arg "PYTHON_VERSIONS=3.10 3.11 3.12 3.13 3.14"

Troubleshooting
---------------

Build Failures
^^^^^^^^^^^^^^

- Verify Docker buildx is installed
- Ensure sufficient disk space for multi-stage builds
- Check network connectivity for downloading CUDA installers

Cache Issues
^^^^^^^^^^^^

- Clear build cache: ``docker system prune``
- Remove and recreate buildx builder:

  .. code-block:: shell

      docker buildx rm cvcuda_multiarch_builder

Registry Authentication
^^^^^^^^^^^^^^^^^^^^^^^

- Authenticate before using ``REGISTRY_PREFIX``
- Use ``docker login`` for private registries

See Also
--------

- :ref:`Building from Source <building-from-source>`
- :ref:`Installation Guide <installation>`
- `Docker Infrastructure README <https://github.com/CVCUDA/CV-CUDA/tree/main/docker>`_
