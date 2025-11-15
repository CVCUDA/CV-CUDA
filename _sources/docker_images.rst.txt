..
  # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
   * - builder_cu12.5.0_gcc10
     - 10
     - 12.5.0
     - CUDA 12.5 builds (multi-arch)
   * - builder_cu12.9.0_gcc10
     - 10
     - 12.9.0
     - CUDA 12.9 builds (multi-arch)
   * - builder_cu13.0.1_gcc10
     - 10
     - 13.0.1
     - CUDA 13.0 builds (multi-arch)

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
- Python 3.9-3.14 from ManyLinux
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
  * - devel_u22.04_cu12.9.0_num2_torch2.8.0
    - ubuntu22.04
    - 12.9.0
    - 2.x
    - 2.8.0
    - 3.9-3.14
  * - devel_u22.04_cu12.5.0_num1_torch2.8.0
    - ubuntu22.04
    - 12.5.0
    - 1.26.4
    - 2.8.0
    - 3.9-3.14
   * - devel_u24.04_cu13.0.1_num2_torch2.9.0
     - ubuntu24.04
     - 13.0.1
     - 2.x
     - 2.9.0
     - 3.10-3.14

**Key Features:**

- Multiple GCC versions (10, 11, 12, 13, 14 on Ubuntu 24.04)
- Multiple Clang versions (11 on Ubuntu 22.04, 14 on all)
- CMake 3.24.3, ninja-build, ccache
- Testing frameworks (Google Test/Mock, pytest)
- ML frameworks (PyTorch, NumPy with version-specific wheels)
- Documentation tools (Doxygen, Sphinx)
- Development tools (git, git-lfs, pre-commit, shellcheck)

Python Requirements Files
--------------------------

CV-CUDA uses multiple requirements files for different purposes:

.. list-table:: Requirements Files
   :header-rows: 1
   :widths: 35 65

   * - File
     - Purpose
   * - requirements.sys_python.txt
     - System Python only: documentation (Sphinx, Breathe), wheel building (setuptools, wheel, build, patchelf, auditwheel), linting (flake8)
   * - requirements.no_torch_no_numpy.txt
     - All Python versions: testing tools (pytest, typing-extensions) - excludes PyTorch and NumPy
   * - requirements.numpy1.txt
     - NumPy 1.x for Python 3.9-3.12 (1.26.4). Not compatible with Python 3.13+
   * - requirements.numpy2.txt
     - NumPy 2.x with version constraints: 2.0.2 (Python 3.9), 2.2.6 (Python 3.10-3.13), 2.3.3 (Python 3.14)

PyTorch is installed separately per Python version in the Dockerfiles (not via requirements files).

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
      devel_u22.04_cu12.5.0_num1_torch2.8.0:v2


Using a builder image for creating manylinux-compatible wheels:

.. code-block:: shell

    docker run -it --gpus all \
      -v /path/to/cvcuda:/workspace \
      builder_cu12.5.0_gcc10:v1


Maintenance
-----------

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
3. Update development image variants

Adding New Python Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Builder images: Python versions come from base ManyLinux
- Development images: Update build arguments in ``build_dockers.sh``:

  .. code-block:: shell

      --build-arg "PYTHON_VERSIONS=3.9 3.10 3.11 3.12 3.13 3.14"

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
