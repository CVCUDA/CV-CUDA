..
  # SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

Pre-requisites
--------------

This section describes the recommended dependencies to install CV-CUDA.

* Ubuntu >= 20.04
* CUDA driver >= 11.7

Setup
-----

The following steps describe how to install CV-CUDA. Choose the installation method that meets your environment needs.
You can download the CV-CUDA tar, deb or wheel packages from `here <https://github.com/CVCUDA/CV-CUDA/releases/tag/v0.6.0-beta>`_

* Tar File Installation

    Unzip the cvcuda runtime package: ::

        tar -xvf cvcuda-lib-x.x.x-cuda11-x86_64-linux.tar.xz

    Unzip the cvcuda developer package: ::

        tar -xvf cvcuda-dev-x.x.x-cuda11-x86_64-linux.tar.xz

    Unzip the cvcuda python package: ::

        tar -xvf cvcuda-python3.*-x.x.x-cuda11-x86_64-linux.tar.xz

    [Optional] Unzip the tests. ::

        tar -xvf cvcuda-tests-cuda11-x86_64-linux.tar.xz


* Debian Installation

    Install the runtime library. ::

        dpkg -i cvcuda-lib-x.x.x-cuda11-x86_64-linux.deb

    Install the developer library. ::

        dpkg -i cvcuda-dev-x.x.x-cuda11-x86_64-linux.deb

    Install the python bindings ::

        dpkg -i cvcuda-python3.*-x.x.x-cuda11-x86_64-linux.deb

    [Optional] Install the tests. ::

        sudo dpkg -i cvcuda-tests-x.x.x-cuda11-x86_64-linux.deb


* Python Wheel File Installation

    Download the appropriate .whl file for your computer architecture, Python and CUDA version from `here <https://github.com/CVCUDA/CV-CUDA/releases/tag/v0.6.0-beta>`_

    Execute the following command to install appropriate CV-CUDA Python wheel ::

        pip install cvcuda_<cu_ver>-0.6.0b0-cp<py_ver>-cp<py_ver>-linux_<arch>.whl

    where <cu_ver> is the desired CUDA version, <py_ver> the desired Python version and <arch> the desired architecture.

    Please note that the Python wheels provided are standalone, they include both the C++/CUDA libraries and the Python bindings.


* Verifying the Debian or TAR installation on Linux

    To verify that CV-CUDA is installed and is running properly, run the tests from the install folder for tests.
    Default installation path is /opt/nvidia/cvcuda0/bin. ::

        cd /opt/nvidia/cvcuda0/bin
        ./run_tests.sh

If CV-CUDA is properly installed and running on your Linux system, all tests will pass.

* Running the samples on Linux. ::

    Follow the instructions written in the README.md file of the samples directory.
