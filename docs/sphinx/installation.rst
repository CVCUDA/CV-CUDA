..
  # SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

This section describes the recommended dependencies to compile cvcuda

* Ubuntu >= 20.04
* CUDA driver >= 11.7

Setup
-----

The following steps describe how to install cvcuda. Choose the installation method that meets your environment needs.

Download the cvcuda tar/deb package from `here <https://github.com/CVCUDA/CV-CUDA/releases/tag/v0.2.0-alpha>`_

* Tar File Installation

Navigate to your <cvcudapath> directory containing the cvcuda tar file.

Unzip the cvcuda runtime package: ::

    tar -xvf nvcv-lib-x.x.x-cuda11-x86_64-linux.tar.xz

Unzip the cvcuda developer package: ::

    tar -xvf nvcv-dev-x.x.x-cuda11-x86_64-linux.tar.xz

Unzip the cvcuda python package: ::

    tar -xvf nvcv-python3.*-x.x.x-cuda11-x86_64-linux.tar.xz

Optionally Unzip the tests. ::

    tar -xvf cvcuda-tests-cuda11-x86_64-linux.tar.xz

Optionally Unzip the tests. ::

    tar -xvf cvcuda-tests-cuda11-x86_64-linux.tar.xz

* Debian Local Installation

Navigate to your <cvcudapath> directory containing the cvcuda Debian local installer file. ::

Install the runtime library. ::

    sudo dpkg -i nvcv-lib-x.x.x-cuda11-x86_64-linux.deb

Install the developer library. ::

    sudo dpkg -i nvcv-dev-x.x.x-cuda11-x86_64-linux.deb

Install the python bindings ::

    sudo dpkg -i nvcv-python3.*-x.x.x-cuda11-x86_64-linux.deb

Optionally install the tests. ::

    sudo dpkg -i cvcuda-tests-x.x.x-cuda11-x86_64-linux.deb

Optionally install the samples. ::

    sudo dpkg -i cvcuda-samples-x.x.x-cuda11-x86_64-linux.deb

* Verifying the Installation on Linux

To verify that cvcuda is installed and is running properly, run the tests from the install folder for tests.
Default installation path is /opt/nvidia/cvcuda0/bin. ::

    cd /opt/nvidia/cvcuda0/bin
    ./run_tests.sh

If CV-CUDA is properly installed and running on your Linux system, all tests will pass.

* Running the samples on Linux. ::

    cd /opt/nvidia/cvcuda0/samples
    ./scripts/install_dependencies.sh
    ./scripts/run_samples.sh
