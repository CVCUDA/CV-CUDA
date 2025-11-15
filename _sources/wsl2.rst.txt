..
  # SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _wsl2:

WSL2 Setup
==========

1. Install CUDA Toolkit
-----------------------

WSL2 uses the NVIDIA driver which is installed in your Windows host.
As such, you should ensure that you do not install a NVIDIA driver in your WSL2 distribution.
More information about WSL2 and using CUDA can be found on this page `CUDA Toolkit - WSL2 <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>`_.

For installation instructions, you can refer to the following links:

- `CUDA Toolkit 13.0`_, if you have driver r580 or later
- `CUDA Toolkit 12.8`_, if you have driver r525 or later

2. Setup Environment Variables
------------------------------

Once you have CUDA Toolkit installed, you need to setup the following environment variables in your WSL2 distribution.

.. code-block:: shell

    export CUDA_PATH=/usr/local/cuda
    export PATH=$CUDA_PATH/bin:$PATH
    export LIBRARY_PATH=$CUDA_PATH/lib64/stubs:$CUDA_PATH/lib64:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

You can use these commands to updates the environment variables from your ``~/.bashrc`` file automatically.

.. code-block:: shell

    echo 'export CUDA_PATH=/usr/local/cuda' >> ~/.bashrc
    echo 'export PATH=$CUDA_PATH/bin:$PATH' >> ~/.bashrc
    echo 'export LIBRARY_PATH=$CUDA_PATH/lib64/stubs:$CUDA_PATH/lib64:$LIBRARY_PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

3. Install CV-CUDA
------------------

Once you have completed the above steps, you can install CV-CUDA in your WSL2 distribution.
Please refer to the :ref:`Installation <installation>` guide for detailed instructions on installing CV-CUDA. A summary of the installation methods is provided below.

Method 1: Install from PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install CV-CUDA from PyPI using the following command:

.. code-block:: shell

    python3 -m pip install cvcuda-cu<CUDA_VERSION>

where ``<CUDA_VERSION>`` is the desired CUDA version, ``cu12`` or ``cu13``.

For example, to install CV-CUDA for CUDA 12:

.. code-block:: shell

    python3 -m pip install cvcuda-cu12

See :ref:`python-wheels-pypi` for more details.

Method 2: Install from Pre-built Packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download and install Debian packages or tar archives from the `CV-CUDA GitHub Releases`_ page.

- For Debian packages (.deb), see :ref:`debian-packages`
- For tar archives (.tar.xz), see :ref:`tar-archives`

Method 3: Build from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For building CV-CUDA from source in WSL2, follow the :ref:`building-from-source` instructions.

.. _CV-CUDA GitHub Releases: https://github.com/CVCUDA/CV-CUDA/releases
.. _CUDA Toolkit 13.0: https://developer.nvidia.com/cuda-13-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
.. _CUDA Toolkit 12.8: https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
