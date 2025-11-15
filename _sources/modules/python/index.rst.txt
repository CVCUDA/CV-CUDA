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

.. _python_api:

Python API
==========

The CV-CUDA Python API provides a high-level interface to CV-CUDA functionality.

.. note::
   For C/C++ developers, see the :ref:`C API documentation <c_api>` and :ref:`C++ API documentation <cpp_api>` for equivalent functionality.

The Python API is available through the ``cvcuda`` module and provides:

* **Zero-copy interoperability** with PyTorch, CuPy, and other Python frameworks
* **Pythonic interfaces** for all operators and data types

.. toctree::
   :maxdepth: 1

   data_types
   operators
   auxiliary_types
   utilities
   cache
