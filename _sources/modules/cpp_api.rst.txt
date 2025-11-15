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

.. _cpp_api:

C++ API
=======

The CV-CUDA C++ API provides RAII wrappers around the C API with additional type safety and convenience.

.. note::
   For C developers, see the :ref:`C API documentation <c_api>` for the low-level C interface.
   For Python developers, see the :ref:`Python API documentation <python_api>` for equivalent functionality.

The C++ API provides classes in ``nvcv`` and ``cvcuda`` namespaces (e.g., ``nvcv::Image``, ``cvcuda::Resize``).

NVCV - Core Types
-----------------

The NVCV library provides fundamental data types and containers for computer vision applications.

.. toctree::
   :glob:
   :maxdepth: 1

   ../_cpp_api/group__NVCV__CPP__CORE__*
   ../_cpp_api/group__NVCV__CPP__UTIL__*

NVCV - CUDA Tools
-----------------

The NVCV library provides CUDA utility classes and tools for device-side operations.

.. toctree::
   :glob:
   :maxdepth: 1

   ../_cpp_api/group__NVCV__CPP__CUDATOOLS__*

CV-CUDA - Operators
-------------------

The CV-CUDA library provides high-performance, GPU-accelerated computer vision and image processing operators.

.. toctree::
   :glob:
   :maxdepth: 1

   ../_cpp_api/group__NVCV__CPP__ALGORITHM__*
