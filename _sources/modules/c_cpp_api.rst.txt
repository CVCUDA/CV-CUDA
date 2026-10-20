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

.. _c_cpp_api:

C/C++ API
=========

The CV-CUDA C/C++ API provides both C and C++ interfaces to all functionality.

.. note::
   For Python developers, see the :ref:`Python API documentation <python_api>` for equivalent functionality.

Both libraries provide complete C and C++ APIs:

* **C API**: Functions and types with ``NVCV`` or ``cvcuda`` prefixes (e.g., ``NVCVImage``, ``cvcudaResize``)
* **C++ API**: Classes in ``nvcv`` and ``cvcuda`` namespaces (e.g., ``nvcv::Image``, ``cvcuda::Resize``)

The C++ API provides RAII wrappers around the C API with additional type safety and convenience.

NVCV - Core Types
-----------------

The NVCV library provides fundamental data types and containers for computer vision applications.

.. toctree::
   :glob:
   :maxdepth: 1

   ../_c_cpp_api/group__NVCV__C__CORE__*
   ../_c_cpp_api/group__NVCV__CPP__CORE__*
   ../_c_cpp_api/group__NVCV__C__UTIL__*
   ../_c_cpp_api/group__NVCV__CPP__UTIL__*
   ../_c_cpp_api/group__NVCV__C__TYPES
   ../_c_cpp_api/group__NVCV__CPP__CUDATOOLS__*

CV-CUDA - Operators
-------------------

The CV-CUDA library provides high-performance, GPU-accelerated computer vision and image processing operators.

.. toctree::
   :glob:
   :maxdepth: 1

   ../_c_cpp_api/group__NVCV__C__ALGORITHM__*
   ../_c_cpp_api/group__NVCV__CPP__ALGORITHM__*
