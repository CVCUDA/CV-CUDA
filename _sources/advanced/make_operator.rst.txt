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

.. _make_operator:

CV-CUDA Make Operator Tool
===========================

The ``mkop.sh`` script is a tool for creating a scaffold for new operators in the CV-CUDA library. It automates several tasks, ensuring consistency and saving time.

Features of ``mkop.sh``
------------------------

1. **Operator Stub Creation**: Generates no-op (no-operation) operator templates, which serve as a starting point for implementing new functionalities.

2. **File Customization**: Modifies template files to include the new operator's name, ensuring consistent naming conventions across the codebase.

3. **CMake Integration**: Adds the new operator files to the appropriate CMakeLists, facilitating seamless compilation and integration into the build system.

4. **Python Bindings**: Creates Python wrapper stubs for the new operator, allowing it to be used within Python environments.

5. **Test Setup**: Generates test files for both C++ and Python, enabling immediate development of unit tests for the new operator.

How to Use ``mkop.sh``
-----------------------

Run the script with the desired operator name. The script assumes it's located in ``cvcuda/tools/mkop``.

.. code-block:: shell

    ./mkop.sh [Operator Name]

If the script is run from a different location, provide the path to the CV-CUDA root directory.

.. code-block:: shell

    ./mkop.sh [Operator Name] [CV-CUDA root]

**NOTE**: The first letter of the new operator name is captitalized where needed to match the rest of the file structures.

Process Details
---------------

* **Initial Setup**: The script begins by validating the input and setting up necessary variables. It then capitalizes the first letter of the operator name to adhere to naming conventions.

* **Template Modification**: It processes various template files (``Public.h``, ``PrivateImpl.cpp``, etc.), replacing placeholders with the new operator name. This includes adjusting file headers, namespaces, and function signatures.

* **CMake and Python Integration**: The script updates ``CMakeLists.txt`` files and Python module files to include the new operator, ensuring it's recognized by the build system and Python interface.

* **Testing Framework**: Finally, it sets up test files for both C++ and Python, allowing developers to immediately start writing tests for the new operator.
