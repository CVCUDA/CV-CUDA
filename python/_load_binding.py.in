# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys
import importlib.util
import sysconfig
from functools import lru_cache


@lru_cache(maxsize=1)
def load_binding(module_name: str, bindings_dir: str):
    """
    Dynamically selects the correct binding for the current Python version
    """
    # Get the Python ABI tag
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    abi_tag = sysconfig.get_config_var('SOABI')

    # Construct the expected filename
    binding_so_filename = f'{module_name}.{abi_tag}.so'

    # Find the .so file in the package directory
    binding_so_path = os.path.join(bindings_dir, binding_so_filename)
    if not os.path.exists(binding_so_path):
        raise ImportError(
            f'Could not find the binding file for Python {python_version} at '
            f'{binding_so_path}. Make sure the package is installed.'
        )

    # Dynamically load the .so file
    spec = importlib.util.spec_from_file_location(module_name, binding_so_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Load a Python binding')
    parser.add_argument('module_name', type=str,
                        help='The name of the module to load')
    parser.add_argument('bindings_dir', type=str,
                        help='The directory containing the bindings')
    args = parser.parse_args()

    binding = load_binding(args.module_name, args.bindings_dir)

    print(f'Loaded module: {binding}')
    print(f'  Binding version: {binding.__version__}')
    print(f'  Binding description: {binding.__doc__}')
    print(f'  Binding functions: {dir(binding)}')
