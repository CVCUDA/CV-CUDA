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

from ._load_binding import load_binding as _load_binding

# Dynamically load the appropriate binding
_binding = _load_binding(
    __name__,
    os.path.join(os.path.dirname(__file__), '_bindings')
)

# Import all symbols from the binding into the top-level namespace
__all__ = dir(_binding)
globals().update({symbol: getattr(_binding, symbol) for symbol in __all__})

# Clean up internal variables to avoid exposing them in the package namespace
del _load_binding, _binding, os
