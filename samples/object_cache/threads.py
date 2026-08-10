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

from __future__ import annotations

# docs-start: main
import threading
import time
import cvcuda
import numpy as np


def create_tensor_and_clear():
    tensor = cvcuda.Tensor(  # noqa: F841
        (16, 32, 4), np.float32, cvcuda.TensorLayout.HWC
    )
    print(cvcuda.cache_size(), cvcuda.cache_size(cvcuda.ThreadScope.LOCAL))  # 2 1
    cvcuda.clear_cache(cvcuda.ThreadScope.LOCAL)
    print(cvcuda.cache_size(), cvcuda.cache_size(cvcuda.ThreadScope.LOCAL))  # 1 0


def main() -> None:
    tensor = cvcuda.Tensor(  # noqa: F841
        (16, 32, 4), np.float32, cvcuda.TensorLayout.HWC
    )
    thread = threading.Thread(target=create_tensor_and_clear)
    thread.start()
    thread.join()

    # WORKAROUND: Race condition between Python thread.join() and C++ thread-local destructors
    #
    # Root Cause:
    # Python's thread.join() returns when the Python thread function completes, but
    # C++ thread-local objects (like the Cache instance) are destroyed asynchronously
    # by the C++ runtime AFTER Python considers the thread "joined". Without this sleep,
    # the main() function exits immediately after join(), triggering the main thread's
    # Cache destructor while the worker thread's Cache destructor is still running.
    # Both destructors then concurrently attempt to clean up CUDA resources (acquire GIL,
    # destroy events, deallocate memory), causing race conditions and segfaults.
    #
    # The sleep keeps the main thread alive longer, ensuring temporal separation between
    # the worker's destructor completing and the main thread's destructor starting.
    #
    # Why This Is Not An Issue In Normal Usage:
    # - Real applications use long-lived thread pools that outlive individual operations
    # - Main threads typically continue running after worker threads complete (serving
    #   requests, processing more data, etc.) providing natural timing separation
    # - This test has the worst-case scenario: main thread exits immediately after
    #   worker joins, with no other work keeping it alive
    #
    # A proper fix would require C++ synchronization primitives (condition variables) to
    # explicitly wait for C++ thread-local destructors, but adds complexity for a rare edge case.
    time.sleep(0.1)  # Keep main thread alive while worker's C++ destructor completes


# docs-end: main

if __name__ == "__main__":
    main()
