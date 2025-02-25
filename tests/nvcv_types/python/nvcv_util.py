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

import os
import threading
import numpy as np
import torch
import copy
from typing_extensions import Callable, Concatenate, ParamSpec

P = ParamSpec("P")


def to_cuda_buffer(host):
    orig_dtype = copy.copy(host.dtype)

    # torch doesn't accept uint16. Let's make it believe
    # it is handling int16 instead.
    if host.dtype == np.uint16:
        host.dtype = np.int16

    dev = torch.as_tensor(host, device="cuda").cuda()
    host.dtype = orig_dtype  # restore it

    class CudaBuffer:
        __cuda_array_interface = None
        obj = None

    # The cuda buffer only needs the cuda array interface.
    # We can then set its dtype to whatever we want.
    buf = CudaBuffer()
    buf.__cuda_array_interface__ = dev.__cuda_array_interface__
    buf.__cuda_array_interface__["typestr"] = orig_dtype.str
    buf.obj = dev  # make sure it holds a reference to the torch buffer

    return buf


def run_parallel(
    target: Callable[Concatenate[int, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
):
    """Run a callable in multiple threads and forward any exception to the main thread.

    Args:
        target (Callable): Callable to be run in multiple threads. The first argument is the thread index
        args: Positional arguments fowarded to the callable
        kwargs: Keyword arguments forwarded to the callable
    """

    def wrapper(thread_no: int):
        nonlocal exception

        barrier.wait()

        try:
            target(thread_no, *args, **kwargs)
        except Exception as exc:
            exception = exc

    nb_threads = len(os.sched_getaffinity(0))
    threads = [
        threading.Thread(target=wrapper, args=(idx,)) for idx in range(nb_threads)
    ]
    barrier = threading.Barrier(nb_threads)
    exception: Exception | None = None

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    if exception is not None:
        raise exception
