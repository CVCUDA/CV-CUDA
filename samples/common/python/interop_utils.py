# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import torch
import copy


class CudaBuffer:
    __cuda_array_interface__ = None
    obj = None


def to_torch_dtype(data_type):
    """Convert a data type into one supported by torch

    Args:
        data_type (numpy dtype): Original data type

    Returns:
        dtype: A data type supported by torch
    """
    if data_type == np.uint16:
        return np.dtype(np.int16)
    elif data_type == np.uint32:
        return np.dtype(np.int32)
    elif data_type == np.uint64:
        return np.dtype(np.int64)
    else:
        return data_type


def to_cpu_numpy_buffer(cuda_buffer):
    """Convert a CUDA buffer to host (CPU) nympy array

    Args:
        cuda_buffer: CUDA buffer with __cuda_array_interface__

    Returns:
        numpy array: The CUDA buffer copied to the CPU
    """
    torch_dtype = copy.copy(cuda_buffer.dtype)
    torch_dtype = to_torch_dtype(torch_dtype)

    buf = CudaBuffer
    buf.obj = cuda_buffer
    buf.__cuda_array_interface__ = cuda_buffer.__cuda_array_interface__
    buf.__cuda_array_interface__["typestr"] = torch_dtype.str

    return torch.as_tensor(buf).cpu().numpy()


def to_cuda_buffer(host_data):
    """Convert host data to a CUDA buffer

    Args:
        host_data (numpy array): Host data

    Returns:
        CudaBuffer: The converted CUDA buffer
    """
    orig_dtype = copy.copy(host_data.dtype)

    host_data.dtype = to_torch_dtype(host_data.dtype)

    dev = torch.as_tensor(host_data, device="cuda").cuda()
    host_data.dtype = orig_dtype  # restore it

    # The cuda buffer only needs the cuda array interface.
    # We can then set its dtype to whatever we want.
    buf = CudaBuffer()
    buf.__cuda_array_interface__ = dev.__cuda_array_interface__
    buf.__cuda_array_interface__["typestr"] = orig_dtype.str
    buf.obj = dev  # make sure it holds a reference to the torch buffer

    return buf
