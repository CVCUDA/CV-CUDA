# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
trt_utils

This file hosts various TensorRT related utilities.
"""

import logging
import torch
import numpy as np
import tensorrt as trt


def convert_onnx_to_tensorrt(
    onnx_file_path,
    trt_engine_file_path,
    max_batch_size,
    max_workspace_size=5,
):
    """
    Converts an ONNX engine to a serialized TensorRT engine.
    :param onnx_file_path: Full path to an existing ONNX file.
    :param trt_engine_file_path: Full path to save the generated TensorRT Engine file.
    :param max_batch_size: The maximum batch size to use in the TensorRT engine.
    :param max_workspace_size: The maximum GPU memory that TensorRT can use (in GB.)
    :return: True if engine was generated. False otherwise.
    """
    current_log_level = logging.root.level
    if current_log_level == logging.INFO:
        trt_log_level = trt.Logger.INFO
    elif current_log_level == logging.ERROR:
        trt_log_level = trt.Logger.ERROR
    elif current_log_level == logging.WARNING:
        trt_log_level = trt.Logger.WARNING
    elif current_log_level == logging.DEBUG:
        trt_log_level = trt.Logger.VERBOSE
    else:
        trt_log_level = trt.Logger.INFO

    trt_logger = trt.Logger(trt_log_level)
    trt_logger.log(trt.Logger.INFO, "Using TensorRT version: %s" % trt.__version__)

    with trt.Builder(trt_logger) as builder, builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, trt_logger) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024 * max_workspace_size
        )  # Sets workspace size in GB.
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            trt_logger.log(trt.Logger.INFO, "Using precision : float16")
        else:
            trt_logger.log(trt.Logger.INFO, "Using precision : float32")

        # Parse model file
        trt_logger.log(
            trt.Logger.INFO, "Loading ONNX file from path %s " % onnx_file_path
        )
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                raise ValueError("Failed to parse the ONNX engine.")

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        profile = builder.create_optimization_profile()
        dynamic_inputs = False
        for input_tensor in inputs:
            if input_tensor.shape[0] == -1:
                dynamic_inputs = True
                min_shape = [1] + list(input_tensor.shape[1:])
                opt_shape = [max_batch_size] + list(input_tensor.shape[1:])
                max_shape = [max_batch_size] + list(input_tensor.shape[1:])
                profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        if dynamic_inputs:
            config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, config)

        if not engine:
            raise ValueError("Failed to generate the TensorRT engine.")

        with open(trt_engine_file_path, "wb") as f:
            f.write(engine)
            trt_logger.log(
                trt.Logger.INFO, "Wrote TensorRT engine file: %s" % trt_engine_file_path
            )


def setup_tensort_bindings(trt_model, batch_size, device_id, logger):
    """
    Setups the I/O bindings for a TensorRT engine for the first time.
    :param trt_model: Full path to the generated TensorRT Engine file.
    :param batch_size: The maximum batch size that should be supported in the model.
    :param device_id: The GPU device id on which you want to allocated the buffers.
    :return: A list of output tensors and the index of the first output.
    """
    # For TensorRT, we need to allocate the output data buffers.
    # The input data buffers are already allocated by us.
    output_binding_idx = 0
    output_idx = 0
    output_tensors = []

    # Loop over all the I/O bindings.
    for b_idx in range(trt_model.num_io_tensors):
        # Get various properties associated with the bindings.
        b_name = trt_model.get_tensor_name(b_idx)
        b_shape = tuple(trt_model.get_tensor_shape(b_name))
        b_dtype = np.dtype(trt.nptype(trt_model.get_tensor_dtype(b_name))).name

        # Append to the appropriate list.
        if trt_model.get_tensor_mode(b_name) == trt.TensorIOMode.OUTPUT:
            # First allocate on device output buffers, using PyTorch.
            # Get the C, H, W dimensions from the layer shape to set the output layer size for the buffer
            output = torch.zeros(
                size=(batch_size, b_shape[-3], b_shape[-2], b_shape[-1]),
                dtype=getattr(torch, b_dtype),
                device="cuda:%d" % device_id,
            )
            # Since we know the name of our output layer, we will check against
            # it and grab its binding index.
            if b_name == "output":
                output_idx = output_binding_idx

            output_binding_idx += 1
            output_tensors.append(output)

    return output_tensors, output_idx
