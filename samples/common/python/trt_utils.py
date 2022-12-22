# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import numpy as np
import tensorrt as trt


def convert_onnx_to_tensorrt(
    onnx_file_path, trt_engine_file_path, max_batch_size, max_workspace_size=5
):
    """
    Converts an ONNX engine to a serialized TensorRT engine.
    :param onnx_file_path: Full path to an existing ONNX file.
    :param trt_engine_file_path: Full path to save the generated TensorRT Engine file.
    :param max_batch_size: The maximum batch size to use in the TensorRT engine.
    :param max_workspace_size: The maximum GPU memory that TensorRT can use (in GB.)
    :return: True if engine was generated. False otherwise.
    """
    print("Using TensorRT version: %s" % trt.__version__)
    trt_logger = trt.Logger(trt.Logger.INFO)

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
            print("Using precision : float16")
        else:
            print("Using precision : float32")

        # Parse model file
        print("Loading ONNX file from path %s " % onnx_file_path)
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return False

        for input_idx in range(network.num_inputs):
            new_shape = [max_batch_size]
            new_shape.extend(network.get_input(input_idx).shape[1:])
            network.get_input(input_idx).shape = new_shape

        for input_idx in range(network.num_inputs):
            print(
                "INPUT[%d] : %s %s"
                % (
                    input_idx,
                    network.get_input(input_idx).name,
                    network.get_input(input_idx).shape,
                )
            )
        for output_idx in range(network.num_outputs):
            print(
                "OUTPUT[%d] : %s %s"
                % (
                    output_idx,
                    network.get_output(output_idx).name,
                    network.get_output(output_idx).shape,
                )
            )
        print("Completed parsing of ONNX file.")

        print(
            "Building an engine from file %s. This may take a while..." % onnx_file_path
        )

        engine = builder.build_serialized_network(network, config)

        if engine:
            print("Completed creating Engine. Saving on disk...")
            with open(trt_engine_file_path, "wb") as f:
                f.write(engine)
                print("Saved to file %s" % trt_engine_file_path)
            return True
        else:
            print("Failed in creating the TensorRT engine.")
            return False


def setup_tensort_bindings(trt_model, device_id):
    """
    Setups the I/O bindings for a TensorRT engine for the first time.
    :param trt_model: Full path to the generated TensorRT Engine file.
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

        print(
            "TensorRT Binding[%d]: %s := shape: %s dtype: %s"
            % (b_idx, b_name, str(b_shape), b_dtype)
        )

        # Append to the appropriate list.
        if trt_model.get_tensor_mode(b_name) == trt.TensorIOMode.OUTPUT:
            # First allocate on device output buffers, using PyTorch.
            output = torch.zeros(
                size=b_shape, dtype=getattr(torch, b_dtype), device=device_id
            )

            print("\tAllocated the binding as an output.")
            # Since we know the name of our output layer, we will check against
            # it and grab its binding index.
            if b_name == "output":
                output_idx = output_binding_idx

            output_binding_idx += 1
            output_tensors.append(output)

    return output_tensors, output_idx
