/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpStack.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <nvcv/python/TensorBatch.hpp>

#include <array>
#include <stdexcept>

namespace cvcudapy {

namespace {

class StackError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

void checkTensorList(std::vector<Tensor> &tensorList, std::array<int64_t, 4> &outputShape, nvcv::TensorLayout &layout,
                     nvcv::DataType &dtype)
{
    int32_t totalTensors = 0;

    if (tensorList.empty())
    {
        throw StackError("Invalid input tensor list");
    }

    for (const auto &tensor : tensorList)
    {
        if (tensor.shape().rank() < 3 || tensor.shape().rank() > 4)
        {
            throw StackError("Invalid input tensor shape");
        }
        if (tensor.shape().rank() == 4)
        {
            totalTensors += tensor.shape()[0];
            outputShape[1] = tensor.shape()[1];
            outputShape[2] = tensor.shape()[2];
            outputShape[3] = tensor.shape()[3];
        }
        else
        {
            totalTensors++;
            outputShape[1] = tensor.shape()[0];
            outputShape[2] = tensor.shape()[1];
            outputShape[3] = tensor.shape()[2];
        }

        if (tensor.shape().layout() == nvcv::TENSOR_CHW || tensor.shape().layout() == nvcv::TENSOR_NCHW)
            layout = nvcv::TENSOR_NCHW;
        else
            layout = nvcv::TENSOR_NHWC;
    }
    outputShape[0] = totalTensors; // set N to total number of tensors
    dtype          = tensorList[0].dtype();
}

void StackIntoInternal(Tensor &output, std::vector<Tensor> &tensorList, std::optional<Stream> pstream,
                       int32_t numberOfTensors)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    nvcvpy::TensorBatch inTensorBatch = nvcvpy::TensorBatch::Create(numberOfTensors);

    for (const auto &tensor : tensorList)
    {
        inTensorBatch.pushBackTensor(tensor);
    }

    auto op = CreateOperator<cvcuda::Stack>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {inTensorBatch});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});
    guard.run([&op, &pstream, &inTensorBatch, &output]() { op->submit(pstream->cudaHandle(), inTensorBatch, output); });
}

Tensor StackInto(Tensor &output, std::vector<Tensor> &tensorList, std::optional<Stream> pstream)
{
    std::array<int64_t, 4> outputShape = {}; // NCHW/NHWC
    nvcv::TensorLayout     layout      = nvcv::TENSOR_CHW;
    nvcv::DataType         dtype;

    checkTensorList(tensorList, outputShape, layout, dtype);

    if (output.shape().layout() != nvcv::TENSOR_NCHW && output.shape().layout() != nvcv::TENSOR_NHWC)
        throw StackError("Invalid output tensor shape");

    if (output.shape()[0] != outputShape[0])
        throw StackError("Invalid output tensor shape");

    StackIntoInternal(output, tensorList, pstream, static_cast<int32_t>(outputShape[0]));
    return std::move(output);
}

Tensor Stack(std::vector<Tensor> &tensorList, std::optional<Stream> pstream)
{
    std::array<int64_t, 4> outputShape = {}; // NCHW/NHWC
    nvcv::TensorLayout     layout      = nvcv::TENSOR_CHW;
    nvcv::DataType         dtype;
    checkTensorList(tensorList, outputShape, layout, dtype);

    //create new output tensor
    Tensor output = Tensor::Create(
        {
            {outputShape[0], outputShape[1], outputShape[2], outputShape[3]},
            layout
    },
        dtype);
    StackIntoInternal(output, tensorList, pstream, static_cast<int32_t>(outputShape[0]));
    return output;
}

// TensorBatch direct input functions
Tensor StackTensorBatchInto(Tensor &output, nvcvpy::TensorBatch &inTensorBatch, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::Stack>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {inTensorBatch});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});
    guard.run([&op, &pstream, &inTensorBatch, &output]() { op->submit(pstream->cudaHandle(), inTensorBatch, output); });
    return std::move(output);
}

Tensor StackTensorBatch(nvcvpy::TensorBatch &inTensorBatch, std::optional<Stream> pstream)
{
    if (inTensorBatch.numTensors() == 0)
    {
        throw StackError("Invalid input tensor batch: empty batch");
    }

    // Get info from the first tensor to determine output shape and layout
    int64_t                totalTensors = 0;
    std::array<int64_t, 4> outputShape  = {};
    nvcv::TensorLayout     layout       = nvcv::TENSOR_NHWC;
    nvcv::DataType         dtype;

    for (int32_t i = 0; i < inTensorBatch.numTensors(); ++i)
    {
        nvcv::Tensor tensor = inTensorBatch[i];
        if (tensor.rank() < 3 || tensor.rank() > 4)
        {
            throw StackError("Invalid input tensor shape");
        }
        if (tensor.rank() == 4)
        {
            totalTensors += tensor.shape()[0];
            outputShape[1] = tensor.shape()[1];
            outputShape[2] = tensor.shape()[2];
            outputShape[3] = tensor.shape()[3];
        }
        else
        {
            totalTensors++;
            outputShape[1] = tensor.shape()[0];
            outputShape[2] = tensor.shape()[1];
            outputShape[3] = tensor.shape()[2];
        }
        if (tensor.layout() == nvcv::TENSOR_CHW || tensor.layout() == nvcv::TENSOR_NCHW)
            layout = nvcv::TENSOR_NCHW;
        else
            layout = nvcv::TENSOR_NHWC;
        if (i == 0)
            dtype = tensor.dtype();
    }
    outputShape[0] = totalTensors;

    Tensor output = Tensor::Create(
        {
            {outputShape[0], outputShape[1], outputShape[2], outputShape[3]},
            layout
    },
        dtype);
    return StackTensorBatchInto(output, inTensorBatch, pstream);
}

// ImageBatchVarShape input functions
Tensor StackVarShapeInto(Tensor &output, ImageBatchVarShape &input, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::Stack>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});
    guard.run([&op, &pstream, &input, &output]() { op->submit(pstream->cudaHandle(), input, output); });
    return std::move(output);
}

Tensor StackVarShape(ImageBatchVarShape &input, std::optional<Stream> pstream)
{
    nvcv::ImageFormat fmt = input.uniqueFormat();
    if (fmt == nvcv::FMT_NONE)
    {
        throw StackError("All images in the input must have the same format");
    }

    Tensor output = Tensor::CreateForImageBatch(input.numImages(), input.maxSize(), fmt);

    return StackVarShapeInto(output, input, pstream);
}

} // namespace

void ExportOpStack(py::module &m)
{
    using namespace pybind11::literals;

    // ImageBatchVarShape overloads (register first - most specific type)
    m.def("stack", NvtxTrace("cvcuda.stack", &StackVarShape), "src"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Stack operation on the given cuda stream. This takes an ImageBatchVarShape and combines images into a N(HWC/CHW) tensor.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images. All images must have the same format and dimensions.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor containing the stacked input images.

    )pbdoc");

    m.def("stack_into", NvtxTrace("cvcuda.stack_into", &StackVarShapeInto), "dst"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the Stack operation on the given cuda stream. This takes an ImageBatchVarShape and combines images into a N(HWC/CHW) tensor.


        Args:
            dst (cvcuda.Tensor): Output N(CHW/HWC) tensor to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images. All images must have the same format and dimensions.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    // TensorBatch overloads
    m.def("stack", NvtxTrace("cvcuda.stack", &StackTensorBatch), "src"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Stack operation on the given cuda stream. This takes a TensorBatch and combines tensors into a N(HWC/CHW) tensor.


        Args:
            src (cvcuda.TensorBatch): Input tensor batch containing one or more tensors. All tensors must be N(HWC/CHW) or HWC/CHW and have the same data type and shape.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor containing the stacked input tensors.

    )pbdoc");

    m.def("stack_into", NvtxTrace("cvcuda.stack_into", &StackTensorBatchInto), "dst"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the Stack operation on the given cuda stream. This takes a TensorBatch and combines tensors into a N(HWC/CHW) tensor.


        Args:
            dst (cvcuda.Tensor): Output N(CHW/HWC) tensor to store the result of the operation.
            src (cvcuda.TensorBatch): Input tensor batch containing one or more tensors. All tensors must be N(HWC/CHW) or HWC/CHW and have the same data type and shape.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    // List[Tensor] overloads (register last - most general type)
    m.def("stack", NvtxTrace("cvcuda.stack", &Stack), "src"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Stack operation on the given cuda stream. This takes input tensors and combines them into a N(HWC/CHW) tensor.


        Args:
            src (List[cvcuda.Tensor]): Input tensors containing one or more samples each images all tensors must be N(HWC/CHW) or HWC/CHW and have the same data type and shape.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor containing the stacked input tensors.

    )pbdoc");

    m.def("stack_into", NvtxTrace("cvcuda.stack_into", &StackInto), "dst"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the Stack operation on the given cuda stream. This takes input tensors and combines them into a N(HWC/CHW) tensor.


        Args:
            dst (cvcuda.Tensor): Output N(CHW/HWC) tensor to store the result of the operation.
            src (List[cvcuda.Tensor]): Input tensors containing one or more samples each images all tensors must be N(HWC/CHW) or HWC/CHW and have the same data type and shape.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
