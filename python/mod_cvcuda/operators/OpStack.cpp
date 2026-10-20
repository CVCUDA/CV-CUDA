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

#include "NativeBinding.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpStack.hpp>

#include <array>
#include <memory>
#include <stdexcept>

namespace cvcudapy {

namespace {

using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;
using native::TensorBatch;

class StackError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

void updateInputMetadata(const nvcv::Tensor &tensor, bool first, int32_t &totalTensors,
                         std::array<int64_t, 4> &outputShape, nvcv::TensorLayout &layout, nvcv::DataType &dtype)
{
    if (tensor.rank() < 3 || tensor.rank() > 4)
    {
        throw StackError("Invalid input tensor shape");
    }

    nvcv::TensorLayout tensorLayout;
    if (tensor.layout() == nvcv::TENSOR_CHW || tensor.layout() == nvcv::TENSOR_NCHW)
    {
        tensorLayout = nvcv::TENSOR_NCHW;
    }
    else if (tensor.layout() == nvcv::TENSOR_HWC || tensor.layout() == nvcv::TENSOR_NHWC)
    {
        tensorLayout = nvcv::TENSOR_NHWC;
    }
    else
    {
        throw StackError("Invalid input tensor layout");
    }

    const int32_t sampleOffset = tensor.rank() == 4 ? 1 : 0;
    if (first)
    {
        outputShape[1] = tensor.shape()[sampleOffset];
        outputShape[2] = tensor.shape()[sampleOffset + 1];
        outputShape[3] = tensor.shape()[sampleOffset + 2];
        layout         = tensorLayout;
        dtype          = tensor.dtype();
    }
    else if (outputShape[1] != tensor.shape()[sampleOffset] || outputShape[2] != tensor.shape()[sampleOffset + 1]
             || outputShape[3] != tensor.shape()[sampleOffset + 2] || layout != tensorLayout || dtype != tensor.dtype())
    {
        throw StackError("Input tensors must have the same sample shape, layout family, and data type");
    }

    totalTensors += tensor.rank() == 4 ? tensor.shape()[0] : 1;
}

void checkTensorList(const std::vector<std::shared_ptr<Tensor>> &tensorList, std::array<int64_t, 4> &outputShape,
                     nvcv::TensorLayout &layout, nvcv::DataType &dtype)
{
    int32_t totalTensors = 0;

    if (tensorList.empty())
    {
        throw StackError("Invalid input tensor list");
    }

    bool first = true;
    for (const auto &tensor : tensorList)
    {
        if (!tensor)
        {
            throw py::type_error("stack input tensor list cannot contain None");
        }

        updateInputMetadata(tensor->impl(), first, totalTensors, outputShape, layout, dtype);
        first = false;
    }
    outputShape[0] = totalTensors; // set N to total number of tensors
}

void checkTensorBatch(TensorBatch &tensorBatch, std::array<int64_t, 4> &outputShape, nvcv::TensorLayout &layout,
                      nvcv::DataType &dtype)
{
    if (tensorBatch.numTensors() == 0)
    {
        throw StackError("Invalid input tensor batch: empty batch");
    }

    int32_t totalTensors = 0;
    for (int32_t i = 0; i < tensorBatch.numTensors(); ++i)
    {
        updateInputMetadata(tensorBatch.at(i)->impl(), i == 0, totalTensors, outputShape, layout, dtype);
    }
    outputShape[0] = totalTensors;
}

void checkTensorBatchInto(TensorBatch &tensorBatch, std::array<int64_t, 4> &outputShape, nvcv::TensorLayout &layout,
                          nvcv::DataType &dtype)
{
    const int32_t numTensors = tensorBatch.numTensors();
    if (numTensors == 0)
    {
        throw StackError("Invalid input tensor batch: empty batch");
    }

    // Native Stack validates every input tensor during submission. For a rank-3 batch,
    // its output cardinality is already known, so another full scan here is redundant.
    if (tensorBatch.rank() == 3)
    {
        int32_t totalTensors = 0;
        updateInputMetadata(tensorBatch.at(0)->impl(), true, totalTensors, outputShape, layout, dtype);
        outputShape[0] = numTensors;
        return;
    }

    checkTensorBatch(tensorBatch, outputShape, layout, dtype);
}

void checkOutputMetadata(const Tensor &output, nvcv::TensorLayout layout, nvcv::DataType dtype)
{
    if (output.impl().layout() != layout || output.impl().dtype() != dtype)
    {
        throw StackError("Output tensor layout and data type must match the input tensors");
    }
}

void checkOutputShape(const Tensor &output, const std::array<int64_t, 4> &expectedShape)
{
    const nvcv::TensorShape &shape = output.impl().shape();
    if (shape.rank() != 4 || shape[0] != expectedShape[0] || shape[1] != expectedShape[1]
        || shape[2] != expectedShape[2] || shape[3] != expectedShape[3])
    {
        throw StackError("Invalid output tensor shape");
    }
}

void StackIntoInternal(Tensor &output, const std::vector<std::shared_ptr<Tensor>> &tensorList,
                       std::optional<Stream *> pstream, int32_t numberOfTensors)
{
    Stream &stream = native::GetStream(pstream);

    auto inTensorBatch = TensorBatch::Create(numberOfTensors);

    for (const auto &tensor : tensorList)
    {
        inTensorBatch->pushBack(*tensor);
    }

    auto op = CreateOperator<cvcuda::Stack>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {*inTensorBatch});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});
    guard.run([&op, &stream, &inTensorBatch, &output]()
              { op->submit(stream.handle(), inTensorBatch->impl(), output.impl()); });
}

Tensor &StackInto(Tensor &output, const std::vector<std::shared_ptr<Tensor>> &tensorList,
                  std::optional<Stream *> pstream)
{
    std::array<int64_t, 4> outputShape = {}; // NCHW/NHWC
    nvcv::TensorLayout     layout      = nvcv::TENSOR_CHW;
    nvcv::DataType         dtype;

    checkTensorList(tensorList, outputShape, layout, dtype);

    if (output.impl().layout() != nvcv::TENSOR_NCHW && output.impl().layout() != nvcv::TENSOR_NHWC)
        throw StackError("Invalid output tensor shape");

    checkOutputShape(output, outputShape);
    checkOutputMetadata(output, layout, dtype);

    StackIntoInternal(output, tensorList, pstream, static_cast<int32_t>(outputShape[0]));
    return output;
}

std::shared_ptr<Tensor> Stack(const std::vector<std::shared_ptr<Tensor>> &tensorList, std::optional<Stream *> pstream)
{
    std::array<int64_t, 4> outputShape = {}; // NCHW/NHWC
    nvcv::TensorLayout     layout      = nvcv::TENSOR_CHW;
    nvcv::DataType         dtype;
    checkTensorList(tensorList, outputShape, layout, dtype);

    //create new output tensor
    auto output = native::CreateTensor({outputShape.data(), static_cast<int32_t>(outputShape.size()), layout}, dtype);
    StackIntoInternal(*output, tensorList, pstream, static_cast<int32_t>(outputShape[0]));
    return output;
}

void StackTensorBatchIntoInternal(Tensor &output, TensorBatch &inTensorBatch, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto op = CreateOperator<cvcuda::Stack>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {inTensorBatch});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});
    guard.run([&op, &stream, &inTensorBatch, &output]()
              { op->submit(stream.handle(), inTensorBatch.impl(), output.impl()); });
}

// TensorBatch direct input functions
Tensor &StackTensorBatchInto(Tensor &output, TensorBatch &inTensorBatch, std::optional<Stream *> pstream)
{
    std::array<int64_t, 4> outputShape = {};
    nvcv::TensorLayout     layout      = nvcv::TENSOR_CHW;
    nvcv::DataType         dtype;
    checkTensorBatchInto(inTensorBatch, outputShape, layout, dtype);
    checkOutputShape(output, outputShape);
    checkOutputMetadata(output, layout, dtype);

    StackTensorBatchIntoInternal(output, inTensorBatch, pstream);
    return output;
}

std::shared_ptr<Tensor> StackTensorBatch(TensorBatch &inTensorBatch, std::optional<Stream *> pstream)
{
    std::array<int64_t, 4> outputShape = {};
    nvcv::TensorLayout     layout      = nvcv::TENSOR_NHWC;
    nvcv::DataType         dtype;
    checkTensorBatch(inTensorBatch, outputShape, layout, dtype);

    auto output = native::CreateTensor({outputShape.data(), static_cast<int32_t>(outputShape.size()), layout}, dtype);
    StackTensorBatchIntoInternal(*output, inTensorBatch, pstream);
    return output;
}

// ImageBatchVarShape input functions
Tensor &StackVarShapeInto(Tensor &output, ImageBatchVarShape &input, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto op = CreateOperator<cvcuda::Stack>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});
    guard.run([&op, &stream, &input, &output]() { op->submit(stream.handle(), input.impl(), output.impl()); });
    return output;
}

std::shared_ptr<Tensor> StackVarShape(ImageBatchVarShape &input, std::optional<Stream *> pstream)
{
    nvcv::ImageFormat fmt = input.impl().uniqueFormat();
    if (fmt == nvcv::FMT_NONE)
    {
        throw StackError("All images in the input must have the same format");
    }

    auto output = Tensor::CreateForImageBatch(input.numImages(), input.maxSize(), fmt, 0);

    StackVarShapeInto(*output, input, pstream);
    return output;
}

} // namespace

void ExportOpStack(py::module_ &m)
{
    using namespace nanobind::literals;

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
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
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
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
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
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
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
