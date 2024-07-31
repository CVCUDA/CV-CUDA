/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <nvcv/python/TensorBatch.hpp>

namespace cvcudapy {

namespace {

void checkTensorList(std::vector<Tensor> &tensorList, int64_t (&outputShape)[4], nvcv::TensorLayout &layout,
                     nvcv::DataType &dtype)
{
    int32_t totalTensors = 0;

    if (tensorList.size() == 0)
    {
        throw std::runtime_error("Invalid input tensor list");
    }

    for (auto &tensor : tensorList)
    {
        if (tensor.shape().rank() < 3 || tensor.shape().rank() > 4)
        {
            throw std::runtime_error("Invalid input tensor shape");
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

Tensor StackIntoInternal(Tensor &output, std::vector<Tensor> &tensorList, std::optional<Stream> pstream,
                         int32_t numberOfTensors)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    nvcvpy::TensorBatch inTensorBatch = nvcvpy::TensorBatch::Create(numberOfTensors);

    for (auto &tensor : tensorList)
    {
        inTensorBatch.pushBack(tensor);
    }

    auto op = CreateOperator<cvcuda::Stack>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {inTensorBatch});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});
    op->submit(pstream->cudaHandle(), inTensorBatch, output);
    return std::move(output);
}

Tensor StackInto(Tensor &output, std::vector<Tensor> &tensorList, std::optional<Stream> pstream)
{
    int64_t            outputShape[4] = {}; // NCHW/NHWC
    nvcv::TensorLayout layout         = nvcv::TENSOR_CHW;
    nvcv::DataType     dtype;

    checkTensorList(tensorList, outputShape, layout, dtype);

    if (output.shape().layout() != nvcv::TENSOR_NCHW && output.shape().layout() != nvcv::TENSOR_NHWC)
        throw std::runtime_error("Invalid output tensor shape");

    if (output.shape()[0] != outputShape[0])
        throw std::runtime_error("Invalid output tensor shape");

    StackIntoInternal(output, tensorList, pstream, outputShape[0]);
    return std::move(output);
}

Tensor Stack(std::vector<Tensor> &tensorList, std::optional<Stream> pstream)
{
    int64_t            outputShape[4] = {}; // NCHW/NHWC
    nvcv::TensorLayout layout         = nvcv::TENSOR_CHW;
    nvcv::DataType     dtype;
    checkTensorList(tensorList, outputShape, layout, dtype);

    //create new output tensor
    Tensor output = Tensor::Create(
        {
            {outputShape[0], outputShape[1], outputShape[2], outputShape[3]},
            layout
    },
        dtype);
    return StackIntoInternal(output, tensorList, pstream, outputShape[0]);
}

} // namespace

void ExportOpStack(py::module &m)
{
    using namespace pybind11::literals;

    m.def("stack", &Stack, "src"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Stack operation on the given cuda stream. This takes input tensors and combines them into a N(HWC/CHW) tensor.

        See also:
            Refer to the CV-CUDA C API reference for the Stack operator
            for more details and usage examples.

        Args:
            src (List[nvcv.Tensor]): Input tensors containing one or more samples each images all tensors must be N(HWC/CHW) or HWC/CHW and have the same data type and shape.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output tensor containing the stacked input tensors.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("stack_into", &StackInto, "dst"_a, "src"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the Stack operation on the given cuda stream. This takes input tensors and combines them into a N(HWC/CHW) tensor.

        See also:
            Refer to the CV-CUDA C API reference for the Stack operator
            for more details and usage examples.

        Args:
            dst (nvcv.Tensor): Output N(CHW/HWC) tensor to store the result of the operation.
            src (List[nvcv.Tensor]): Input tensors containing one or more samples each images all tensors must be N(HWC/CHW) or HWC/CHW and have the same data type and shape.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
