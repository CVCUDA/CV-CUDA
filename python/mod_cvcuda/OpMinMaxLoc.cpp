/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpMinMaxLoc.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

#include <string>

namespace cvcudapy {

namespace {

using TupleTensor3 = std::tuple<Tensor, Tensor, Tensor>;
using TupleTensor6 = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;

// Auxiliary function to get the value data type (for minVal or maxVal) for the given input data type
nvcv::DataType GetValDataType(nvcv::DataType inDataType)
{
    switch (inDataType)
    {
    case nvcv::TYPE_S8:
    case nvcv::TYPE_S16:
    case nvcv::TYPE_S32:
        return nvcv::TYPE_S32;

    case nvcv::TYPE_U8:
    case nvcv::TYPE_U16:
    case nvcv::TYPE_U32:
        return nvcv::TYPE_U32;

    case nvcv::TYPE_F32:
    case nvcv::TYPE_F64:
        return inDataType;

    default:
        throw std::runtime_error("Input data type not supported");
    }
    return nvcv::DataType();
}

// Get default number of maximum locations given width and height (1% of total pixels or 1)
inline int GetDefaultMaxLocs(int width, int height)
{
    return std::max(width * height / 100, 1);
}

template<class InputContainer>
TupleTensor3 MinLocInto(Tensor &minVal, Tensor &minLoc, Tensor &numMin, InputContainer &input,
                        std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::MinMaxLoc>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {minVal, minLoc, numMin});
    guard.add(LockMode::LOCK_NONE, {*op});

    op->submit(pstream->cudaHandle(), input, minVal, minLoc, numMin, nullptr, nullptr, nullptr);

    return TupleTensor3(std::move(minVal), std::move(minLoc), std::move(numMin));
}

TupleTensor3 MinLocTensorInto(Tensor &minVal, Tensor &minLoc, Tensor &numMin, Tensor &input,
                              std::optional<Stream> pstream)
{
    return MinLocInto(minVal, minLoc, numMin, input, pstream);
}

TupleTensor3 MinLocVarShapeInto(Tensor &minVal, Tensor &minLoc, Tensor &numMin, ImageBatchVarShape &input,
                                std::optional<Stream> pstream)
{
    return MinLocInto(minVal, minLoc, numMin, input, pstream);
}

template<class InputContainer>
TupleTensor3 MaxLocInto(Tensor &maxVal, Tensor &maxLoc, Tensor &numMax, InputContainer &input,
                        std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::MinMaxLoc>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {maxVal, maxLoc, numMax});
    guard.add(LockMode::LOCK_NONE, {*op});

    op->submit(pstream->cudaHandle(), input, nullptr, nullptr, nullptr, maxVal, maxLoc, numMax);

    return TupleTensor3(std::move(maxVal), std::move(maxLoc), std::move(numMax));
}

TupleTensor3 MaxLocTensorInto(Tensor &maxVal, Tensor &maxLoc, Tensor &numMax, Tensor &input,
                              std::optional<Stream> pstream)
{
    return MaxLocInto(maxVal, maxLoc, numMax, input, pstream);
}

TupleTensor3 MaxLocVarShapeInto(Tensor &maxVal, Tensor &maxLoc, Tensor &numMax, ImageBatchVarShape &input,
                                std::optional<Stream> pstream)
{
    return MaxLocInto(maxVal, maxLoc, numMax, input, pstream);
}

template<class InputContainer>
TupleTensor6 MinMaxLocInto(Tensor &minVal, Tensor &minLoc, Tensor &numMin, Tensor &maxVal, Tensor &maxLoc,
                           Tensor &numMax, InputContainer &input, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::MinMaxLoc>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {minVal, minLoc, numMin, maxVal, maxLoc, numMax});
    guard.add(LockMode::LOCK_NONE, {*op});

    op->submit(pstream->cudaHandle(), input, minVal, minLoc, numMin, maxVal, maxLoc, numMax);

    return TupleTensor6(std::move(minVal), std::move(minLoc), std::move(numMin), std::move(maxVal), std::move(maxLoc),
                        std::move(numMax));
}

TupleTensor6 MinMaxLocTensorInto(Tensor &minVal, Tensor &minLoc, Tensor &numMin, Tensor &maxVal, Tensor &maxLoc,
                                 Tensor &numMax, Tensor &input, std::optional<Stream> pstream)
{
    return MinMaxLocInto(minVal, minLoc, numMin, maxVal, maxLoc, numMax, input, pstream);
}

TupleTensor6 MinMaxLocVarShapeInto(Tensor &minVal, Tensor &minLoc, Tensor &numMin, Tensor &maxVal, Tensor &maxLoc,
                                   Tensor &numMax, ImageBatchVarShape &input, std::optional<Stream> pstream)
{
    return MinMaxLocInto(minVal, minLoc, numMin, maxVal, maxLoc, numMax, input, pstream);
}

template<class InputContainer>
TupleTensor3 MinLoc(InputContainer &input, nvcv::DataType inDataType, int numSamples, int maxLocs,
                    std::optional<Stream> pstream)
{
    // Row align must be 1 in below tensors so last 2 dimensions are packed

    // clang-format off

    Tensor minVal = Tensor::Create({{numSamples, 1}, "NC"}, GetValDataType(inDataType), 1);
    Tensor minLoc = Tensor::Create({{numSamples, maxLocs, 2}, "NMC"}, nvcv::TYPE_S32, 1);
    Tensor numMin = Tensor::Create({{numSamples, 1}, "NC"}, nvcv::TYPE_S32, 1);

    // clang-format on

    return MinLocInto(minVal, minLoc, numMin, input, pstream);
}

TupleTensor3 MinLocTensor(Tensor &input, int maxLocs, std::optional<Stream> pstream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(input.exportData());

    maxLocs = maxLocs == 0 ? GetDefaultMaxLocs(inAccess->numCols(), inAccess->numRows()) : maxLocs;

    return MinLoc(input, input.dtype(), inAccess->numSamples(), maxLocs, pstream);
}

TupleTensor3 MinLocVarShape(ImageBatchVarShape &input, int maxLocs, std::optional<Stream> pstream)
{
    maxLocs = maxLocs == 0 ? GetDefaultMaxLocs(input.maxSize().w, input.maxSize().h) : maxLocs;

    return MinLoc(input, input.uniqueFormat().planeDataType(0), input.numImages(), maxLocs, pstream);
}

template<class InputContainer>
TupleTensor3 MaxLoc(InputContainer &input, nvcv::DataType inDataType, int numSamples, int maxLocs,
                    std::optional<Stream> pstream)
{
    // Row align must be 1 in below tensors so last 2 dimensions are packed

    // clang-format off

    Tensor maxVal = Tensor::Create({{numSamples, 1}, "NC"}, GetValDataType(inDataType), 1);
    Tensor maxLoc = Tensor::Create({{numSamples, maxLocs, 2}, "NMC"}, nvcv::TYPE_S32, 1);
    Tensor numMax = Tensor::Create({{numSamples, 1}, "NC"}, nvcv::TYPE_S32, 1);

    // clang-format on

    return MaxLocInto(maxVal, maxLoc, numMax, input, pstream);
}

TupleTensor3 MaxLocTensor(Tensor &input, int maxLocs, std::optional<Stream> pstream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(input.exportData());

    maxLocs = maxLocs == 0 ? GetDefaultMaxLocs(inAccess->numCols(), inAccess->numRows()) : maxLocs;

    return MaxLoc(input, input.dtype(), inAccess->numSamples(), maxLocs, pstream);
}

TupleTensor3 MaxLocVarShape(ImageBatchVarShape &input, int maxLocs, std::optional<Stream> pstream)
{
    maxLocs = maxLocs == 0 ? GetDefaultMaxLocs(input.maxSize().w, input.maxSize().h) : maxLocs;

    return MaxLoc(input, input.uniqueFormat().planeDataType(0), input.numImages(), maxLocs, pstream);
}

template<class InputContainer>
TupleTensor6 MinMaxLoc(InputContainer &input, nvcv::DataType inDataType, int numSamples, int maxLocs,
                       std::optional<Stream> pstream)
{
    // Row align must be 1 in below tensors so last 2 dimensions are packed

    // clang-format off

    Tensor minVal = Tensor::Create({{numSamples, 1}, "NC"}, GetValDataType(inDataType), 1);
    Tensor minLoc = Tensor::Create({{numSamples, maxLocs, 2}, "NMC"}, nvcv::TYPE_S32, 1);
    Tensor numMin = Tensor::Create({{numSamples, 1}, "NC"}, nvcv::TYPE_S32, 1);
    Tensor maxVal = Tensor::Create({{numSamples, 1}, "NC"}, GetValDataType(inDataType), 1);
    Tensor maxLoc = Tensor::Create({{numSamples, maxLocs, 2}, "NMC"}, nvcv::TYPE_S32, 1);
    Tensor numMax = Tensor::Create({{numSamples, 1}, "NC"}, nvcv::TYPE_S32, 1);

    // clang-format on

    return MinMaxLocInto(minVal, minLoc, numMin, maxVal, maxLoc, numMax, input, pstream);
}

TupleTensor6 MinMaxLocTensor(Tensor &input, int maxLocs, std::optional<Stream> pstream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(input.exportData());

    maxLocs = maxLocs == 0 ? GetDefaultMaxLocs(inAccess->numCols(), inAccess->numRows()) : maxLocs;

    return MinMaxLoc(input, input.dtype(), inAccess->numSamples(), maxLocs, pstream);
}

TupleTensor6 MinMaxLocVarShape(ImageBatchVarShape &input, int maxLocs, std::optional<Stream> pstream)
{
    maxLocs = maxLocs == 0 ? GetDefaultMaxLocs(input.maxSize().w, input.maxSize().h) : maxLocs;

    return MinMaxLoc(input, input.uniqueFormat().planeDataType(0), input.numImages(), maxLocs, pstream);
}

// Function to get the docstring for an entry function

inline std::string GetDocString(const std::string &strInto, const std::string &strTensor, const std::string &strMinMax)
{
    std::string strSrc;
    if (strTensor.find("tensor") != std::string::npos)
    {
        strSrc = std::string(R"pbdoc(
            src (Tensor): Input tensor to get minimum/maximum values/locations.)pbdoc");
    }
    else if (strTensor.find("batch") != std::string::npos)
    {
        strSrc = std::string(R"pbdoc(
            src (ImageBatchVarShape): Input image batch to get minimum/maximum values/locations.)pbdoc");
    }

    std::string strArgs;
    if (strInto.find("into") != std::string::npos)
    {
        if (strMinMax.find("min") != std::string::npos)
        {
            strArgs += std::string(R"pbdoc(
            min_val (Tensor): Output tensor with minimum value.
            min_loc (Tensor): Output tensor with minimum locations.
            num_min (Tensor): Output tensor with number of minimum locations found.)pbdoc");
        }
        if (strMinMax.find("max") != std::string::npos)
        {
            strArgs += std::string(R"pbdoc(
            max_val (Tensor): Output tensor with maximum value.
            max_loc (Tensor): Output tensor with maximum locations.
            num_max (Tensor): Output tensor with number of maximum locations found.)pbdoc");
        }
        strArgs += strSrc;
    }
    else
    {
        strArgs += strSrc;
        strArgs += std::string(R"pbdoc(
            max_locations (Number, optional): Number of maximum locations to find, default is 1% of total
                                              pixels at a minimum of 1.)pbdoc");
    }

    std::string strReturns;
    if (strMinMax.find("minimum/maximum") != std::string::npos)
    {
        strReturns = std::string(R"pbdoc(
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: A tuple with minimum value, locations and number
            of minima, and also maximum value, locations and number of maxima.)pbdoc");
    }
    else if (strMinMax.find("min") != std::string::npos)
    {
        strReturns = std::string(R"pbdoc(
            Tuple[Tensor, Tensor, Tensor]: A tuple with minimum value, locations and number
            of minima.)pbdoc");
    }
    else if (strMinMax.find("max") != std::string::npos)
    {
        strReturns = std::string(R"pbdoc(
            Tuple[Tensor, Tensor, Tensor]: A tuple with maximum value, locations and number
            of maxima.)pbdoc");
    }

    return std::string(R"pbdoc(

        Finds )pbdoc")
         + strMinMax + std::string(R"pbdoc( on the input )pbdoc") + strTensor + std::string(R"pbdoc(.

        See also:
            Refer to the CV-CUDA C API reference for the MinMaxLoc operator
            for more details and usage examples.

        Args:)pbdoc")
         + strArgs + std::string(R"pbdoc(
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:)pbdoc")
         + strReturns + std::string(R"pbdoc(

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace

void ExportOpMinMaxLoc(py::module &m)
{
    using namespace pybind11::literals;

    m.def("min_loc", &MinLocTensor, "src"_a, "max_locations"_a = 0, py::kw_only(), "stream"_a = nullptr,
          GetDocString("", "tensor", "minimum").c_str());

    m.def("min_loc", &MinLocVarShape, "src"_a, "max_locations"_a = 0, py::kw_only(), "stream"_a = nullptr,
          GetDocString("", "batch", "minimum").c_str());

    m.def("max_loc", &MaxLocTensor, "src"_a, "max_locations"_a = 0, py::kw_only(), "stream"_a = nullptr,
          GetDocString("", "tensor", "maximum").c_str());

    m.def("max_loc", &MaxLocVarShape, "src"_a, "max_locations"_a = 0, py::kw_only(), "stream"_a = nullptr,
          GetDocString("", "batch", "maximum").c_str());

    m.def("min_max_loc", &MinMaxLocTensor, "src"_a, "max_locations"_a = 0, py::kw_only(), "stream"_a = nullptr,
          GetDocString("", "tensor", "minimum/maximum").c_str());

    m.def("min_max_loc", &MinMaxLocVarShape, "src"_a, "max_locations"_a = 0, py::kw_only(), "stream"_a = nullptr,
          GetDocString("", "batch", "minimum/maximum").c_str());

    m.def("min_loc_into", &MinLocTensorInto, "min_val"_a, "min_loc"_a, "num_min"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, GetDocString("into", "tensor", "minimum").c_str());

    m.def("min_loc_into", &MinLocVarShapeInto, "min_val"_a, "min_loc"_a, "num_min"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, GetDocString("into", "batch", "minimum").c_str());

    m.def("max_loc_into", &MaxLocTensorInto, "max_val"_a, "max_loc"_a, "num_max"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, GetDocString("into", "tensor", "maximum").c_str());

    m.def("max_loc_into", &MaxLocVarShapeInto, "max_val"_a, "max_loc"_a, "num_max"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, GetDocString("into", "batch", "maximum").c_str());

    m.def("min_max_loc_into", &MinMaxLocTensorInto, "min_val"_a, "min_loc"_a, "num_min"_a, "max_val"_a, "max_loc"_a,
          "num_max"_a, "src"_a, py::kw_only(), "stream"_a = nullptr,
          GetDocString("into", "tensor", "minimum/maximum").c_str());

    m.def("min_max_loc_into", &MinMaxLocVarShapeInto, "min_val"_a, "min_loc"_a, "num_min"_a, "max_val"_a, "max_loc"_a,
          "num_max"_a, "src"_a, py::kw_only(), "stream"_a = nullptr,
          GetDocString("into", "batch", "minimum/maximum").c_str());
}

} // namespace cvcudapy
