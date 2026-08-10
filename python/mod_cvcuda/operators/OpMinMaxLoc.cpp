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
#include <cvcuda/OpMinMaxLoc.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cvcudapy {

namespace {

class MinMaxLocError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

using TupleTensor3 = std::tuple<Tensor, Tensor, Tensor>;
using TupleTensor6 = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;

// Auxiliary function to get the value data type (for minVal or maxVal) for the given input data type
nvcv::DataType GetValDataType(nvcv::DataType inDataType)
{
    switch (static_cast<NVCVDataType>(inDataType))
    {
    case NVCV_DATA_TYPE_S8:
    case NVCV_DATA_TYPE_S16:
    case NVCV_DATA_TYPE_S32:
        return nvcv::TYPE_S32;

    case NVCV_DATA_TYPE_U8:
    case NVCV_DATA_TYPE_U16:
    case NVCV_DATA_TYPE_U32:
        return nvcv::TYPE_U32;

    case NVCV_DATA_TYPE_F32:
    case NVCV_DATA_TYPE_F64:
        return inDataType;

    default:
        throw MinMaxLocError("Input data type not supported");
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
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {minVal, minLoc, numMin});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    guard.run(
        [&op, &pstream, &input, &minVal, &minLoc, &numMin]()
        {
            op->submit(pstream->cudaHandle(), input, minVal, minLoc, numMin, nvcv::Tensor{nullptr},
                       nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr});
        });

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
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {maxVal, maxLoc, numMax});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    guard.run(
        [&op, &pstream, &input, &maxVal, &maxLoc, &numMax]()
        {
            op->submit(pstream->cudaHandle(), input, nvcv::Tensor{nullptr}, nvcv::Tensor{nullptr},
                       nvcv::Tensor{nullptr}, maxVal, maxLoc, numMax);
        });

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
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {minVal, minLoc, numMin, maxVal, maxLoc, numMax});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    guard.run([&op, &pstream, &input, &minVal, &minLoc, &numMin, &maxVal, &maxLoc, &numMax]()
              { op->submit(pstream->cudaHandle(), input, minVal, minLoc, numMin, maxVal, maxLoc, numMax); });

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

    return MinLoc(input, input.dtype(), static_cast<int>(inAccess->numSamples()), maxLocs, pstream);
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

    return MaxLoc(input, input.dtype(), static_cast<int>(inAccess->numSamples()), maxLocs, pstream);
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

    return MinMaxLoc(input, input.dtype(), static_cast<int>(inAccess->numSamples()), maxLocs, pstream);
}

TupleTensor6 MinMaxLocVarShape(ImageBatchVarShape &input, int maxLocs, std::optional<Stream> pstream)
{
    maxLocs = maxLocs == 0 ? GetDefaultMaxLocs(input.maxSize().w, input.maxSize().h) : maxLocs;

    return MinMaxLoc(input, input.uniqueFormat().planeDataType(0), input.numImages(), maxLocs, pstream);
}

// Function to get the docstring for an entry function.
//
// Assembles a Google-style docstring with 4-space continuation indent so that
// Napoleon + docutils render cleanly (no block-quote or unexpected-indent
// warnings).  Parameters follow the original convention:
//   strInto   — "" for allocating variants, "into" for *_into variants
//   strTensor — "tensor" or "batch"
//   strMinMax — "minimum", "maximum", or "minimum/maximum"

inline std::string GetDocString(std::string_view strInto, std::string_view strTensor, std::string_view strMinMax)
{
    const bool isInto  = strInto.find("into") != std::string_view::npos;
    const bool isBatch = strTensor.find("batch") != std::string_view::npos;
    const bool hasMin  = strMinMax.find("min") != std::string_view::npos;
    const bool hasMax  = strMinMax.find("max") != std::string_view::npos;

    const std::string srcType  = isBatch ? "cvcuda.ImageBatchVarShape" : "cvcuda.Tensor";
    const std::string srcDesc  = isBatch ? "Input image batch to get minimum/maximum values/locations."
                                         : "Input tensor to get minimum/maximum values/locations.";
    const std::string kindDesc = isBatch ? "image batch" : "tensor";

    std::ostringstream out;
    out << "\n"
        << "        Finds " << strMinMax << " values and locations on the input " << kindDesc << ".\n"
        << "\n"
        << "\n"
        << "        Args:\n";

    if (isInto)
    {
        if (hasMin)
        {
            out << "            min_val (cvcuda.Tensor): Output tensor with minimum value.\n"
                << "            min_loc (cvcuda.Tensor): Output tensor with minimum locations.\n"
                << "            num_min (cvcuda.Tensor): Output tensor with number of minimum locations found.\n";
        }
        if (hasMax)
        {
            out << "            max_val (cvcuda.Tensor): Output tensor with maximum value.\n"
                << "            max_loc (cvcuda.Tensor): Output tensor with maximum locations.\n"
                << "            num_max (cvcuda.Tensor): Output tensor with number of maximum locations found.\n";
        }
        out << "            src (" << srcType << "): " << srcDesc << "\n";
    }
    else
    {
        out << "            src (" << srcType << "): " << srcDesc << "\n"
            << "            max_locations (Number, optional): Number of maximum locations to find,\n"
            << "                default is 1% of total pixels at a minimum of 1.\n";
    }
    out << "            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.\n"
        << "\n"
        << "        Returns:\n";

    if (strMinMax == "minimum/maximum")
    {
        out << "            Tuple[cvcuda.Tensor, cvcuda.Tensor, cvcuda.Tensor,"
            << " cvcuda.Tensor, cvcuda.Tensor, cvcuda.Tensor]: A tuple with minimum\n"
            << "                value, locations and number of minima, and also maximum value,\n"
            << "                locations and number of maxima.\n";
    }
    else if (hasMin)
    {
        out << "            Tuple[cvcuda.Tensor, cvcuda.Tensor, cvcuda.Tensor]: A tuple with\n"
            << "                minimum value, locations and number of minima.\n";
    }
    else if (hasMax)
    {
        out << "            Tuple[cvcuda.Tensor, cvcuda.Tensor, cvcuda.Tensor]: A tuple with\n"
            << "                maximum value, locations and number of maxima.\n";
    }
    out << "    ";
    return out.str();
}

} // namespace

void ExportOpMinMaxLoc(py::module &m)
{
    using namespace pybind11::literals;

    m.def("min_loc", NvtxTrace("cvcuda.min_loc", &MinLocTensor), "src"_a, "max_locations"_a = 0, py::kw_only(),
          "stream"_a = nullptr, GetDocString("", "tensor", "minimum").c_str());

    m.def("min_loc", NvtxTrace("cvcuda.min_loc", &MinLocVarShape), "src"_a, "max_locations"_a = 0, py::kw_only(),
          "stream"_a = nullptr, GetDocString("", "batch", "minimum").c_str());

    m.def("max_loc", NvtxTrace("cvcuda.max_loc", &MaxLocTensor), "src"_a, "max_locations"_a = 0, py::kw_only(),
          "stream"_a = nullptr, GetDocString("", "tensor", "maximum").c_str());

    m.def("max_loc", NvtxTrace("cvcuda.max_loc", &MaxLocVarShape), "src"_a, "max_locations"_a = 0, py::kw_only(),
          "stream"_a = nullptr, GetDocString("", "batch", "maximum").c_str());

    m.def("min_max_loc", NvtxTrace("cvcuda.min_max_loc", &MinMaxLocTensor), "src"_a, "max_locations"_a = 0,
          py::kw_only(), "stream"_a = nullptr, GetDocString("", "tensor", "minimum/maximum").c_str());

    m.def("min_max_loc", NvtxTrace("cvcuda.min_max_loc", &MinMaxLocVarShape), "src"_a, "max_locations"_a = 0,
          py::kw_only(), "stream"_a = nullptr, GetDocString("", "batch", "minimum/maximum").c_str());

    m.def("min_loc_into", NvtxTrace("cvcuda.min_loc_into", &MinLocTensorInto), "min_val"_a, "min_loc"_a, "num_min"_a,
          "src"_a, py::kw_only(), "stream"_a = nullptr, GetDocString("into", "tensor", "minimum").c_str());

    m.def("min_loc_into", NvtxTrace("cvcuda.min_loc_into", &MinLocVarShapeInto), "min_val"_a, "min_loc"_a, "num_min"_a,
          "src"_a, py::kw_only(), "stream"_a = nullptr, GetDocString("into", "batch", "minimum").c_str());

    m.def("max_loc_into", NvtxTrace("cvcuda.max_loc_into", &MaxLocTensorInto), "max_val"_a, "max_loc"_a, "num_max"_a,
          "src"_a, py::kw_only(), "stream"_a = nullptr, GetDocString("into", "tensor", "maximum").c_str());

    m.def("max_loc_into", NvtxTrace("cvcuda.max_loc_into", &MaxLocVarShapeInto), "max_val"_a, "max_loc"_a, "num_max"_a,
          "src"_a, py::kw_only(), "stream"_a = nullptr, GetDocString("into", "batch", "maximum").c_str());

    m.def("min_max_loc_into", NvtxTrace("cvcuda.min_max_loc_into", &MinMaxLocTensorInto), "min_val"_a, "min_loc"_a,
          "num_min"_a, "max_val"_a, "max_loc"_a, "num_max"_a, "src"_a, py::kw_only(), "stream"_a = nullptr,
          GetDocString("into", "tensor", "minimum/maximum").c_str());

    m.def("min_max_loc_into", NvtxTrace("cvcuda.min_max_loc_into", &MinMaxLocVarShapeInto), "min_val"_a, "min_loc"_a,
          "num_min"_a, "max_val"_a, "max_loc"_a, "num_max"_a, "src"_a, py::kw_only(), "stream"_a = nullptr,
          GetDocString("into", "batch", "minimum/maximum").c_str());
}

} // namespace cvcudapy
