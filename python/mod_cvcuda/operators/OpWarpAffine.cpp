/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "NativeVarShapeUtils.hpp"
#include "Operators.hpp"

#include <common/Binding.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpWarpAffine.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>

#include <stdexcept>

namespace cvcudapy {

class WarpAffineVarShapeState final : public OperatorState<cvcuda::WarpAffine, void(int32_t)>
{
public:
    using Base = OperatorState<cvcuda::WarpAffine, void(int32_t)>;
    using Base::Base;

    class Key final : public nvcvpy::IKey
    {
    public:
        explicit Key(int32_t capacity)
            : m_capacity(capacity)
        {
        }

    private:
        size_t doGetHash() const override
        {
            return nvcvpy::util::ComputeHash(m_capacity);
        }

        bool doIsCompatible(const nvcvpy::IKey &that_) const override
        {
            return m_capacity == static_cast<const Key &>(that_).m_capacity;
        }

        int32_t m_capacity;
    };
};

// VarShape submissions write the operator's persistent coefficient buffer
// asynchronously, so an in-flight state cannot serve another stream.
template<>
struct OperatorCacheUsesCompletionLease<WarpAffineVarShapeState> : std::true_type
{
};

namespace {

using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

class WarpAffineError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

Tensor &WarpAffineInto(Tensor &output, Tensor &input, const pyarray &xform, const int32_t flags,
                       const NVCVBorderType borderMode, const pyarray &borderValue, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    float4 bValue = GetFloat4FromPyArray(borderValue);

    if (size_t xformDims = xform.ndim(); !(xformDims == 2 && xform.shape(0) == 2 && xform.shape(1) == 3))
    {
        throw WarpAffineError(
            util::ConcatString("Details of transformation matrix: nDim == 2, shape == (2, 3) but current is '",
                               xformDims, "', ('", xform.shape(0), "', '", xform.shape(1), "') respectively"));
    }

    NVCVAffineTransform xformOutput;
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            xformOutput[i * 3 + j] = xform.data()[i * 3 + j];
        }
    }

    auto warpAffine = CreateOperator<cvcuda::WarpAffine>(0);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*warpAffine});

    guard.run(
        [&warpAffine, &stream, &input, &output, &xformOutput, &flags, &borderMode, &bValue]()
        { warpAffine->submit(stream.handle(), input.impl(), output.impl(), xformOutput, flags, borderMode, bValue); });

    return output;
}

std::shared_ptr<Tensor> WarpAffine(Tensor &input, const pyarray &xform, const int32_t flags,
                                   const NVCVBorderType borderMode, const pyarray &borderValue,
                                   std::optional<Stream *> pstream)
{
    auto output = native::CreateTensor(input.impl().shape(), input.dtype());

    WarpAffineInto(*output, input, xform, flags, borderMode, borderValue, pstream);
    return output;
}

ImageBatchVarShape &WarpAffineVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &xform,
                                           const int32_t flags, const NVCVBorderType borderMode,
                                           const pyarray &borderValue, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    size_t bValueSize = borderValue.size();
    if (size_t bValueDims = borderValue.ndim(); bValueSize > 4 || bValueDims != 1)
    {
        const std::string message
            = util::ConcatString("Channels of borderValue should <= 4 and dimension should be 1, current is '",
                                 bValueSize, "', '", bValueDims, "' respectively");
        throw py::value_error(message.c_str());
    }
    float4 bValue;
    for (int i = 0; i < 4; i++)
    {
        const auto valueIdx               = static_cast<size_t>(i);
        nvcv::cuda::GetElement(bValue, i) = bValueSize > valueIdx ? borderValue.data()[valueIdx] : 0.f;
    }

    auto warpAffine = CreateOperatorEx<WarpAffineVarShapeState>(input.capacity());

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input, xform});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*warpAffine});

    guard.run(
        [&warpAffine, &stream, &input, &output, &xform, &flags, &borderMode, &bValue]()
        { warpAffine->submit(stream.handle(), input.impl(), output.impl(), xform.impl(), flags, borderMode, bValue); });

    return output;
}

std::shared_ptr<ImageBatchVarShape> WarpAffineVarShape(ImageBatchVarShape &input, Tensor &xform, const int32_t flags,
                                                       const NVCVBorderType borderMode, const pyarray &borderValue,
                                                       std::optional<Stream *> pstream)
{
    auto output = CreateSameShapeImageBatch(input);

    WarpAffineVarShapeInto(*output, input, xform, flags, borderMode, borderValue, pstream);
    return output;
}

} // namespace

void ExportOpWarpAffine(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("warp_affine", NvtxTrace("cvcuda.warp_affine", &WarpAffine), "src"_a, "xform"_a.none(), "flags"_a,
          py::kw_only(), "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "border_value"_a.none() = 0,
          "stream"_a = nullptr, R"pbdoc(
        Executes the Warp Affine operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            xform (cvcuda.Tensor): 2x3 float affine transformation matrix.
            flags (int): Combination of interpolation methods(cvcuda.Interp.NEAREST, cvcuda.Interp.LINEAR or cvcuda.Interp.CUBIC)
                and the optional flag cvcuda.Interp.WARP_INVERSE_MAP, that sets xform as the inverse transformation.
            border_mode (cvcuda.Border, optional): Pixel extrapolation method (cvcuda.Border.CONSTANT,
                cvcuda.Border.REPLICATE, or cvcuda.Border.CONSTANT_COVERAGE).
            border_value (numpy.ndarray, optional):Used to specify values for a constant border, must be a size <= 4 and dim of 1,
                where the values specify the border color for each color channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("warp_affine_into", NvtxTrace("cvcuda.warp_affine_into", &WarpAffineInto), "dst"_a, "src"_a, "xform"_a.none(),
          "flags"_a, py::kw_only(), "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "border_value"_a.none() = 0,
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Executes the Warp Affine operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            xform (cvcuda.Tensor): 2x3 float affine transformation matrix.
            flags (int): Combination of interpolation methods(cvcuda.Interp.NEAREST, cvcuda.Interp.LINEAR or cvcuda.Interp.CUBIC)
                          and the optional flag cvcuda.Interp.WARP_INVERSE_MAP, that sets xform as the inverse transformation.
            border_mode (cvcuda.Border, optional): Pixel extrapolation method (cvcuda.Border.CONSTANT,
                cvcuda.Border.REPLICATE, or cvcuda.Border.CONSTANT_COVERAGE).
            border_value (numpy.ndarray, optional):Used to specify values for a constant border, must be a size <= 4 and dim of 1,
                                    where the values specify the border color for each color channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("warp_affine", NvtxTrace("cvcuda.warp_affine", &WarpAffineVarShape), "src"_a, "xform"_a, "flags"_a,
          py::kw_only(), "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "border_value"_a.none() = 0,
          "stream"_a = nullptr, R"pbdoc(
        Executes the Warp Affine operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            xform (cvcuda.Tensor): 2x3 float affine transformation matrix(s) for each image.
            flags (int): Combination of interpolation methods(cvcuda.Interp.NEAREST, cvcuda.Interp.LINEAR or cvcuda.Interp.CUBIC)
                and the optional flag cvcuda.Interp.WARP_INVERSE_MAP, that sets xform as the inverse transformation.
            border_mode (cvcuda.Border, optional): Pixel extrapolation method (cvcuda.Border.CONSTANT or
                cvcuda.Border.REPLICATE). cvcuda.Border.CONSTANT_COVERAGE is not supported for variable-shape batches.
            border_value (numpy.ndarray, optional):Used to specify values for a constant border, must be a size <= 4 and dim of 1,
                where the values specify the border color for each color channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("warp_affine_into", NvtxTrace("cvcuda.warp_affine_into", &WarpAffineVarShapeInto), "dst"_a, "src"_a,
          "xform"_a, "flags"_a, py::kw_only(), "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
          "border_value"_a.none() = 0, "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Executes the Warp Affine operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            xform (cvcuda.Tensor): 2x3 float affine transformation matrix(s) for each image in batch.
            flags (int): Combination of interpolation methods(cvcuda.Interp.NEAREST, cvcuda.Interp.LINEAR or cvcuda.Interp.CUBIC)
                and the optional flag cvcuda.Interp.WARP_INVERSE_MAP, that sets xform as the inverse transformation.
            border_mode (cvcuda.Border, optional): Pixel extrapolation method (cvcuda.Border.CONSTANT or
                cvcuda.Border.REPLICATE). cvcuda.Border.CONSTANT_COVERAGE is not supported for variable-shape batches.
            border_value (numpy.ndarray, optional):Used to specify values for a constant border, must be a size <= 4 and dim of 1,
                where the values specify the border color for each color channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
