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

#include "Operators.hpp"
#include "VarShapeUtils.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpNormalize.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

#include <stdexcept>

namespace cvcudapy {

namespace {

enum OpFlags : uint32_t
{
    SCALE_IS_STDDEV = CVCUDA_NORMALIZE_SCALE_IS_STDDEV
};

} // namespace

namespace {
Tensor NormalizeInto(Tensor &output, Tensor &input, Tensor &base, Tensor &scale, std::optional<uint32_t> flags,
                     float globalScale, float globalShift, float epsilon, std::optional<Stream> pstream)
{
    if (!pstream.has_value())
    {
        pstream = Stream::Current();
    }

    if (!flags.has_value())
    {
        flags = 0;
    }

    auto normalize = CreateOperator<cvcuda::Normalize>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, base, scale});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*normalize});

    guard.run(
        [&normalize, &pstream, &input, &base, &scale, &output, &globalScale, &globalShift, &epsilon, &flags]() {
            normalize->submit(pstream->cudaHandle(), input, base, scale, output, globalScale, globalShift, epsilon,
                              *flags);
        });

    return std::move(output);
}

Tensor Normalize(Tensor &input, Tensor &base, Tensor &scale, std::optional<uint32_t> flags, float globalScale,
                 float globalShift, float epsilon, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return NormalizeInto(output, input, base, scale, flags, globalScale, globalShift, epsilon, pstream);
}

// Packs a Python list/tuple of 1..4 floats into a float4 (unused lanes zeroed) and reports how many
// values were supplied, so the tensor-free normalize overload can pass base/scale by value.
void ToFloat4AndCount(const std::vector<float> &values, const char *name, float4 &out, int32_t &count)
{
    const size_t n = values.size();
    if (n < 1 || n > 4)
    {
        throw std::invalid_argument(
            util::ConcatString(name, " must have 1 to 4 values (1 = broadcast, or the channel count), got ", n));
    }
    out.x = n > 0 ? values[0] : 0.f;
    out.y = n > 1 ? values[1] : 0.f;
    out.z = n > 2 ? values[2] : 0.f;
    out.w = n > 3 ? values[3] : 0.f;
    count = static_cast<int32_t>(n);
}

Tensor NormalizeScalarInto(Tensor &output, Tensor &input, const std::vector<float> &base,
                           const std::vector<float> &scale, std::optional<uint32_t> flags, float globalScale,
                           float globalShift, float epsilon, std::optional<Stream> pstream)
{
    if (!pstream.has_value())
    {
        pstream = Stream::Current();
    }

    if (!flags.has_value())
    {
        flags = 0;
    }

    float4  base4;
    float4  scale4;
    int32_t baseCount;
    int32_t scaleCount;
    ToFloat4AndCount(base, "base", base4, baseCount);
    ToFloat4AndCount(scale, "scale", scale4, scaleCount);

    auto normalize = CreateOperator<cvcuda::Normalize>();

    // base/scale are host values passed by value into the kernel launch, so only input/output are
    // device resources that need guarding.
    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*normalize});

    guard.run(
        [&normalize, &pstream, &input, base4, scale4, baseCount, scaleCount, &output, &globalScale, &globalShift,
         &epsilon, &flags]()
        {
            normalize->submit(pstream->cudaHandle(), input, base4, scale4, baseCount, scaleCount, output, globalScale,
                              globalShift, epsilon, *flags);
        });

    return std::move(output);
}

Tensor NormalizeScalar(Tensor &input, const std::vector<float> &base, const std::vector<float> &scale,
                       std::optional<uint32_t> flags, float globalScale, float globalShift, float epsilon,
                       std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return NormalizeScalarInto(output, input, base, scale, flags, globalScale, globalShift, epsilon, pstream);
}

ImageBatchVarShape VarShapeNormalizeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &base,
                                         Tensor &scale, std::optional<uint32_t> flags, float globalScale,
                                         float globalShift, float epsilon, std::optional<Stream> pstream)
{
    if (!pstream.has_value())
    {
        pstream = Stream::Current();
    }

    if (!flags.has_value())
    {
        flags = 0;
    }

    auto normalize = CreateOperator<cvcuda::Normalize>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, base, scale});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*normalize});

    guard.run(
        [&normalize, &pstream, &input, &base, &scale, &output, &globalScale, &globalShift, &epsilon, &flags]() {
            normalize->submit(pstream->cudaHandle(), input, base, scale, output, globalScale, globalShift, epsilon,
                              *flags);
        });

    return output;
}

ImageBatchVarShape VarShapeNormalize(ImageBatchVarShape &input, Tensor &base, Tensor &scale,
                                     std::optional<uint32_t> flags, float globalScale, float globalShift, float epsilon,
                                     std::optional<Stream> pstream)
{
    ImageBatchVarShape output = CreateSameShapeImageBatch(input);

    return VarShapeNormalizeInto(output, input, base, scale, flags, globalScale, globalShift, epsilon, pstream);
}

} // namespace

void ExportOpNormalize(py::module &m)
{
    using namespace pybind11::literals;

    py::enum_<OpFlags>(m, "NormalizeFlags").value("SCALE_IS_STDDEV", OpFlags::SCALE_IS_STDDEV);

    float defGlobalScale = 1;
    float defGlobalShift = 0;
    float defEpsilon     = 0;

    m.def("normalize", NvtxTrace("cvcuda.normalize", &Normalize), "src"_a, "base"_a, "scale"_a,
          "flags"_a = std::nullopt, py::kw_only(), "globalscale"_a = defGlobalScale, "globalshift"_a = defGlobalShift,
          "epsilon"_a = defEpsilon, "stream"_a = nullptr, R"pbdoc(
        Executes the Normalize operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            base (cvcuda.Tensor): Tensor providing base values for normalization.
            scale (cvcuda.Tensor): Tensor providing scale values for normalization.
            flags (int, optional): Algorithm flags, use cvcuda.NormalizeFlags.SCALE_IS_STDDEV if scale passed as argument
                is standard deviation instead or 0 if it is scaling.
            globalscale (float, optional): Additional scale value to be used in addition to scale.
            globalshift (float, optional): Additional bias value to be used in addition to base.
            epsilon (float, optional): Epsilon to use when cvcuda.NormalizeFlags.SCALE_IS_STDDEV flag is set as a regularizing
                term to be added to variance.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("normalize_into", NvtxTrace("cvcuda.normalize_into", &NormalizeInto), "dst"_a, "src"_a, "base"_a, "scale"_a,
          "flags"_a = std::nullopt, py::kw_only(), "globalscale"_a = defGlobalScale, "globalshift"_a = defGlobalShift,
          "epsilon"_a = defEpsilon, "stream"_a = nullptr, R"pbdoc(
        Executes the Normalize operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            base (cvcuda.Tensor): Tensor providing base values for normalization.
            scale (cvcuda.Tensor): Tensor providing scale values for normalization.
            flags (int, optional): Algorithm flags, use cvcuda.NormalizeFlags.SCALE_IS_STDDEV if scale passed as argument
                is standard deviation instead or 0 if it is scaling.
            globalscale (float, optional): Additional scale value to be used in addition to scale.
            globalshift (float, optional): Additional bias value to be used in addition to base.
            epsilon (float, optional): Epsilon to use when cvcuda.NormalizeFlags.SCALE_IS_STDDEV flag is set as a regularizing
                term to be added to variance.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("normalize", NvtxTrace("cvcuda.normalize", &VarShapeNormalize), "src"_a, "base"_a, "scale"_a,
          "flags"_a = std::nullopt, py::kw_only(), "globalscale"_a = defGlobalScale, "globalshift"_a = defGlobalShift,
          "epsilon"_a = defEpsilon, "stream"_a = nullptr, R"pbdoc(
        Executes the Normalize operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            base (cvcuda.Tensor): Tensor providing base values for normalization.
            scale (cvcuda.Tensor): Tensor providing scale values for normalization.
            flags (int, optional): Algorithm flags, use cvcuda.NormalizeFlags.SCALE_IS_STDDEV if scale passed as argument
                is standard deviation instead or 0 if it is scaling.
            globalscale (float, optional): Additional scale value to be used in addition to scale.
            globalshift (float, optional): Additional bias value to be used in addition to base.
            epsilon (float, optional): Epsilon to use when cvcuda.NormalizeFlags.SCALE_IS_STDDEV flag is set as a regularizing
                term to be added to variance.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("normalize_into", NvtxTrace("cvcuda.normalize_into", &VarShapeNormalizeInto), "dst"_a, "src"_a, "base"_a,
          "scale"_a, "flags"_a = std::nullopt, py::kw_only(), "globalscale"_a = defGlobalScale,
          "globalshift"_a = defGlobalShift, "epsilon"_a = defEpsilon, "stream"_a = nullptr, R"pbdoc(
        Executes the Normalize operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output image batch containing the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input image batch containing one or more images.
            base (cvcuda.Tensor): Tensor providing base values for normalization.
            scale (cvcuda.Tensor): Tensor providing scale values for normalization.
            flags (int, optional): Algorithm flags, use cvcuda.NormalizeFlags.SCALE_IS_STDDEV if scale passed as argument
                is standard deviation instead or 0 if it is scaling.
            globalscale (float, optional): Additional scale value to be used in addition to scale.
            globalshift (float, optional): Additional bias value to be used in addition to base.
            epsilon (float, optional): Epsilon to use when cvcuda.NormalizeFlags.SCALE_IS_STDDEV flag is set as a regularizing
                term to be added to variance.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");

    m.def("normalize", NvtxTrace("cvcuda.normalize", &NormalizeScalar), "src"_a, "base"_a, "scale"_a,
          "flags"_a = std::nullopt, py::kw_only(), "globalscale"_a = defGlobalScale, "globalshift"_a = defGlobalShift,
          "epsilon"_a = defEpsilon, "stream"_a = nullptr, R"pbdoc(
        Executes the Normalize operation on the given cuda stream.

        base and scale are given by value as Python lists/tuples of floats (not tensors), so no
        parameter tensor is allocated or uploaded; interleaved (NHWC/HWC) and planar (NCHW/CHW) input
        are both supported.

        Args:
            src (cvcuda.Tensor): Tensor of input images.
            base (List[float]): One broadcast base or one base per channel.
            scale (List[float]): One broadcast scale or one scale per channel.
            flags (int, optional): Set cvcuda.NormalizeFlags.SCALE_IS_STDDEV when scale represents standard deviation;
                otherwise use 0.
            globalscale (float, optional): Scale applied in addition to the per-channel scale.
            globalshift (float, optional): Bias applied in addition to the per-channel base.
            epsilon (float, optional): Variance regularizer used with cvcuda.NormalizeFlags.SCALE_IS_STDDEV.
            stream (cvcuda.Stream, optional): CUDA stream used to run the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("normalize_into", NvtxTrace("cvcuda.normalize_into", &NormalizeScalarInto), "dst"_a, "src"_a, "base"_a,
          "scale"_a, "flags"_a = std::nullopt, py::kw_only(), "globalscale"_a = defGlobalScale,
          "globalshift"_a = defGlobalShift, "epsilon"_a = defEpsilon, "stream"_a = nullptr, R"pbdoc(
        Executes the Normalize operation on the given cuda stream.

        base and scale are given by value as Python lists/tuples of floats (not tensors), so no
        parameter tensor is allocated or uploaded; interleaved (NHWC/HWC) and planar (NCHW/CHW) input
        are both supported.

        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            base (List[float]): Base values for normalization: length 1 (broadcast) or the channel count.
            scale (List[float]): Scale values for normalization: length 1 (broadcast) or the channel count.
            flags (int, optional): Algorithm flags, use cvcuda.NormalizeFlags.SCALE_IS_STDDEV if scale passed as argument
                is standard deviation instead or 0 if it is scaling.
            globalscale (float, optional): Additional scale value to be used in addition to scale.
            globalshift (float, optional): Additional bias value to be used in addition to base.
            epsilon (float, optional): Epsilon to use when cvcuda.NormalizeFlags.SCALE_IS_STDDEV flag is set as a regularizing
                term to be added to variance.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
