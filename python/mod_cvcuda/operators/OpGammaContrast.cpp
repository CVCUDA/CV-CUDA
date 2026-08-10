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
#include <cvcuda/OpGammaContrast.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/RoundMode.h>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
ImageBatchVarShape VarShapeGammaContrastInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &gamma,
                                             std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto gamma_contrast = CreateOperator<cvcuda::GammaContrast>(input.capacity(), input.uniqueFormat().numChannels());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, gamma});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*gamma_contrast});

    guard.run([&gamma, &gamma_contrast, &pstream, &input, &output]()
              { gamma_contrast->submit(pstream->cudaHandle(), input, output, gamma); });

    return output;
}

ImageBatchVarShape VarShapeGammaContrast(ImageBatchVarShape &input, Tensor &gamma, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = CreateSameShapeImageBatch(input);

    return VarShapeGammaContrastInto(output, input, gamma, pstream);
}

Tensor TensorGammaContrastInto(Tensor &output, Tensor &input, Tensor &gamma, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    // Number of samples bounds the gamma scratch; 4 is the max supported channel count (the operator
    // validates the actual channel count against the input layout).
    const int numSamples = (input.shape().size() == 4) ? static_cast<int>(input.shape()[0]) : 1;

    auto gamma_contrast = CreateOperator<cvcuda::GammaContrast>(numSamples, 4);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, gamma});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*gamma_contrast});

    guard.run([&gamma, &gamma_contrast, &pstream, &input, &output]()
              { gamma_contrast->submit(pstream->cudaHandle(), input, output, gamma); });

    return output;
}

Tensor TensorGammaContrast(Tensor &input, Tensor &gamma, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return TensorGammaContrastInto(output, input, gamma, pstream);
}

Tensor TensorGammaContrastScalarInto(Tensor &output, Tensor &input, float gamma, float gain, NVCVRoundMode round,
                                     std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    // Scalar gamma/gain are baked into the kernel launch, so the create-time max-batch/max-channel
    // capacities (which only bound the gamma-tensor staging of the other overloads) don't constrain
    // this path. Use the minimum the C API accepts so one cached operator serves every scalar call
    // regardless of batch size.
    auto gamma_contrast = CreateOperator<cvcuda::GammaContrast>(1, 1);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*gamma_contrast});

    guard.run([&gamma, &gain, &round, &gamma_contrast, &pstream, &input, &output]()
              { gamma_contrast->submit(pstream->cudaHandle(), input, output, gamma, gain, round); });

    return output;
}

Tensor TensorGammaContrastScalar(Tensor &input, float gamma, float gain, NVCVRoundMode round,
                                 std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return TensorGammaContrastScalarInto(output, input, gamma, gain, round, pstream);
}

} // namespace

void ExportOpGammaContrast(py::module &m)
{
    using namespace pybind11::literals;

    m.def("gamma_contrast", NvtxTrace("cvcuda.gamma_contrast", &TensorGammaContrast), "src"_a, "gamma"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Executes the Gamma Contrast operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor (interleaved (N)HWC or planar (N)CHW layout).
            gamma (cvcuda.Tensor): 1D Tensor with the gamma value for each sample / sample channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("gamma_contrast_into", NvtxTrace("cvcuda.gamma_contrast_into", &TensorGammaContrastInto), "dst"_a, "src"_a,
          "gamma"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Gamma Contrast operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor (interleaved (N)HWC or planar (N)CHW layout).
            gamma (cvcuda.Tensor): 1D Tensor with the gamma value for each sample / sample channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("gamma_contrast", NvtxTrace("cvcuda.gamma_contrast", &TensorGammaContrastScalar), "src"_a, "gamma"_a,
          "gain"_a = 1.0, py::kw_only(), "round"_a = NVCV_ROUND_NEAREST, "stream"_a = nullptr, R"pbdoc(
        Executes the Gamma Contrast operation with host-scalar gamma and gain on the given cuda stream.

        Applies ``out = gain * in**gamma`` (the torchvision ``adjust_gamma`` formula) with a single
        ``gamma``/``gain`` for every sample and channel. The scalars are passed by value into the kernel
        launch -- no gamma tensor is allocated and no host-to-device copy is performed. With ``gain == 1.0``
        and ``round == cvcuda.Round.NEAREST``, the result is bit-exact with the device-tensor gamma
        overload fed a gamma tensor filled with the same value. ``gamma`` is not range-validated: a
        negative value follows powf semantics (NaN for fractional powers of negative inputs).

        Args:
            src (cvcuda.Tensor): Input tensor (interleaved (N)HWC or planar (N)CHW layout).
            gamma (float): Gamma exponent applied to every sample / channel.
            gain (float, optional): Output gain applied to every sample / channel. Defaults to 1.0.
            round (cvcuda.Round, optional): Rounding mode used for integer outputs. Defaults to
                cvcuda.Round.NEAREST; use cvcuda.Round.TRUNCATE to truncate toward zero. Has no
                effect for floating-point outputs.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("gamma_contrast_into", NvtxTrace("cvcuda.gamma_contrast_into", &TensorGammaContrastScalarInto), "dst"_a,
          "src"_a, "gamma"_a, "gain"_a = 1.0, py::kw_only(), "round"_a = NVCV_ROUND_NEAREST, "stream"_a = nullptr,
          R"pbdoc(
        Executes the Gamma Contrast operation with host-scalar gamma and gain on the given cuda stream.

        Applies ``out = gain * in**gamma`` (the torchvision ``adjust_gamma`` formula) with a single
        ``gamma``/``gain`` for every sample and channel. The scalars are passed by value into the kernel
        launch -- no gamma tensor is allocated and no host-to-device copy is performed. ``gamma`` is not
        range-validated: a negative value follows powf semantics (NaN for fractional powers of negative
        inputs).

        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor (interleaved (N)HWC or planar (N)CHW layout).
            gamma (float): Gamma exponent applied to every sample / channel.
            gain (float, optional): Output gain applied to every sample / channel. Defaults to 1.0.
            round (cvcuda.Round, optional): Rounding mode used for integer outputs. Defaults to
                cvcuda.Round.NEAREST; use cvcuda.Round.TRUNCATE to truncate toward zero. Has no
                effect for floating-point outputs.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("gamma_contrast", NvtxTrace("cvcuda.gamma_contrast", &VarShapeGammaContrast), "src"_a, "gamma"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Gamma Contrast operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input tensor containing one or more images.
            gamma (cvcuda.Tensor): 1D Tensor with the the gamma value for each image / image channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("gamma_contrast_into", NvtxTrace("cvcuda.gamma_contrast_into", &VarShapeGammaContrastInto), "dst"_a, "src"_a,
          "gamma"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the Gamma Contrast operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output tensor to store the result of the operation.
            src (cvcuda.ImageBatchVarShape): Input tensor containing one or more images.
            gamma (cvcuda.Tensor): 1D Tensor with the the gamma value for each image / image channel.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
