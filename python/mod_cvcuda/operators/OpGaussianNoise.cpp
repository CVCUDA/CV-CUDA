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
#include <cvcuda/OpGaussianNoise.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

#include <stdexcept>

namespace cvcudapy {

namespace {

class GaussianNoiseError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

Tensor GaussianNoiseInto(Tensor &output, Tensor &input, Tensor &mu, Tensor &sigma, bool per_channel,
                         unsigned long long seed, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    // HWC inputs (rank 3) have no N dim, so shape[0] is H — fall back to 1.
    int  batchSize     = (input.shape().size() == 4) ? (int)input.shape()[0] : 1;
    auto gaussiannoise = CreateOperator<cvcuda::GaussianNoise>(batchSize);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, mu, sigma});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*gaussiannoise});

    guard.run([&gaussiannoise, &pstream, &input, &output, &mu, &sigma, &per_channel, &seed]()
              { gaussiannoise->submit(pstream->cudaHandle(), input, output, mu, sigma, per_channel, seed); });

    return output;
}

Tensor GaussianNoise(Tensor &input, Tensor &mu, Tensor &sigma, bool per_channel, unsigned long long seed,
                     std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return GaussianNoiseInto(output, input, mu, sigma, per_channel, seed, pstream);
}

Tensor GaussianNoiseScalarInto(Tensor &output, Tensor &input, float mu, float sigma, bool per_channel,
                               std::optional<unsigned long long> seed, bool clip, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    int  batchSize     = input.shape().size() == 4 ? static_cast<int>(input.shape()[0]) : 1;
    auto gaussiannoise = CreateOperator<cvcuda::GaussianNoise>(batchSize);
    auto seedValue     = seed.value_or(0);
    bool reseed        = seed.has_value();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*gaussiannoise});

    guard.run(
        [gaussiannoise, pstream, &input, &output, mu, sigma, per_channel, seedValue, reseed, clip]() {
            gaussiannoise->submit(pstream->cudaHandle(), input, output, mu, sigma, per_channel, seedValue, reseed,
                                  clip);
        });

    return output;
}

Tensor GaussianNoiseScalar(Tensor &input, float mu, float sigma, bool per_channel,
                           std::optional<unsigned long long> seed, bool clip, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());
    return GaussianNoiseScalarInto(output, input, mu, sigma, per_channel, seed, clip, pstream);
}

ImageBatchVarShape GaussianNoiseVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &mu,
                                             Tensor &sigma, bool per_channel, unsigned long long seed,
                                             std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto gaussiannoise = CreateOperator<cvcuda::GaussianNoise>(input.numImages());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input, mu, sigma});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*gaussiannoise});

    guard.run([&gaussiannoise, &pstream, &input, &output, &mu, &sigma, &per_channel, &seed]()
              { gaussiannoise->submit(pstream->cudaHandle(), input, output, mu, sigma, per_channel, seed); });

    return output;
}

ImageBatchVarShape GaussianNoiseVarShape(ImageBatchVarShape &input, Tensor &mu, Tensor &sigma, bool per_channel,
                                         unsigned long long seed, std::optional<Stream> pstream)
{
    auto format = input.uniqueFormat();
    if (!format)
    {
        throw GaussianNoiseError("All images in input must have the same format.");
    }

    ImageBatchVarShape output = CreateSameShapeImageBatch(input, format, input.numImages());

    return GaussianNoiseVarShapeInto(output, input, mu, sigma, per_channel, seed, pstream);
}

} // namespace

void ExportOpGaussianNoise(py::module &m)
{
    using namespace pybind11::literals;

    m.def("gaussiannoise", NvtxTrace("cvcuda.gaussiannoise", &GaussianNoise), "src"_a, "mu"_a, "sigma"_a,
          "per_channel"_a, "seed"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the GaussianNoise operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input image batch containing one or more images.
            mu (cvcuda.Tensor): An array of size batch that gives the mu value of each image.
            sigma (cvcuda.Tensor): An array of size batch that gives the sigma value of each image.
            per_channel (bool): Whether to add the same noise for all channels.
            seed (int): Seed for random numbers.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output image batch.

    )pbdoc");

    m.def("gaussiannoise_into", NvtxTrace("cvcuda.gaussiannoise_into", &GaussianNoiseInto), "dst"_a, "src"_a, "mu"_a,
          "sigma"_a, "per_channel"_a, "seed"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the GaussianNoise operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output image batch containing the result of the operation.
            src (cvcuda.Tensor): Input image batch containing one or more images.
            mu (cvcuda.Tensor): An array of size batch that gives the mu value of each image.
            sigma (cvcuda.Tensor): An array of size batch that gives the sigma value of each image.
            per_channel (bool): Whether to add the same noise for all channels.
            seed (int): Seed for random numbers.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("gaussiannoise", NvtxTrace("cvcuda.gaussiannoise", &GaussianNoiseScalar), "src"_a, "mu"_a, "sigma"_a,
          "per_channel"_a, py::kw_only(), "seed"_a = std::nullopt, "clip"_a = true, "stream"_a = nullptr, R"pbdoc(
        Adds Gaussian noise using scalar mean and standard-deviation values.

        This overload passes mu and sigma by value, so it does not allocate or upload parameter tensors. When seed is
        None, a cached CV-CUDA random-number stream advances between calls. Supplying a seed forcibly reseeds the call,
        so repeating the same explicit seed reproduces the same result.

        Args:
            src (cvcuda.Tensor): Input image tensor.
            mu (float): Gaussian mean in input-value units.
            sigma (float): Non-negative Gaussian standard deviation in input-value units.
            per_channel (bool): Whether to generate independent noise for every channel.
            seed (int, optional): Non-negative 64-bit seed. None advances the cached random-number stream.
            clip (bool, optional): Clamp uint8 to [0, 255] and float32 to [0, 1]. With False, float32 remains unbounded
                and uint8 wraps modulo 256.
            stream (cvcuda.Stream, optional): CUDA stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output image tensor.
    )pbdoc");

    m.def("gaussiannoise_into", NvtxTrace("cvcuda.gaussiannoise_into", &GaussianNoiseScalarInto), "dst"_a, "src"_a,
          "mu"_a, "sigma"_a, "per_channel"_a, py::kw_only(), "seed"_a = std::nullopt, "clip"_a = true,
          "stream"_a = nullptr, R"pbdoc(
        Adds Gaussian noise using scalar mean and standard-deviation values into a supplied tensor.

        Args:
            dst (cvcuda.Tensor): Output tensor.
            src (cvcuda.Tensor): Input image tensor.
            mu (float): Gaussian mean in input-value units.
            sigma (float): Non-negative Gaussian standard deviation in input-value units.
            per_channel (bool): Whether to generate independent noise for every channel.
            seed (int, optional): Non-negative 64-bit seed. None advances the cached random-number stream.
            clip (bool, optional): Whether to clamp output to the image dtype's expected range.
            stream (cvcuda.Stream, optional): CUDA stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");

    m.def("gaussiannoise", NvtxTrace("cvcuda.gaussiannoise", &GaussianNoiseVarShape), "src"_a, "mu"_a, "sigma"_a,
          "per_channel"_a, "seed"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the GaussianNoise operation on the given cuda stream.


        Args:
            src (ImageBatchVarShape): Input image batch containing one or more images.
            mu (cvcuda.Tensor): An array of size batch that gives the mu value of each image.
            sigma (cvcuda.Tensor): An array of size batch that gives the sigma value of each image.
            per_channel (bool): Whether to add the same noise for all channels.
            seed (int): Seed for random numbers.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

    )pbdoc");

    m.def("gaussiannoise_into", NvtxTrace("cvcuda.gaussiannoise_into", &GaussianNoiseVarShapeInto), "dst"_a, "src"_a,
          "mu"_a, "sigma"_a, "per_channel"_a, "seed"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the GaussianNoise operation on the given cuda stream.


        Args:
            dst (ImageBatchVarShape): Output image batch containing the result of the operation.
            src (ImageBatchVarShape): Input image batch containing one or more images.
            mu (cvcuda.Tensor): An array of size batch that gives the mu value of each image.
            sigma (cvcuda.Tensor): An array of size batch that gives the sigma value of each image.
            per_channel (bool): Whether to add the same noise for all channels.
            seed (int): Seed for random numbers.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
