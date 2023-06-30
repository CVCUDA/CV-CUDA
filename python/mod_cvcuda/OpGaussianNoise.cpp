/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <common/String.hpp>
#include <cvcuda/OpGaussianNoise.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor GaussianNoiseInto(Tensor &output, Tensor &input, Tensor &mu, Tensor &sigma, bool per_channel,
                         unsigned long long seed, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    nvcv::TensorShape shape         = input.shape();
    auto              gaussiannoise = CreateOperator<cvcuda::GaussianNoise>((int)shape[0]);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, mu, sigma});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_WRITE, {*gaussiannoise});

    gaussiannoise->submit(pstream->cudaHandle(), input, output, mu, sigma, per_channel, seed);

    return output;
}

Tensor GaussianNoise(Tensor &input, Tensor &mu, Tensor &sigma, bool per_channel, unsigned long long seed,
                     std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return GaussianNoiseInto(output, input, mu, sigma, per_channel, seed, pstream);
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
    guard.add(LockMode::LOCK_READ, {input, mu, sigma});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_WRITE, {*gaussiannoise});

    gaussiannoise->submit(pstream->cudaHandle(), input, output, mu, sigma, per_channel, seed);

    return output;
}

ImageBatchVarShape GaussianNoiseVarShape(ImageBatchVarShape &input, Tensor &mu, Tensor &sigma, bool per_channel,
                                         unsigned long long seed, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.numImages());

    auto format = input.uniqueFormat();
    if (!format)
    {
        throw std::runtime_error("All images in input must have the same format.");
    }

    for (auto img = input.begin(); img != input.end(); ++img)
    {
        auto newimg = Image::Create(img->size(), format);
        output.pushBack(newimg);
    }

    return GaussianNoiseVarShapeInto(output, input, mu, sigma, per_channel, seed, pstream);
}

} // namespace

void ExportOpGaussianNoise(py::module &m)
{
    using namespace pybind11::literals;

    m.def("gaussiannoise", &GaussianNoise, "src"_a, "mu"_a, "sigma"_a, "per_channel"_a, "seed"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the GaussianNoise operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the GaussianNoise operator
            for more details and usage examples.

        Args:
            src (Tensor): Input image batch containing one or more images.
            mu (Tensor): An array of size batch that gives the mu value of each image.
            sigma (Tensor): An array of size batch that gives the sigma value of each image.
            per_channel (bool): Whether to add the same noise for all channels.
            seed (int): Seed for random numbers.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("gaussiannoise_into", &GaussianNoiseInto, "dst"_a, "src"_a, "mu"_a, "sigma"_a, "per_channel"_a, "seed"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the GaussianNoise operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the GaussianNoise operator
            for more details and usage examples.

        Args:
            dst (Tensor): Output image batch containing the result of the operation.
            src (Tensor): Input image batch containing one or more images.
            mu (Tensor): An array of size batch that gives the mu value of each image.
            sigma (Tensor): An array of size batch that gives the sigma value of each image.
            per_channel (bool): Whether to add the same noise for all channels.
            seed (int): Seed for random numbers.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("gaussiannoise", &GaussianNoiseVarShape, "src"_a, "mu"_a, "sigma"_a, "per_channel"_a, "seed"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the GaussianNoise operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the GaussianNoise operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input image batch containing one or more images.
            mu (Tensor): An array of size batch that gives the mu value of each image.
            sigma (Tensor): An array of size batch that gives the sigma value of each image.
            per_channel (bool): Whether to add the same noise for all channels.
            seed (int): Seed for random numbers.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("gaussiannoise_into", &GaussianNoiseVarShapeInto, "dst"_a, "src"_a, "mu"_a, "sigma"_a, "per_channel"_a,
          "seed"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the GaussianNoise operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the GaussianNoise operator
            for more details and usage examples.

        Args:
            dst (ImageBatchVarShape): Output image batch containing the result of the operation.
            src (ImageBatchVarShape): Input image batch containing one or more images.
            mu (Tensor): An array of size batch that gives the mu value of each image.
            sigma (Tensor): An array of size batch that gives the sigma value of each image.
            per_channel (bool): Whether to add the same noise for all channels.
            seed (int): Seed for random numbers.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            None

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
