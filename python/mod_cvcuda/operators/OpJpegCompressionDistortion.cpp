/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "UnaryElementwiseOp.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpJpegCompressionDistortion.hpp>

namespace cvcudapy {

namespace {
using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

// Scalar-quality wrappers share the generic unary-elementwise plumbing; the quality tensor
// variants are hand-rolled because the parameter tensor needs its own read lock.

Tensor &JpegCompressionDistortionScalarInto(Tensor &output, Tensor &input, int quality, std::optional<Stream *> pstream)
{
    return UnaryElementwiseInto<cvcuda::JpegCompressionDistortion>(output, input, pstream, quality);
}

std::shared_ptr<Tensor> JpegCompressionDistortionScalar(Tensor &input, int quality, std::optional<Stream *> pstream)
{
    return UnaryElementwiseTensor<cvcuda::JpegCompressionDistortion>(input, pstream, quality);
}

ImageBatchVarShape &JpegCompressionDistortionVarShapeScalarInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                                                int quality, std::optional<Stream *> pstream)
{
    return UnaryElementwiseInto<cvcuda::JpegCompressionDistortion>(output, input, pstream, quality);
}

std::shared_ptr<ImageBatchVarShape> JpegCompressionDistortionVarShapeScalar(ImageBatchVarShape &input, int quality,
                                                                            std::optional<Stream *> pstream)
{
    return UnaryElementwiseVarShape<cvcuda::JpegCompressionDistortion>(input, pstream, quality);
}

Tensor &JpegCompressionDistortionInto(Tensor &output, Tensor &input, Tensor &quality, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto op = CreateOperator<cvcuda::JpegCompressionDistortion>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input, quality});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    guard.run([&op, &stream, &input, &output, &quality]()
              { op->submit(stream.handle(), input.impl(), output.impl(), quality.impl()); });

    return output;
}

std::shared_ptr<Tensor> JpegCompressionDistortion(Tensor &input, Tensor &quality, std::optional<Stream *> pstream)
{
    auto output = Tensor::CreateFromReqs(nvcv::Tensor::CalcRequirements(input.impl().shape(), input.dtype()));

    JpegCompressionDistortionInto(*output, input, quality, pstream);
    return output;
}

ImageBatchVarShape &JpegCompressionDistortionVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                                          Tensor &quality, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto op = CreateOperator<cvcuda::JpegCompressionDistortion>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input, quality});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    guard.run([&op, &stream, &input, &output, &quality]()
              { op->submit(stream.handle(), input.impl(), output.impl(), quality.impl()); });

    return output;
}

std::shared_ptr<ImageBatchVarShape> JpegCompressionDistortionVarShape(ImageBatchVarShape &input, Tensor &quality,
                                                                      std::optional<Stream *> pstream)
{
    auto output = CreateSameShapeImageBatch(input);

    JpegCompressionDistortionVarShapeInto(*output, input, quality, pstream);
    return output;
}

} // namespace

void ExportOpJpegCompressionDistortion(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("jpeg_compression_distortion", NvtxTrace("cvcuda.jpeg_compression_distortion", &JpegCompressionDistortion),
          "src"_a, "quality"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the JpegCompressionDistortion operation on the given cuda stream.

        Simulates the artifacts of a JPEG compression/decompression round trip (full-range JFIF
        YCbCr conversion, 4:2:0 chroma subsampling, per-8x8-block DCT and Annex-K quantization with
        the libjpeg quality scaling). Entropy coding is not simulated, so results approximate — but
        do not bit-match — a real JPEG codec. 1-channel images take a luma-only path.

        See also:
            Refer to the CV-CUDA C API reference for the JpegCompressionDistortion operator
            for more details and usage examples.

        Args:
            src (nvcv.Tensor): Input tensor containing one or more uint8 images with (N)HWC or
                (N)CHW layout and 1 or 3 channels.
            quality (nvcv.Tensor): Per-image JPEG quality, from 1 (strongest distortion) to 100
                (weakest); rank-1 int32 tensor with one value per image. Values are clamped to
                [1, 100] on the device.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output tensor, with the same shape, layout and data type as the input.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("jpeg_compression_distortion",
          NvtxTrace("cvcuda.jpeg_compression_distortion", &JpegCompressionDistortionScalar), "src"_a, "quality"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the JpegCompressionDistortion operation on the given cuda stream.

        Overload applying one JPEG quality to the whole batch.

        Args:
            src (nvcv.Tensor): Input tensor containing one or more uint8 images with (N)HWC or
                (N)CHW layout and 1 or 3 channels.
            quality (int): JPEG quality applied to all images, from 1 (strongest distortion) to
                100 (weakest). Must be in [1, 100].
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output tensor, with the same shape, layout and data type as the input.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("jpeg_compression_distortion_into",
          NvtxTrace("cvcuda.jpeg_compression_distortion_into", &JpegCompressionDistortionInto), "dst"_a, "src"_a,
          "quality"_a, py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(

        Executes the JpegCompressionDistortion operation on the given cuda stream, writing into a
        caller-provided output tensor.

        Args:
            dst (nvcv.Tensor): Output tensor; must match the input's shape, layout and data type.
            src (nvcv.Tensor): Input tensor containing one or more uint8 images with (N)HWC or
                (N)CHW layout and 1 or 3 channels.
            quality (nvcv.Tensor): Per-image JPEG quality, from 1 (strongest distortion) to 100
                (weakest); rank-1 int32 tensor with one value per image. Values are clamped to
                [1, 100] on the device.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("jpeg_compression_distortion_into",
          NvtxTrace("cvcuda.jpeg_compression_distortion_into", &JpegCompressionDistortionScalarInto), "dst"_a, "src"_a,
          "quality"_a, py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(

        Executes the JpegCompressionDistortion operation on the given cuda stream, writing into a
        caller-provided output tensor.

        Overload applying one JPEG quality to the whole batch.

        Args:
            dst (nvcv.Tensor): Output tensor; must match the input's shape, layout and data type.
            src (nvcv.Tensor): Input tensor containing one or more uint8 images with (N)HWC or
                (N)CHW layout and 1 or 3 channels.
            quality (int): JPEG quality applied to all images, from 1 (strongest distortion) to
                100 (weakest). Must be in [1, 100].
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("jpeg_compression_distortion",
          NvtxTrace("cvcuda.jpeg_compression_distortion", &JpegCompressionDistortionVarShape), "src"_a, "quality"_a,
          py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the JpegCompressionDistortion operation on the given cuda stream.

        Variable-shape overload: all images in the batch must share one uint8 image format with 1
        or 3 (RGB) channels.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch of uint8 images with 1 or 3 (RGB)
                channels.
            quality (nvcv.Tensor): Per-image JPEG quality, from 1 (strongest distortion) to 100
                (weakest); rank-1 int32 tensor with one value per image. Values are clamped to
                [1, 100] on the device.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.ImageBatchVarShape: The output image batch, matching the input formats and sizes.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("jpeg_compression_distortion",
          NvtxTrace("cvcuda.jpeg_compression_distortion", &JpegCompressionDistortionVarShapeScalar), "src"_a,
          "quality"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the JpegCompressionDistortion operation on the given cuda stream.

        Variable-shape overload applying one JPEG quality to the whole batch.

        Args:
            src (nvcv.ImageBatchVarShape): Input image batch of uint8 images with 1 or 3 (RGB)
                channels.
            quality (int): JPEG quality applied to all images, from 1 (strongest distortion) to
                100 (weakest). Must be in [1, 100].
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.ImageBatchVarShape: The output image batch, matching the input formats and sizes.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("jpeg_compression_distortion_into",
          NvtxTrace("cvcuda.jpeg_compression_distortion_into", &JpegCompressionDistortionVarShapeInto), "dst"_a,
          "src"_a, "quality"_a, py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(

        Executes the JpegCompressionDistortion operation on the given cuda stream, writing into a
        caller-provided output image batch.

        Args:
            dst (nvcv.ImageBatchVarShape): Output image batch; must match the input formats and
                sizes.
            src (nvcv.ImageBatchVarShape): Input image batch of uint8 images with 1 or 3 (RGB)
                channels.
            quality (nvcv.Tensor): Per-image JPEG quality, from 1 (strongest distortion) to 100
                (weakest); rank-1 int32 tensor with one value per image. Values are clamped to
                [1, 100] on the device.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("jpeg_compression_distortion_into",
          NvtxTrace("cvcuda.jpeg_compression_distortion_into", &JpegCompressionDistortionVarShapeScalarInto), "dst"_a,
          "src"_a, "quality"_a, py::kw_only(), "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(

        Executes the JpegCompressionDistortion operation on the given cuda stream, writing into a
        caller-provided output image batch.

        Variable-shape overload applying one JPEG quality to the whole batch.

        Args:
            dst (nvcv.ImageBatchVarShape): Output image batch; must match the input formats and
                sizes.
            src (nvcv.ImageBatchVarShape): Input image batch of uint8 images with 1 or 3 (RGB)
                channels.
            quality (int): JPEG quality applied to all images, from 1 (strongest distortion) to
                100 (weakest). Must be in [1, 100].
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.ImageBatchVarShape: The output image batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
