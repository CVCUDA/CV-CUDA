/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ChannelCopyUtils.hpp"

#include <cvcuda/OpSplit.hpp>

namespace cvcudapy {
namespace {

using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

py::sequence SplitInto(py::sequence output, Tensor &input, std::optional<Stream *> pstream)
{
    auto tensors = py::cast<std::vector<std::shared_ptr<Tensor>>>(output);
    auto bands   = channelcopy::MakeBands(tensors);

    Stream &stream = native::GetStream(pstream);
    auto    op     = CreateOperator<cvcuda::Split>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    for (const auto &tensor : tensors) guard.add(LockMode::LOCK_MODE_WRITE, {*tensor});
    guard.add(LockMode::LOCK_MODE_NONE, {*bands, *op});
    guard.run([&op, &stream, &input, &bands]() { op->submit(stream.handle(), input.impl(), bands->impl()); });
    return output;
}

py::tuple Split(Tensor &input, std::optional<Stream *> pstream)
{
    const int  channelAxis = channelcopy::CheckImage(input);
    const auto channels    = static_cast<int>(input.impl().shape()[channelAxis]);
    const auto shape       = channelcopy::WithChannels(input.impl().shape(), 1);
    auto       stream      = native::GetStream(pstream).sharedStream();
    py::list   bands;
    for (int channel = 0; channel < channels; ++channel)
    {
        auto band = Tensor::CreateFromReqs(nvcv::Tensor::CalcRequirements(shape, input.dtype()), stream.get());
        bands.append(py::cast(band));
    }
    py::tuple output(bands);
    SplitInto(py::borrow<py::sequence>(output), input, stream.get());
    return output;
}

py::sequence SplitVarShapeInto(py::sequence output, const std::shared_ptr<ImageBatchVarShape> &input,
                               std::optional<Stream *> pstream)
{
    if (!input)
        throw py::type_error("Source image batch cannot be None");
    auto batches = py::cast<std::vector<std::shared_ptr<ImageBatchVarShape>>>(output);
    channelcopy::ValidateBands(batches);
#ifdef NB_FREE_THREADED
    auto                         mutexes = channelcopy::BatchMutexes(batches, input.get());
    nvcvpy::priv::BatchLockScope batchScope(mutexes.data(), mutexes.size());
#endif
    auto bands = channelcopy::MakeBands(batches);

    Stream &stream = native::GetStream(pstream);
    auto    op     = CreateOperator<cvcuda::Split>();

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {*input});
    for (const auto &batch : batches) guard.add(LockMode::LOCK_MODE_WRITE, {*batch});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});
    guard.run([&op, &stream, &input, &bands]()
              { op->submit(stream.handle(), input->impl(), bands.data(), static_cast<int32_t>(bands.size())); });
    return output;
}

py::tuple SplitVarShape(const std::shared_ptr<ImageBatchVarShape> &input, std::optional<Stream *> pstream)
{
    if (!input)
        throw py::type_error("Source image batch cannot be None");
#ifdef NB_FREE_THREADED
    auto                         mutexes = channelcopy::BatchMutexes({}, input.get());
    nvcvpy::priv::BatchLockScope batchScope(mutexes.data(), mutexes.size());
#endif
    const auto              format = channelcopy::CheckImage(*input);
    const auto              dtype  = format.planeDataType(0).channelType(0);
    const nvcv::ImageFormat bandFormat(nvcv::MemLayout::PITCH_LINEAR, dtype.dataKind(), nvcv::Swizzle::S_X000,
                                       dtype.packing());
    auto                    stream = native::GetStream(pstream).sharedStream();
    py::list                bands;
    for (int channel = 0; channel < format.numChannels(); ++channel)
        bands.append(py::cast(channelcopy::CreateImageBatch(*input, bandFormat, *stream)));
    py::tuple output(bands);
    SplitVarShapeInto(py::borrow<py::sequence>(output), input, stream.get());
    return output;
}

} // namespace

void ExportOpSplit(py::module_ &m)
{
    using namespace nanobind::literals;
    m.def("split", NvtxTrace("cvcuda.split", &Split), "src"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Copy each channel into an independent tensor, in channel order.

        Matches Pillow Image.split pixel-copy semantics without carrying image metadata.
        Supports HWC, NHWC, CHW and NCHW, 1-4 channels, and U8/S8/U16/S16/U32/S32/F16/F32/F64.
        Each band retains the source layout and an explicit singleton channel dimension.
        No color conversion, blending or alpha processing occurs. Even C=1 makes a copy.
        Work is asynchronous; the current stream is used when stream is omitted.

        Storage must be CUDA-accessible and pitch-linear, with base addresses and positive
        strides aligned to the scalar element size. Non-singleton strides must cover
        the inner dimensions' byte span in layout order; singleton strides need not.
        Row/sample padding is allowed and is not written.

        Args:
            src (cvcuda.Tensor): Positive-sized, pitch-linear CUDA image tensor.
            stream (cvcuda.Stream, optional): CUDA stream for submission.

        Returns:
            tuple[cvcuda.Tensor, ...]: Independently allocated single-channel tensors.
    )pbdoc");
    m.def("split_into", NvtxTrace("cvcuda.split_into", &SplitInto), "dst"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Copy channels into preallocated single-channel tensors; return dst unchanged.

        The band count must match src's channel count. Bands must have identical shapes,
        layouts and dtypes, matching src's rank, batch size, height, width and dtype.
        Their layout may differ from src's interleaved/planar layout. Source/destination
        and destination/destination byte spans must not overlap, including internal padding.
        Uses split's types, layouts, storage requirements and asynchronous semantics.

        Args:
            dst (Sequence[cvcuda.Tensor]): Ordered, preallocated single-channel destinations.
            src (cvcuda.Tensor): Image tensor whose channels are copied.
            stream (cvcuda.Stream, optional): CUDA stream for submission.

        Returns:
            Sequence[cvcuda.Tensor]: The original dst sequence.
    )pbdoc");
    m.def("split", NvtxTrace("cvcuda.split", &SplitVarShape), "src"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Copy an image batch's physical storage channels into independent band batches.

        The source must be nonempty and have a uniform format with 1-4 channels,
        stored in one interleaved plane or full-resolution single-channel planes.
        Supports U8/S8/U16/S16/U32/S32/F16/F32/F64 with contiguous, byte-aligned,
        equal-width scalar components, without intra-pixel padding.
        Sizes, pitches and allocations may differ between images and planes. Each output
        batch has the source's image count and corresponding sizes, with a single-channel
        non-color format of the same scalar dtype. Empty batches cannot determine C.

        Copies preserve every element's bits, even for C=1; outputs are not views.
        Order is interleaved component order or planar plane order, without interpreting
        color models, swizzles or alpha. CUDA-accessible pitch-linear storage is required;
        bases and row pitches must be scalar-aligned and positive pitches must cover a row.
        Subsampled, sub-byte, unequal-component and hybrid plane packings, extra-channel
        metadata and CUDA arrays are unsupported.
        Repeated read-only source images are allowed. Padding is not written.
        Work is asynchronous on the supplied/current stream. Batches, images, descriptors
        and backing storage must remain valid and must not be mutated by callers until
        completion; the binding retains submitted resources on the stream.

        Args:
            src (cvcuda.ImageBatchVarShape): Nonempty, uniform-format source image batch.
            stream (cvcuda.Stream, optional): CUDA stream for submission.

        Returns:
            tuple[cvcuda.ImageBatchVarShape, ...]: Independently allocated single-channel batches.
    )pbdoc");
    m.def("split_into", NvtxTrace("cvcuda.split_into", &SplitVarShapeInto), "dst"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(
        Copy an image batch's channels into preallocated band batches; return dst unchanged.

        Uses the image-batch split types, formats, storage requirements and asynchronous
        semantics. There must be one destination batch per source channel. Every batch
        must be nonempty with a uniform format and the same image count. Each band has
        one plane, one channel and the source's scalar dtype; semantic format tags may
        differ across bands. Corresponding image sizes must match; no resizing or
        broadcasting occurs. Every write/read and write/write
        overlap across the entire submission is rejected, including across image indices,
        bands, planes and padding inside their byte spans. Padding is not written.

        Args:
            dst (Sequence[cvcuda.ImageBatchVarShape]): Ordered, preallocated band batches.
            src (cvcuda.ImageBatchVarShape): Source image batch whose channels are copied.
            stream (cvcuda.Stream, optional): CUDA stream for submission.

        Returns:
            Sequence[cvcuda.ImageBatchVarShape]: The original dst sequence.
    )pbdoc");
}

} // namespace cvcudapy
