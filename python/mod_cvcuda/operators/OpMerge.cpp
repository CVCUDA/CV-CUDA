/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ChannelCopyUtils.hpp"

#include <cvcuda/OpMerge.hpp>

namespace cvcudapy {
namespace {

using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

Tensor &MergeInto(Tensor &output, const std::vector<std::shared_ptr<Tensor>> &input, std::optional<Stream *> pstream)
{
    auto bands = channelcopy::MakeBands(input);

    Stream &stream = native::GetStream(pstream);
    auto    op     = CreateOperator<cvcuda::Merge>();

    ResourceGuard guard(stream);
    for (const auto &tensor : input) guard.add(LockMode::LOCK_MODE_READ, {*tensor});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*bands, *op});
    guard.run([&op, &stream, &bands, &output]() { op->submit(stream.handle(), bands->impl(), output.impl()); });
    return output;
}

std::shared_ptr<Tensor> Merge(const std::vector<std::shared_ptr<Tensor>> &input,
                              std::optional<nvcv::TensorLayout> layout, std::optional<Stream *> pstream)
{
    channelcopy::ValidateBands(input);
    const auto outputLayout = layout.value_or(input.front()->impl().layout());
    channelcopy::ChannelAxis(outputLayout);
    if (outputLayout.rank() != input.front()->rank())
        throw py::value_error("Merge must preserve batched/unbatched rank");
    const auto shape  = channelcopy::WithChannels(Permute(input.front()->impl().shape(), outputLayout), input.size());
    auto       stream = native::GetStream(pstream).sharedStream();
    auto output = Tensor::CreateFromReqs(nvcv::Tensor::CalcRequirements(shape, input.front()->dtype()), stream.get());
    MergeInto(*output, input, stream.get());
    return output;
}

ImageBatchVarShape &MergeVarShapeInto(const std::shared_ptr<ImageBatchVarShape>              &output,
                                      const std::vector<std::shared_ptr<ImageBatchVarShape>> &input,
                                      std::optional<Stream *>                                 pstream)
{
    if (!output)
        throw py::type_error("Destination image batch cannot be None");
    channelcopy::ValidateBands(input);
#ifdef NB_FREE_THREADED
    auto                         mutexes = channelcopy::BatchMutexes(input, output.get());
    nvcvpy::priv::BatchLockScope batchScope(mutexes.data(), mutexes.size());
#endif
    auto bands = channelcopy::MakeBands(input);

    Stream &stream = native::GetStream(pstream);
    auto    op     = CreateOperator<cvcuda::Merge>();

    ResourceGuard guard(stream);
    for (const auto &batch : input) guard.add(LockMode::LOCK_MODE_READ, {*batch});
    guard.add(LockMode::LOCK_MODE_WRITE, {*output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});
    guard.run([&op, &stream, &bands, &output]()
              { op->submit(stream.handle(), bands.data(), static_cast<int32_t>(bands.size()), output->impl()); });
    return *output;
}

std::shared_ptr<ImageBatchVarShape> MergeVarShape(const std::vector<std::shared_ptr<ImageBatchVarShape>> &input,
                                                  nvcv::ImageFormat format, std::optional<Stream *> pstream)
{
    channelcopy::ValidateBands(input);
#ifdef NB_FREE_THREADED
    auto                         mutexes = channelcopy::BatchMutexes(input);
    nvcvpy::priv::BatchLockScope batchScope(mutexes.data(), mutexes.size());
#endif
    channelcopy::CheckImage(*input.front());
    if (format.numChannels() != static_cast<int>(input.size()))
        throw py::value_error("Output format channel count must match the band count");
    auto stream = native::GetStream(pstream).sharedStream();
    auto output = channelcopy::CreateImageBatch(*input.front(), format, *stream);
    MergeVarShapeInto(output, input, stream.get());
    return output;
}

} // namespace

void ExportOpMerge(py::module_ &m)
{
    using namespace nanobind::literals;
    m.def("merge", NvtxTrace("cvcuda.merge", &Merge), "src"_a, py::kw_only(), "layout"_a = nullptr,
          "stream"_a = nullptr, R"pbdoc(
        Copy ordered single-channel tensors into a newly allocated tensor's channels.

        Matches Pillow Image.merge pixel-copy semantics, without a mode or color conversion.
        Supports HWC, NHWC, CHW and NCHW and U8/S8/U16/S16/U32/S32/F16/F32/F64.
        All 1-4 bands must have identical positive shapes, layouts and scalar dtypes, with C=1.
        Repeated read-only inputs are allowed, for example merge([gray, gray, gray]).
        Even one band makes a copy. Work is asynchronous on the supplied/current stream.

        Storage must be CUDA-accessible and pitch-linear, with base addresses and positive
        strides aligned to the scalar element size. Non-singleton strides must cover
        the inner dimensions' byte span in layout order; singleton strides need not.
        Row/sample padding is allowed and is not written.

        Args:
            src (Sequence[cvcuda.Tensor]): Ordered single-channel source tensors.
            layout (cvcuda.TensorLayout, optional): Output layout; defaults to the first band's
                layout. May select interleaved or planar output but must preserve batch rank.
            stream (cvcuda.Stream, optional): CUDA stream for submission.

        Returns:
            cvcuda.Tensor: Newly allocated tensor with len(src) channels.
    )pbdoc");
    m.def("merge_into", NvtxTrace("cvcuda.merge_into", &MergeInto), "dst"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Copy ordered bands into a preallocated tensor; return dst unchanged.

        Uses merge's types, storage requirements and asynchronous semantics.
        The destination determines the output layout and must match the bands'
        dtype, rank, sample count, height and width, with len(src) channels.
        Input/output byte spans must not overlap, including internal padding.
        Read-only source bands may alias each other. No resizing or broadcasting occurs.

        Args:
            dst (cvcuda.Tensor): Preallocated destination tensor.
            src (Sequence[cvcuda.Tensor]): Ordered single-channel source tensors.
            stream (cvcuda.Stream, optional): CUDA stream for submission.

        Returns:
            cvcuda.Tensor: The original dst tensor.
    )pbdoc");
    m.def("merge", NvtxTrace("cvcuda.merge", &MergeVarShape), "src"_a, py::kw_only(), "format"_a, "stream"_a = nullptr,
          R"pbdoc(
        Copy ordered band batches into a newly allocated image batch's storage channels.

        All 1-4 bands must be nonempty uniform-format batches of one-plane, single-channel
        images with the same scalar dtype, image count and corresponding positive sizes.
        Semantic format tags may differ across bands.
        Supports U8/S8/U16/S16/U32/S32/F16/F32/F64. Sizes, pitches and allocations may
        vary across image indices. The required output format selects one interleaved
        plane or full-resolution single-channel planes with len(src) channels of the
        bands' dtype. No color model or planar/interleaved format is guessed.

        Copies preserve every element's bits, even for one band; outputs are not views.
        Channels follow physical component/plane order without interpreting color models,
        swizzles or alpha. Repeated read-only bands or images are allowed. Storage must
        be CUDA-accessible and pitch-linear, with scalar-aligned bases and row pitches
        and positive pitches covering a full row. Components must be contiguous,
        byte-aligned, equal-width scalar elements without intra-pixel padding. Subsampled,
        sub-byte, unequal-component and hybrid plane packings, extra-channel metadata and
        CUDA arrays are unsupported. Padding is not written.
        Work is asynchronous on the supplied/current stream. Batches, images, descriptors
        and backing storage must remain valid and must not be mutated by callers until
        completion; the binding retains submitted resources on the stream.

        Args:
            src (Sequence[cvcuda.ImageBatchVarShape]): Ordered single-channel source batches.
            format (cvcuda.Format): Required output image format, specifying dtype and storage layout.
            stream (cvcuda.Stream, optional): CUDA stream for submission.

        Returns:
            cvcuda.ImageBatchVarShape: Newly allocated batch with len(src) channels per image.
    )pbdoc");
    m.def("merge_into", NvtxTrace("cvcuda.merge_into", &MergeVarShapeInto), "dst"_a, "src"_a, py::kw_only(),
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Copy ordered band batches into a preallocated image batch; return dst unchanged.

        Uses the image-batch merge types, formats, storage requirements and asynchronous
        semantics. The destination's uniform format determines the output storage layout.
        Every batch must be nonempty with the same image count and corresponding positive
        sizes; all scalar dtypes must match, and dst must have len(src) channels.
        Every write/read and write/write overlap across the entire submission is rejected,
        including across image indices, bands, planes and padding inside their byte spans.
        Read-only sources may alias each other. Padding is not written; there is no resizing
        or broadcasting.

        Args:
            dst (cvcuda.ImageBatchVarShape): Preallocated destination image batch.
            src (Sequence[cvcuda.ImageBatchVarShape]): Ordered single-channel source batches.
            stream (cvcuda.Stream, optional): CUDA stream for submission.

        Returns:
            cvcuda.ImageBatchVarShape: The original dst batch.
    )pbdoc");
}

} // namespace cvcudapy
