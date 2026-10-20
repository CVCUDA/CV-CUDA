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

#include "../WorkspaceCache.hpp"
#include "NativeBinding.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpTvResize.hpp>

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace cvcudapy {
namespace {

using native::Image;
using native::ImageBatchVarShape;
using native::Stream;
using native::Tensor;

class TvResizeArgumentError final : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

void Require(bool valid, const char *message)
{
    if (!valid)
        throw TvResizeArgumentError(message);
}

void CheckInterpolation(NVCVInterpolationType interp)
{
    Require(interp == NVCV_INTERP_NEAREST || interp == NVCV_INTERP_NEAREST_FLOOR || interp == NVCV_INTERP_LINEAR
                || interp == NVCV_INTERP_CUBIC,
            "Unsupported interpolation mode");
}

bool SupportedType(nvcv::DataType type)
{
    return type == nvcv::TYPE_U8 || type == nvcv::TYPE_U16 || type == nvcv::TYPE_S16 || type == nvcv::TYPE_F16
        || type == nvcv::TYPE_F32;
}

int32_t Extent(int64_t value)
{
    Require(value > 0 && value <= std::numeric_limits<int32_t>::max(), "Extent must be positive and fit int32");
    return static_cast<int32_t>(value);
}

struct TensorExtents
{
    int32_t n;
    int32_t h;
    int32_t w;
    int32_t c;
};

TensorExtents GetExtents(const nvcv::Tensor &tensor)
{
    auto layout = tensor.layout();
    bool planar = layout == nvcv::TENSOR_CHW || layout == nvcv::TENSOR_NCHW;
    Require(planar || layout == nvcv::TENSOR_HWC || layout == nvcv::TENSOR_NHWC, "Unsupported tensor layout");
    Require(SupportedType(tensor.dtype()), "Unsupported scalar tensor dtype");
    auto    shape = tensor.shape();
    int     nAxis = layout.find('N');
    int64_t n     = nAxis < 0 ? 1 : shape[nAxis];
    Require(n >= 0 && n <= 65535, "Batch size must be in [0,65535]");
    TensorExtents result{static_cast<int32_t>(n), Extent(shape[layout.find('H')]), Extent(shape[layout.find('W')]),
                         Extent(shape[layout.find('C')])};
    Require(planar ? n * result.c <= 65535 : result.c <= 4, "Unsupported channel count");
    return result;
}

int32_t CheckBatch(const nvcv::ImageBatchVarShape &batch)
{
    Require(batch.numImages() <= 65535, "Batch size exceeds 65535");
    if (batch.numImages() == 0)
        return 1;
    auto format = batch.uniqueFormat();
    Require(static_cast<bool>(format), "Image batch must have a uniform format");
    int channels = format.numChannels();
    Require(format.numPlanes() == 1 && format.memLayout() == nvcv::MemLayout::PITCH_LINEAR
                && (channels == 1 || channels == 3 || channels == 4),
            "Unsupported image format");
    auto type   = format.planeDataType(0);
    auto scalar = type.channelType(0);
    Require(SupportedType(scalar) && type.numChannels() == channels, "Unsupported image scalar dtype");
    for (int c = 1; c < channels; ++c)
        Require(type.channelType(c) == scalar, "Image channels must have the same scalar dtype");
    for (int i = 0; i < batch.numImages(); ++i)
    {
        Extent(batch[i].size().h);
        Extent(batch[i].size().w);
    }
    return channels;
}

int32_t SizeInteger(py::handle value)
{
    if (!PyIndex_Check(value.ptr()))
        throw py::type_error("Size extents must be integers");
    auto index = py::steal<py::object>(PyNumber_Index(value.ptr()));
    if (!index.is_valid())
        throw py::python_error();
    int  overflow = 0;
    auto number   = PyLong_AsLongLongAndOverflow(index.ptr(), &overflow);
    if (PyErr_Occurred())
        throw py::python_error();
    if (overflow || number <= 0 || number > std::numeric_limits<int32_t>::max())
        throw py::value_error("Size extents must be positive int32 values");
    return static_cast<int32_t>(number);
}

std::vector<int32_t> ParseSize(py::handle value)
{
    if (value.is_none())
        return {};
    if (PyIndex_Check(value.ptr()))
        return {SizeInteger(value)};
    if (!PySequence_Check(value.ptr()) || PyUnicode_Check(value.ptr()) || PyBytes_Check(value.ptr()))
        throw py::type_error("size must be an integer or a sequence of integers");
    Py_ssize_t count = PySequence_Size(value.ptr());
    if (count < 0)
        throw py::python_error();
    if (count != 1 && count != 2)
        throw py::value_error("size must contain one or two extents");
    std::vector<int32_t> size;
    for (Py_ssize_t i = 0; i < count; ++i)
    {
        auto item = py::steal<py::object>(PySequence_GetItem(value.ptr(), i));
        if (!item.is_valid())
            throw py::python_error();
        size.push_back(SizeInteger(item));
    }
    return size;
}

struct SizeRequest
{
    std::vector<int32_t>   size;
    std::optional<int32_t> maximum;

    SizeRequest(py::handle value, py::handle maxSize)
    {
        size = ParseSize(value);
        if (!maxSize.is_none())
            maximum = SizeInteger(maxSize);
        if (size.empty() && !maximum.has_value())
            throw py::value_error("size=None requires max_size");
        if (maximum.has_value() && (size.size() == 2 || (size.size() == 1 && *maximum <= size[0])))
            throw py::value_error("max_size requires a shorter-edge size smaller than max_size");
    }

    nvcv::Size2D apply(int32_t h, int32_t w) const
    {
        if (size.size() == 2)
            return {size[1], size[0]};
        const int32_t shorter = std::min(h, w);
        const int32_t longer  = std::max(h, w);
        int64_t       newShort;
        int64_t       newLong;
        if (size.empty())
        {
            newLong  = *maximum;
            newShort = static_cast<int64_t>(static_cast<double>(newLong) * shorter / longer);
        }
        else
        {
            newShort = size[0];
            newLong  = static_cast<int64_t>(static_cast<double>(newShort) * longer / shorter);
            if (maximum.has_value() && newLong > *maximum)
            {
                // The cap uses the already truncated longer edge, not the original aspect ratio.
                newShort = static_cast<int64_t>(static_cast<double>(*maximum) * static_cast<double>(newShort)
                                                / static_cast<double>(newLong));
                newLong  = *maximum;
            }
        }
        if (newShort <= 0 || newLong <= 0 || newLong > std::numeric_limits<int32_t>::max())
            throw py::value_error("Resulting spatial extent is zero or exceeds int32");
        return w <= h ? nvcv::Size2D{static_cast<int32_t>(newShort), static_cast<int32_t>(newLong)}
                      : nvcv::Size2D{static_cast<int32_t>(newLong), static_cast<int32_t>(newShort)};
    }
};

cvcuda::WorkspaceRequirements Requirements(const cvcuda::TvResize &op, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                           NVCVInterpolationType interp, bool aa)
{
    auto in  = GetExtents(src);
    auto out = GetExtents(dst);
    Require(src.dtype() == dst.dtype() && src.layout() == dst.layout() && in.n == out.n && in.c == out.c,
            "Input/output dtype, layout, count and channels must match");
    return op.getWorkspaceRequirements(in.n, in.h, in.w, out.h, out.w, in.c, interp, aa);
}

cvcuda::WorkspaceRequirements Requirements(const cvcuda::TvResize &op, const nvcv::ImageBatchVarShape &src,
                                           const nvcv::ImageBatchVarShape &dst, NVCVInterpolationType interp, bool aa)
{
    int32_t channels = CheckBatch(src);
    CheckBatch(dst);
    Require(src.numImages() == dst.numImages() && src.uniqueFormat() == dst.uniqueFormat(),
            "Input/output image count and uniform format must match");
    int                  n = src.numImages();
    // Empty C queries still require nonnull extent arrays.
    std::vector<int32_t> sh(std::max(n, 1), 1);
    std::vector<int32_t> sw(std::max(n, 1), 1);
    std::vector<int32_t> dh(std::max(n, 1), 1);
    std::vector<int32_t> dw(std::max(n, 1), 1);
    for (int i = 0; i < n; ++i)
    {
        sh[i] = src[i].size().h;
        sw[i] = src[i].size().w;
        dh[i] = dst[i].size().h;
        dw[i] = dst[i].size().w;
    }
    return op.getWorkspaceRequirements(n, sh.data(), sw.data(), dh.data(), dw.data(), channels, interp, aa);
}

class PyTvResize : public OperatorResource
{
public:
    class Key : public nvcvpy::IKey
    {
    private:
        size_t doGetHash() const override
        {
            return 0;
        }

        bool doIsCompatible(const nvcvpy::IKey &other) const override
        {
            return dynamic_cast<const Key *>(&other) != nullptr;
        }
    };

    template<class Container>
    void submit(cudaStream_t stream, const Container &src, const Container &dst, NVCVInterpolationType interp,
                bool aa) const
    {
        CheckInterpolation(interp);
        auto requirements = Requirements(m_op, src, dst, interp, aa);
        auto workspace    = WorkspaceCache::instance().get(requirements, stream);
        m_op(stream, workspace.get(), src, dst, interp, aa);
    }

private:
    cvcuda::TvResize m_op;
};

template<class Container>
Container &ResizeInto(Container &dst, Container &src, NVCVInterpolationType interp, std::optional<bool> antialias,
                      std::optional<Stream *> pstream)
{
    Stream       &stream = native::GetStream(pstream);
    auto          op     = CreateOperatorEx<PyTvResize>();
    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {src});
    guard.add(LockMode::LOCK_MODE_WRITE, {dst});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});
    guard.run([&op, &stream, &src, &dst, interp, antialias]
              { op->submit(stream.handle(), src.impl(), dst.impl(), interp, antialias.value_or(false)); });
    return dst;
}

std::shared_ptr<Tensor> ResizeTensor(Tensor &src, py::object size, NVCVInterpolationType interp, py::object maxSize,
                                     std::optional<bool> antialias, std::optional<Stream *> pstream)
{
    CheckInterpolation(interp);
    auto extents    = GetExtents(src.impl());
    auto outputSize = SizeRequest(size, maxSize).apply(extents.h, extents.w);
    cvcuda::TvResize{}.getWorkspaceRequirements(extents.n, extents.h, extents.w, outputSize.h, outputSize.w, extents.c,
                                                interp, antialias.value_or(false));
    auto                 shape = src.impl().shape();
    std::vector<int64_t> dims(shape.rank());
    for (int i = 0; i < shape.rank(); ++i) dims[i] = shape[i];
    dims[shape.layout().find('H')] = outputSize.h;
    dims[shape.layout().find('W')] = outputSize.w;
    auto dst = native::CreateTensor({dims.data(), shape.rank(), shape.layout()}, src.impl().dtype());
    ResizeInto(*dst, src, interp, antialias, pstream);
    return dst;
}

std::shared_ptr<ImageBatchVarShape> ResizeBatch(ImageBatchVarShape &src, py::object size, NVCVInterpolationType interp,
                                                py::object maxSize, std::optional<bool> antialias,
                                                std::optional<Stream *> pstream)
{
    CheckInterpolation(interp);
    int32_t                   channels = CheckBatch(src.impl());
    SizeRequest               request(size, maxSize);
    std::vector<nvcv::Size2D> sizes;
    int                       count = src.numImages();
    std::vector<int32_t>      sh(std::max(count, 1), 1);
    std::vector<int32_t>      sw(std::max(count, 1), 1);
    std::vector<int32_t>      dh(std::max(count, 1), 1);
    std::vector<int32_t>      dw(std::max(count, 1), 1);
    for (int i = 0; i < count; ++i)
    {
        auto inputSize  = src.impl()[i].size();
        auto outputSize = request.apply(inputSize.h, inputSize.w);
        sizes.push_back(outputSize);
        sh[i] = inputSize.h;
        sw[i] = inputSize.w;
        dh[i] = outputSize.h;
        dw[i] = outputSize.w;
    }
    cvcuda::TvResize{}.getWorkspaceRequirements(count, sh.data(), sw.data(), dh.data(), dw.data(), channels, interp,
                                                antialias.value_or(false));
    auto dst   = ImageBatchVarShape::Create(src.capacity());
    auto image = src.begin();
    for (auto outputSize : sizes)
    {
        auto resized = Image::Create({outputSize.w, outputSize.h}, (*image)->format(), 0);
        dst->pushBack(*resized);
        ++image;
    }
    ResizeInto(*dst, src, interp, antialias, pstream);
    return dst;
}

} // namespace

void ExportOpTvResize(py::module_ &m)
{
    using namespace nanobind::literals;

    const char *tensorDoc = R"pbdoc(
        Resize a tensor with TorchVision-style half-pixel sampling.

        Args:
            src (cvcuda.Tensor): Scalar U8/U16/S16/F16/F32 HWC/NHWC (C1..4) or CHW/NCHW tensor.
                Planar C is positive, N*C <= 65535; N <= 65535. Only H/W change.
                The shared Tensor API requires all extents to be positive, so empty Tensor submits are unavailable.
            size (int or sequence, optional): Shorter edge (integer or one element), or exact (H,W).
                Longer edge is truncated. If omitted, max_size sets the longer edge, including enlargement.
            interp (cvcuda.Interp): LINEAR (default), CUBIC, NEAREST (exact), or NEAREST_FLOOR.
            max_size (int, optional): Cap longer edge after truncation; must exceed requested shorter edge.
                Not allowed with a two-element size. Resulting extents must be positive int32.
            antialias (bool, optional): Defaults to True; None means False. Ignored for nearest.
                Cubic AA uses Keys a=-0.5 even when enlarging, plain cubic uses a=-0.75.
            stream (cvcuda.Stream, optional): Keyword-only CUDA stream; execution is asynchronous.

        Returns:
            cvcuda.Tensor: Newly allocated output with matching dtype/layout/count/channels.

        Integer results round ties-to-even and saturate (including U16/S16 cubic overshoot).
        Interpolation guarantees cover finite inputs with finite representable results.
        See the TvResize C API for the complete sampling and workspace contract.
    )pbdoc";
    const char *batchDoc  = R"pbdoc(
        Resize each image in a uniform-format variable-shape batch asynchronously.

        Args:
            src (cvcuda.ImageBatchVarShape): Uniform single-plane pitch-linear C1/3/4 images,
                scalar U8/U16/S16/F16/F32, at most 65535 samples. Mixed formats are rejected.
            size (int or sequence, optional): Shorter edge or exact (H,W), applied to each image.
                Omitted size requires max_size and may enlarge images.
            interp (cvcuda.Interp): LINEAR (default), CUBIC, NEAREST, or NEAREST_FLOOR.
            max_size (int, optional): Longer-edge cap, using the already-truncated resized aspect ratio.
                Must exceed requested shorter edge; forbidden with exact (H,W). Extents are positive int32.
            antialias (bool, optional): True by default; None means False; ignored for nearest.
            stream (cvcuda.Stream, optional): Keyword-only CUDA stream.

        Returns:
            cvcuda.ImageBatchVarShape: New output batch preserving count, capacity and format.

        Sampling and rounding follow tvresize for Tensor inputs; empty batches are supported.
    )pbdoc";
    const char *intoDoc   = R"pbdoc(
        Resize into a supplied destination, returning that same object asynchronously.

        Args:
            dst (cvcuda.Tensor or cvcuda.ImageBatchVarShape): Destination defining output H/W.
                Dtype, layout, sample count and channels/format must match src.
            src (cvcuda.Tensor or cvcuda.ImageBatchVarShape): Source; supported types/layouts
                and numerical semantics are the same as tvresize.
            interp (cvcuda.Interp): LINEAR (default), CUBIC, NEAREST, or NEAREST_FLOOR.
            antialias (bool, optional): True by default; None means False; ignored for nearest.
            stream (cvcuda.Stream, optional): Keyword-only CUDA stream.

        Returns:
            cvcuda.Tensor or cvcuda.ImageBatchVarShape: The supplied dst object.

        size and max_size are not accepted. Source and destination storage must not overlap,
        including identity resize. Empty matched containers enqueue no work.
    )pbdoc";

    m.def("tvresize", NvtxTrace("cvcuda.tvresize", &ResizeTensor), "src"_a, "size"_a = nullptr,
          "interp"_a = NVCV_INTERP_LINEAR, "max_size"_a = nullptr, "antialias"_a = true, py::kw_only(),
          "stream"_a = nullptr, tensorDoc);
    m.def("tvresize", NvtxTrace("cvcuda.tvresize", &ResizeBatch), "src"_a, "size"_a = nullptr,
          "interp"_a = NVCV_INTERP_LINEAR, "max_size"_a = nullptr, "antialias"_a = true, py::kw_only(),
          "stream"_a = nullptr, batchDoc);
    m.def("tvresize_into", NvtxTrace("cvcuda.tvresize_into", &ResizeInto<Tensor>), "dst"_a, "src"_a,
          "interp"_a = NVCV_INTERP_LINEAR, "antialias"_a = true, py::kw_only(), "stream"_a = nullptr,
          py::return_value_policy::reference, intoDoc);
    m.def("tvresize_into", NvtxTrace("cvcuda.tvresize_into", &ResizeInto<ImageBatchVarShape>), "dst"_a, "src"_a,
          "interp"_a = NVCV_INTERP_LINEAR, "antialias"_a = true, py::kw_only(), "stream"_a = nullptr,
          py::return_value_policy::reference, intoDoc);
}

} // namespace cvcudapy
