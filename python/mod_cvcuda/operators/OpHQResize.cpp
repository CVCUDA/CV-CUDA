/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpHQResize.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ImageFormat.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <nvcv/python/TensorBatch.hpp>

#include <array>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cvcudapy {

namespace {

using Roi  = pybind11::tuple;
using Rois = std::vector<Roi>;

class HQResizeError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

inline void GetMinMagInterpolation(NVCVInterpolationType                      &minInterpolationArg,
                                   NVCVInterpolationType                      &magInterpolationArg,
                                   const std::optional<NVCVInterpolationType> &interpolation,
                                   const std::optional<NVCVInterpolationType> &minInterpolation,
                                   const std::optional<NVCVInterpolationType> &magInterpolation)
{
    if (interpolation)
    {
        if (minInterpolation || magInterpolation)
        {
            throw py::value_error(
                "When `interpolation` is specified, the `min_interpolation` and `mag_interpolation` should not be "
                "specified.");
        }
        minInterpolationArg = magInterpolationArg = *interpolation;
    }
    else
    {
        if (!minInterpolation || !magInterpolation)
        {
            throw py::value_error(
                "Either `interpolation`, or both `min_interpolation` and `mag_interpolation` must be specified.");
        }
        minInterpolationArg = *minInterpolation;
        magInterpolationArg = *magInterpolation;
    }
}

inline void ParseRoi(HQResizeRoiF &parsedRoi, const Roi &roi, int ndim)
{
    assert(ndim == 2 || ndim == 3);
    if (auto roiSize = roi.size(); roiSize != static_cast<decltype(roiSize)>(2 * ndim))
    {
        if (ndim == 2)
        {
            throw HQResizeError(
                "Got wrong number of ROI components. For image resize, 4 integers are expected: "
                "low_height, low_width, high_height, high_width describing the bounding box for "
                "the input.");
        }
        else
        {
            throw HQResizeError(
                "Got wrong number of ROI components. For volumetric data, 6 integers are expected: "
                "low_depth, low_height, low_width, high_depth, high_height, high_width "
                "describing the bounding box for the input.");
        }
    }
    for (int d = 0; d < ndim; d++)
    {
        parsedRoi.lo[d] = roi[d].cast<float>();
    }
    for (int d = 0; d < ndim; d++)
    {
        parsedRoi.hi[d] = roi[ndim + d].cast<float>();
    }
}

class RoiHelper
{
public:
    RoiHelper(const std::optional<Rois> &maybeRois, int ndim)
        : m_ndim{ndim}
    {
        if (maybeRois)
        {
            auto &rois = *maybeRois;
            m_rois.resize(rois.size());
            for (uint64_t i = 0; i < rois.size(); i++)
            {
                auto &roi       = m_rois[i];
                auto &passedRoi = rois[i];
                ParseRoi(roi, passedRoi, ndim);
            }
        }
    }

    RoiHelper(const std::optional<Roi> &maybeRoi, int ndim)
        : m_ndim{ndim}
    {
        if (maybeRoi)
        {
            m_rois.resize(1);
            ParseRoi(m_rois[0], *maybeRoi, ndim);
        }
    }

    HQResizeRoisF NonOwningHandle()
    {
        auto                size = static_cast<int32_t>(m_rois.size());
        const HQResizeRoiF *data = size == 0 ? nullptr : m_rois.data();
        return {size, m_ndim, data};
    }

private:
    int                       m_ndim;
    std::vector<HQResizeRoiF> m_rois;
};

inline HQResizeTensorShapeI TensorShape(const nvcv::TensorLayout &layout, const nvcv::TensorShape &shape,
                                        int resizeNDim)
{
    assert(resizeNDim == 2 || resizeNDim == 3);

    constexpr std::array<char, 3> shapeArgLayout = {'D', 'H', 'W'};
    HQResizeTensorShapeI          tensorShape{};
    for (int d = 0; d < resizeNDim; d++)
    {
        int axis = layout.find(shapeArgLayout[d + 3 - resizeNDim]);
        if (axis < 0)
        {
            throw HQResizeError(
                "The layout of an input tensor to the resize operator must contain HW extents in the layout (for "
                "images) or DHW extents (for 3D resampling). Some extents are missing in the input tensor.");
        }
        tensorShape.extent[d] = static_cast<int32_t>(shape[axis]);
    }
    int channelAxis         = layout.find('C');
    tensorShape.numChannels = channelAxis < 0 ? 1 : static_cast<int32_t>(shape[channelAxis]);
    tensorShape.ndim        = resizeNDim;
    return tensorShape;
}

inline bool IsPlanarTensorBatchLayout(const nvcv::TensorLayout &layout)
{
    return layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW;
}

template<typename Visitor>
bool VisitExpandedTensorBatchShapes(const TensorBatch &batch, int resizeNDim, const nvcv::TensorLayout &layout,
                                    Visitor &&visitor)
{
    const bool planar = IsPlanarTensorBatchLayout(layout);
    for (int i = 0; i < batch.numTensors(); i++)
    {
        HQResizeTensorShapeI shape = TensorShape(layout, batch[i].shape(), resizeNDim);
        const int            reps  = planar ? shape.numChannels : 1;
        if (planar)
        {
            shape.numChannels = 1;
        }

        for (int r = 0; r < reps; r++)
        {
            if (!visitor(shape))
            {
                return false;
            }
        }
    }
    return true;
}

class BatchShapesHelper
{
public:
    explicit BatchShapesHelper(const nvcv::ImageBatchVarShape &batch)
        : m_ndim(2)
    {
        // Planar (multi-plane, e.g. RGB8p) batches are processed plane-by-plane: the operator expands
        // to numImages*channels single-channel samples, so the workspace must be sized the same way.
        const auto    fmt         = batch.uniqueFormat();
        const int32_t planes      = fmt ? fmt.numPlanes() : 1;
        const int32_t fmtChannels = fmt ? fmt.numChannels() : 1;
        const bool    planar      = planes > 1;
        m_numChannels             = planar ? 1 : fmtChannels;
        const int32_t reps        = planar ? planes : 1;

        int32_t numImages = batch.numImages();
        m_shapes.reserve(static_cast<size_t>(numImages) * reps);
        for (int i = 0; i < numImages; i++)
        {
            const auto &imgShape = batch[i].size();
            for (int r = 0; r < reps; r++)
            {
                HQResizeTensorShapeI shape{};
                shape.extent[0]   = imgShape.h;
                shape.extent[1]   = imgShape.w;
                shape.ndim        = 2;
                shape.numChannels = m_numChannels;
                m_shapes.push_back(shape);
            }
        }
    }

    explicit BatchShapesHelper(const TensorBatch &batch)
        : m_ndim(batch.layout().find('D') >= 0 ? 3 : 2)
        , m_numChannels(-1)
        , m_layout(batch.layout())
        , m_dtype(batch.dtype())
        , m_numTensors(batch.numTensors())
    {
        const bool planar = IsPlanarTensorBatchLayout(m_layout);
        if (planar)
        {
            // Each channel plane becomes a single-channel sample; matches the operator's expansion.
            m_numChannels = 1;
        }
        m_shapes.reserve(m_numTensors);

        auto appendShape = [this, planar](const HQResizeTensorShapeI &shape)
        {
            if (!planar)
            {
                if (m_shapes.empty())
                {
                    m_numChannels = shape.numChannels;
                }
                else if (m_numChannels != shape.numChannels)
                {
                    m_numChannels = -1;
                }
            }
            m_shapes.push_back(shape);
            return true;
        };
        VisitExpandedTensorBatchShapes(batch, m_ndim, m_layout, appendShape);
    }

    HQResizeTensorShapesI NonOwningHandle()
    {
        auto size = static_cast<int32_t>(m_shapes.size());
        return {size ? m_shapes.data() : nullptr, size, m_ndim, m_numChannels};
    }

    bool Matches(const TensorBatch &batch) const
    {
        if (batch.numTensors() != m_numTensors || batch.layout() != m_layout || batch.dtype() != m_dtype)
        {
            return false;
        }

        size_t shapeIndex   = 0;
        auto   compareShape = [this, &shapeIndex](const HQResizeTensorShapeI &shape)
        {
            if (shapeIndex >= m_shapes.size() || !SameShape(m_shapes[shapeIndex], shape))
            {
                return false;
            }
            shapeIndex++;
            return true;
        };
        return VisitExpandedTensorBatchShapes(batch, m_ndim, m_layout, compareShape) && shapeIndex == m_shapes.size();
    }

private:
    static bool SameShape(const HQResizeTensorShapeI &lhs, const HQResizeTensorShapeI &rhs)
    {
        if (lhs.ndim != rhs.ndim || lhs.numChannels != rhs.numChannels)
        {
            return false;
        }
        for (int d = 0; d < lhs.ndim; d++)
        {
            if (lhs.extent[d] != rhs.extent[d])
            {
                return false;
            }
        }
        return true;
    }

    int32_t                           m_ndim;
    int32_t                           m_numChannels;
    std::vector<HQResizeTensorShapeI> m_shapes;
    nvcv::TensorLayout                m_layout;
    nvcv::DataType                    m_dtype;
    int32_t                           m_numTensors{-1};
};

class TensorBatchRequirementsCache
{
public:
    TensorBatchRequirementsCache(BatchShapesHelper inShapes, BatchShapesHelper outShapes,
                                 NVCVInterpolationType minInterpolation, NVCVInterpolationType magInterpolation,
                                 bool antialias, HQResizeRoisF rois, const cvcuda::WorkspaceRequirements &requirements)
        : m_inShapes(std::move(inShapes))
        , m_outShapes(std::move(outShapes))
        , m_minInterpolation(minInterpolation)
        , m_magInterpolation(magInterpolation)
        , m_antialias(antialias)
        , m_roiNDim(rois.ndim)
        , m_requirements(requirements)
    {
        if (rois.size > 0)
        {
            m_rois.assign(rois.roi, rois.roi + rois.size);
        }
    }

    bool Matches(const TensorBatch &in, const TensorBatch &out, NVCVInterpolationType minInterpolation,
                 NVCVInterpolationType magInterpolation, bool antialias, HQResizeRoisF rois) const
    {
        if (minInterpolation != m_minInterpolation || magInterpolation != m_magInterpolation || antialias != m_antialias
            || rois.ndim != m_roiNDim || rois.size != static_cast<int32_t>(m_rois.size()) || !m_inShapes.Matches(in)
            || !m_outShapes.Matches(out))
        {
            return false;
        }

        for (int i = 0; i < rois.size; i++)
        {
            for (int d = 0; d < rois.ndim; d++)
            {
                if (m_rois[i].lo[d] != rois.roi[i].lo[d] || m_rois[i].hi[d] != rois.roi[i].hi[d])
                {
                    return false;
                }
            }
        }
        return true;
    }

    const cvcuda::WorkspaceRequirements &requirements() const
    {
        return m_requirements;
    }

private:
    BatchShapesHelper             m_inShapes;
    BatchShapesHelper             m_outShapes;
    NVCVInterpolationType         m_minInterpolation;
    NVCVInterpolationType         m_magInterpolation;
    bool                          m_antialias;
    int32_t                       m_roiNDim;
    std::vector<HQResizeRoiF>     m_rois;
    cvcuda::WorkspaceRequirements m_requirements;
};

inline Shape ResizedTensorShape(const nvcv::TensorLayout &srcLayout, const nvcv::TensorShape &srcShape,
                                const Shape &outShape)
{
    auto resizeNDim = static_cast<int>(outShape.size());
    if (resizeNDim != 2 && resizeNDim != 3)
    {
        throw HQResizeError(
            "The `out_shape` must be a tuple of 2 or 3 integers (for 2D or 3D resampling respectively).");
    }

    bool hasDepth = srcLayout.find('D') >= 0;
    if (int expectedNDim = hasDepth ? 3 : 2; expectedNDim != resizeNDim)
    {
        if (hasDepth)
        {
            throw HQResizeError(
                "The input tensor contains depth extent (`D`) in the layout. For 3D resize, please specify the resized "
                "shape for 3 extents: depth, height, and width. Got 2 extents.");
        }
        else
        {
            throw HQResizeError(
                "Expected the resized shape to consists of 2 integers: for resized height and width. Got 3 integers.");
        }
    }

    constexpr std::array<char, 3> shapeArgLayout = {'D', 'H', 'W'};
    std::array<int, 3>            shapeArg       = {};
    for (int d = 0; d < resizeNDim; d++)
    {
        shapeArg[d] = outShape[d].cast<int>();
    }

    Shape resizedShape(srcShape.rank());
    for (int i = 0; i < srcShape.rank(); i++)
    {
        resizedShape[i] = srcShape[i];
    }

    assert(srcShape.rank() == srcLayout.rank());
    for (int d = 0; d < resizeNDim; d++)
    {
        int axis = srcLayout.find(shapeArgLayout[d + 3 - resizeNDim]);
        if (axis < 0)
        {
            throw HQResizeError(
                "The layout of an input tensor to the resize operator must contain HW extents in the layout (for "
                "images) or DHW extents (for 3D resampling). Some extents are missing in the input tensor.");
        }
        resizedShape[axis] = shapeArg[d];
    }
    return resizedShape;
}

class PyOpHQResize : public nvcvpy::Container // NOSONAR: operator wrappers share the Python cache hierarchy.
{
public:
    // Define a Key class to be used by the cache to fetch similar items for potential reuse.
    class Key : public nvcvpy::IKey
    {
    public:
        // the filters are generated by the operator constructor for a given device
        explicit Key(int deviceId)
            : m_deviceId{deviceId}
        {
        }

    private:
        size_t doGetHash() const override
        {
            return m_deviceId;
        }

        bool doIsCompatible(const nvcvpy::IKey &that_) const override
        {
            const auto *thatKey = dynamic_cast<const Key *>(&that_);
            return thatKey != nullptr && thatKey->m_deviceId == m_deviceId;
        }

        int m_deviceId;
    };

    explicit PyOpHQResize(int deviceId)
        : m_key(deviceId)
    {
    }

    void submit(cudaStream_t stream, const Tensor &in, const Tensor &out, const NVCVInterpolationType minInterpolation,
                const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoiF *roi) const
    {
        if (in.layout() != out.layout())
        {
            throw HQResizeError("Input and output tensors must have the same layout");
        }

        int resizeNDim = in.layout().find('D') >= 0 ? 3 : 2;

        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(in.exportData());
        if (!inAccess)
        {
            throw HQResizeError("Incompatible input tensor layout");
        }

        if (auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(out.exportData()); !outAccess)
        {
            throw HQResizeError("Incompatible input tensor layout");
        }

        auto numSamples = static_cast<int>(inAccess->numSamples());
        auto inShape    = TensorShape(in.layout(), in.shape(), resizeNDim);
        auto outShape   = TensorShape(out.layout(), out.shape(), resizeNDim);

        auto req = m_op.getWorkspaceRequirements(numSamples, inShape, outShape, minInterpolation, magInterpolation,
                                                 antialias, roi);
        auto ws  = WorkspaceCache::instance().get(req, stream);
        m_op(stream, ws.get(), in, out, minInterpolation, magInterpolation, antialias, roi);
    }

    void submit(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation,
                bool antialias, const HQResizeRoisF rois) const
    {
        BatchShapesHelper inShapes(in);
        BatchShapesHelper outShapes(out);
        // numSamples must be the (possibly plane-expanded) shape count, not the image count, so the
        // workspace metadata matches the operator's internal planar expansion.
        auto              inHandle  = inShapes.NonOwningHandle();
        auto              outHandle = outShapes.NonOwningHandle();
        auto req = m_op.getWorkspaceRequirements(inHandle.size, inHandle, outHandle, minInterpolation, magInterpolation,
                                                 antialias, rois);
        auto ws  = WorkspaceCache::instance().get(req, stream);
        m_op(stream, ws.get(), in, out, minInterpolation, magInterpolation, antialias, rois);
    }

    void submit(cudaStream_t stream, const TensorBatch &in, const TensorBatch &out,
                const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation,
                bool antialias, const HQResizeRoisF rois) const
    {
        if (in.layout() != out.layout())
        {
            throw HQResizeError("Input and output batches must have the same layout");
        }

        std::optional<cvcuda::WorkspaceRequirements> req;
        {
            std::scoped_lock lock(m_tensorBatchRequirementsMutex);
            if (m_tensorBatchRequirements
                && m_tensorBatchRequirements->Matches(in, out, minInterpolation, magInterpolation, antialias, rois))
            {
                req = m_tensorBatchRequirements->requirements();
            }
        }

        if (!req)
        {
            BatchShapesHelper inShapes(in);
            BatchShapesHelper outShapes(out);
            // numSamples must be the (possibly plane-expanded) shape count, not the tensor count.
            auto              inHandle  = inShapes.NonOwningHandle();
            auto              outHandle = outShapes.NonOwningHandle();
            req = m_op.getWorkspaceRequirements(inHandle.size, inHandle, outHandle, minInterpolation, magInterpolation,
                                                antialias, rois);

            // Keep retained metadata bounded even if a caller submits an unusually large batch.
            constexpr int32_t kMaxCachedExpandedSamples = 4096;
            if (inHandle.size <= kMaxCachedExpandedSamples && outHandle.size <= kMaxCachedExpandedSamples
                && rois.size <= kMaxCachedExpandedSamples)
            {
                TensorBatchRequirementsCache entry(std::move(inShapes), std::move(outShapes), minInterpolation,
                                                   magInterpolation, antialias, rois, *req);
                std::scoped_lock             lock(m_tensorBatchRequirementsMutex);
                m_tensorBatchRequirements = std::move(entry);
            }
        }

        auto ws = WorkspaceCache::instance().get(*req, stream);
        m_op(stream, ws.get(), in, out, minInterpolation, magInterpolation, antialias, rois);
    }

    // Required override to get the py object container.
    py::object container() const override
    {
        return py::reinterpret_borrow<py::object>(this->ptr());
    }

    // Required override to get the key as the base interface class.
    const nvcvpy::IKey &key() const override
    {
        return m_key;
    }

    static std::shared_ptr<nvcvpy::ICacheItem> fetch(std::vector<std::shared_ptr<nvcvpy::ICacheItem>> &cache)
    {
        assert(!cache.empty());
        return cache[0];
    }

private:
    Key                                                 m_key;
    cvcuda::HQResize                                    m_op;
    mutable std::mutex                                  m_tensorBatchRequirementsMutex;
    mutable std::optional<TensorBatchRequirementsCache> m_tensorBatchRequirements;
};

template<typename Op, typename Src, typename Dst, typename Call>
auto RunGuard(Op &op, Src &src, Dst &dst, Stream &stream, Call &&call)
{
    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {src});
    guard.add(LockMode::LOCK_MODE_WRITE, {dst});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    guard.run(std::forward<Call>(call));
}

auto CreatePyOpHQResize()
{
    int deviceId;
    NVCV_CHECK_THROW(cudaGetDevice(&deviceId));
    return CreateOperatorEx<PyOpHQResize>(deviceId);
}

Tensor TensorHQResizeInto(Tensor &dst, Tensor &src, std::optional<bool> antialias, std::optional<Roi> maybeRoi,
                          std::optional<NVCVInterpolationType> interpolation,
                          std::optional<NVCVInterpolationType> minInterpolation,
                          std::optional<NVCVInterpolationType> magInterpolation, std::optional<Stream> pstream)
{
    Stream stream = pstream ? *pstream : Stream::Current();
    auto   op     = CreatePyOpHQResize();

    bool                hasDepth   = src.layout().find('D') >= 0;
    int                 resizeNDim = hasDepth ? 3 : 2;
    RoiHelper           parsedRoi(maybeRoi, resizeNDim);
    const HQResizeRoiF *roi = parsedRoi.NonOwningHandle().roi;

    NVCVInterpolationType minInterpolationArg;
    NVCVInterpolationType magInterpolationArg;
    GetMinMagInterpolation(minInterpolationArg, magInterpolationArg, interpolation, minInterpolation, magInterpolation);

    RunGuard(op, src, dst, stream,
             [&op, &stream, &src, &dst, &minInterpolationArg, &magInterpolationArg, &antialias, &roi]()
             {
                 op->submit(stream.cudaHandle(), src, dst, minInterpolationArg, magInterpolationArg,
                            antialias.value_or(false), roi);
             });
    return dst;
}

Tensor TensorHQResize(Tensor &src, const Shape &outShape, std::optional<bool> antialias, std::optional<Roi> roi,
                      std::optional<NVCVInterpolationType> interpolation,
                      std::optional<NVCVInterpolationType> minInterpolation,
                      std::optional<NVCVInterpolationType> magInterpolation, std::optional<Stream> pstream)
{
    auto   resizedShape = ResizedTensorShape(src.layout(), src.shape(), outShape);
    Tensor dst          = Tensor::Create(resizedShape, src.dtype(), src.layout());
    return TensorHQResizeInto(dst, src, antialias, roi, interpolation, minInterpolation, magInterpolation, pstream);
}

ImageBatchVarShape VarShapeHQResizeInto(ImageBatchVarShape &dst, const ImageBatchVarShape &src,
                                        std::optional<bool> antialias, const std::optional<Rois> &roi,
                                        std::optional<NVCVInterpolationType> interpolation,
                                        std::optional<NVCVInterpolationType> minInterpolation,
                                        std::optional<NVCVInterpolationType> magInterpolation,
                                        std::optional<Stream>                pstream)
{
    Stream stream = pstream ? *pstream : Stream::Current();
    auto   op     = CreatePyOpHQResize();

    RoiHelper             parsedRoi(roi, 2);
    NVCVInterpolationType minInterpolationArg;
    NVCVInterpolationType magInterpolationArg;
    GetMinMagInterpolation(minInterpolationArg, magInterpolationArg, interpolation, minInterpolation, magInterpolation);

    RunGuard(op, src, dst, stream,
             [&op, &stream, &src, &dst, &minInterpolationArg, &magInterpolationArg, &antialias, &parsedRoi]()
             {
                 op->submit(stream.cudaHandle(), src, dst, minInterpolationArg, magInterpolationArg,
                            antialias.value_or(false), parsedRoi.NonOwningHandle());
             });
    return dst;
}

ImageBatchVarShape VarShapeHQResize(ImageBatchVarShape &src, const std::vector<std::tuple<int, int>> &outShape,
                                    std::optional<bool> antialias, const std::optional<Rois> &roi,
                                    std::optional<NVCVInterpolationType> interpolation,
                                    std::optional<NVCVInterpolationType> minInterpolation,
                                    std::optional<NVCVInterpolationType> magInterpolation,
                                    std::optional<Stream>                pstream)
{
    auto out = ImageBatchVarShape::Create(src.capacity());

    auto numOutSizes = static_cast<int32_t>(outShape.size());
    if (numOutSizes != src.numImages() && numOutSizes != 1)
    {
        throw HQResizeError(
            "The list of output shapes `out_size` must either contain a single shape to be used for all output images "
            "or its length must match the number of input samples.");
    }

    for (int i = 0; i < src.numImages(); ++i)
    {
        auto [size0, size1] = outShape[numOutSizes == 1 ? 0 : i];
        auto image          = Image::Create({size1, size0}, src[i].format());
        out.pushBackImage(image);
    }

    return VarShapeHQResizeInto(out, src, antialias, roi, interpolation, minInterpolation, magInterpolation, pstream);
}

TensorBatch TensorBatchHQResizeInto(TensorBatch &dst, const TensorBatch &src, std::optional<bool> antialias,
                                    const std::optional<Rois> &roi, std::optional<NVCVInterpolationType> interpolation,
                                    std::optional<NVCVInterpolationType> minInterpolation,
                                    std::optional<NVCVInterpolationType> magInterpolation,
                                    std::optional<Stream>                pstream)
{
    Stream stream = pstream ? *pstream : Stream::Current();
    auto   op     = CreatePyOpHQResize();

    bool      hasDepth   = src.layout().find('D') >= 0;
    int       resizeNDim = hasDepth ? 3 : 2;
    RoiHelper parsedRoi(roi, resizeNDim);

    NVCVInterpolationType minInterpolationArg;
    NVCVInterpolationType magInterpolationArg;
    GetMinMagInterpolation(minInterpolationArg, magInterpolationArg, interpolation, minInterpolation, magInterpolation);

    RunGuard(op, src, dst, stream,
             [&op, &stream, &src, &dst, &minInterpolationArg, &magInterpolationArg, &antialias, &parsedRoi]()
             {
                 op->submit(stream.cudaHandle(), src, dst, minInterpolationArg, magInterpolationArg,
                            antialias.value_or(false), parsedRoi.NonOwningHandle());
             });
    return dst;
}

TensorBatch TensorBatchHQResize(TensorBatch &src, const std::vector<Shape> &outShape, std::optional<bool> antialias,
                                const std::optional<Rois> &roi, std::optional<NVCVInterpolationType> interpolation,
                                std::optional<NVCVInterpolationType> minInterpolation,
                                std::optional<NVCVInterpolationType> magInterpolation, std::optional<Stream> pstream)
{
    auto out = TensorBatch::Create(src.numTensors());

    auto numOutSizes = static_cast<int32_t>(outShape.size());
    if (numOutSizes != src.numTensors() && numOutSizes != 1)
    {
        throw HQResizeError(
            "The list of output shapes `out_size` must either contain a single shape to be used for all output tensors "
            "or its length must match the number of input tensors.");
    }

    for (int i = 0; i < src.numTensors(); ++i)
    {
        auto        sampleShape  = outShape[numOutSizes == 1 ? 0 : i];
        const auto &inSample     = src[i];
        auto        resizedShape = ResizedTensorShape(inSample.layout(), inSample.shape(), sampleShape);
        Tensor      dst          = Tensor::Create(resizedShape, src.dtype(), src.layout());
        out.pushBackTensor(dst);
    }

    return TensorBatchHQResizeInto(out, src, antialias, roi, interpolation, minInterpolation, magInterpolation,
                                   pstream);
}

} // namespace

void ExportOpHQResize(py::module &m)
{
    using namespace pybind11::literals;

    m.def("hq_resize", NvtxTrace("cvcuda.hq_resize", &TensorHQResize), "src"_a, "out_size"_a, py::kw_only(),
          "antialias"_a = false, "roi"_a = nullptr, "interpolation"_a = nullptr, "min_interpolation"_a = nullptr,
          "mag_interpolation"_a = nullptr, "stream"_a = nullptr, R"pbdoc(
        Executes the HQ Resize operation on the given cuda stream. The operator
        supports resampling for 2D (images) and 3D volumetric samples.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
                The tensor layout must match: (N)(D)HW(C).
            out_size (tuple): Tuple of 2 or 3 ints describing the output shape in (D)HW layout.
            antialias (bool): If set to true, an antialiasing is enabled for scaling down.
            roi (Tuple): Optional bounding box describing the input's region of interest.
                For 2D resampling it should be (lowH, lowW, highH, highW),
                for 3D: (lowD, lowH, lowW, highD, highH, highW).
                If, for some axis, the low bound is bigger than the high bound,
                the image is flipped across the axis.
            interpolation (cvcuda.Interp): Interpolation type used. Used both for scaling down and up,
                cannot be specified together with (min_interpolation or mag_interpolation).
            min_interpolation (cvcuda.Interp): Interpolation type used for scaling down.
            mag_interpolation (cvcuda.Interp): Interpolation type used for scaling up.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");
    m.def("hq_resize", NvtxTrace("cvcuda.hq_resize", &VarShapeHQResize), "src"_a, "out_size"_a, py::kw_only(),
          "antialias"_a = false, "roi"_a = nullptr, "interpolation"_a = nullptr, "min_interpolation"_a = nullptr,
          "mag_interpolation"_a = nullptr, "stream"_a = nullptr, R"pbdoc(
        Executes the HQ Resize operation on the given cuda stream.


        Args:
            src (cvcuda.ImageBatchVarShape): Input batch of images.
            out_size (tuple): Tuple of 2 ints describing the output shape in HW layout.
            antialias (bool): If set to true, an antialiasing is enabled for scaling down.
            roi (List[Tuple[int]]): Optional bounding boxes describing the input's region of interest.
                It should be a list of tuples. The list length must match the number
                of input tensors or be 1 (so that the same ROI is used for all samples).
                Each tuple must be of the form (lowH, lowW, highH, highW).
                If, for some axis, the low bound is bigger than the high bound,
                the image is flipped across the axis.
            interpolation (cvcuda.Interp): Interpolation type used. Used both for scaling down and up,
                cannot be specified together with (min_interpolation or mag_interpolation).
            min_interpolation (cvcuda.Interp): Interpolation type used for scaling down.
            mag_interpolation (cvcuda.Interp): Interpolation type used for scaling up.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The batch of resized images.

    )pbdoc");
    m.def("hq_resize", NvtxTrace("cvcuda.hq_resize", &TensorBatchHQResize), "src"_a, "out_size"_a, py::kw_only(),
          "antialias"_a = false, "roi"_a = nullptr, "interpolation"_a = nullptr, "min_interpolation"_a = nullptr,
          "mag_interpolation"_a = nullptr, "stream"_a = nullptr, R"pbdoc(
        Executes the HQ Resize operation on the given cuda stream. The operator
        supports resampling for 2D (images) and 3D volumetric samples.


        Args:
            src (cvcuda.TensorBatch): Input batch containing one or more tensors of (D)HW(C) layout.
            out_size (tuple): Tuple of 2 or 3 ints describing the output shape in (D)HW layout.
            antialias (bool): If set to true, an antialiasing is enabled for scaling down.
            roi (List[Tuple[int]]): Optional bounding boxes describing the input's region of interest.
                It should be a list of tuples. The list length must match the number
                of input tensors or be 1 (so that the same ROI is used for all samples).
                Each tuple must be of the form:
                * for 2D resampling: (lowH, lowW, highH, highW),
                * for 3D: (lowD, lowH, lowW, highD, highH, highW).
                If, for some axis, the low bound is bigger than the high bound,
                the tensor is flipped across the axis.
            interpolation (cvcuda.Interp): Interpolation type used. Used both for scaling down and up,
                cannot be specified together with (min_interpolation or mag_interpolation).
            min_interpolation (cvcuda.Interp): Interpolation type used for scaling down.
            mag_interpolation (cvcuda.Interp): Interpolation type used for scaling up.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.TensorBatch: The batch of resized tensors.

    )pbdoc");
    m.def("hq_resize_into", NvtxTrace("cvcuda.hq_resize_into", &TensorHQResizeInto), "dst"_a, "src"_a, py::kw_only(),
          "antialias"_a = false, "roi"_a = nullptr, "interpolation"_a = nullptr, "min_interpolation"_a = nullptr,
          "mag_interpolation"_a = nullptr, "stream"_a = nullptr, R"pbdoc(
        Executes the HQ Resize operation on the given cuda stream. The operator
        supports resampling for 2D (images) and 3D volumetric samples.


        Args:
            dst (cvcuda.Tensor): Output tensor. It's layout must match the src tensor.
                The size of D, H, and W extents may be different. The dst
                type must match the src's type or be float32.
            src (cvcuda.Tensor): Input tensor containing one or more images.
                The tensor layout must match: (N)(D)HW(C).
            antialias (bool): If set to true, an antialiasing is enabled for scaling down.
            roi (Tuple[int]): Optional bounding box describing the input's region of interest.
                For 2D resampling it should be (lowH, lowW, highH, highW),
                for 3D: (lowD, lowH, lowW, highD, highH, highW).
                If, for some axis, the low bound is bigger than the high bound,
                the image is flipped across the axis.
            interpolation (cvcuda.Interp): Interpolation type used. Used both for scaling down and up,
                cannot be specified together with (min_interpolation or mag_interpolation).
            min_interpolation (cvcuda.Interp): Interpolation type used for scaling down.
            mag_interpolation (cvcuda.Interp): Interpolation type used for scaling up.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");
    m.def("hq_resize_into", NvtxTrace("cvcuda.hq_resize_into", &VarShapeHQResizeInto), "dst"_a, "src"_a, py::kw_only(),
          "antialias"_a = false, "roi"_a = nullptr, "interpolation"_a = nullptr, "min_interpolation"_a = nullptr,
          "mag_interpolation"_a = nullptr, "stream"_a = nullptr, R"pbdoc(
        Executes the HQ Resize operation on the given cuda stream.


        Args:
            dst (cvcuda.ImageBatchVarShape): Output batch. The layout must match the input batch.
                The size of D, H, and W extents may be different. The dst
                type must match the src's type or be float32.
            src (cvcuda.ImageBatchVarShape): Input batch of images.
            antialias (bool): If set to true, an antialiasing is enabled for scaling down.
            roi (List[Tuple[int]]): Optional bounding boxes describing the input's region of interest.
                It should be a list of tuples. The list length must match the number
                of input tensors or be 1 (so that the same ROI is used for all samples).
                Each tuple must be of the form (lowH, lowW, highH, highW).
                If, for some axis, the low bound is bigger than the high bound,
                the image is flipped across the axis.
            interpolation (cvcuda.Interp): Interpolation type used. Used both for scaling down and up,
                cannot be specified together with (min_interpolation or mag_interpolation).
            min_interpolation (cvcuda.Interp): Interpolation type used for scaling down.
            mag_interpolation (cvcuda.Interp): Interpolation type used for scaling up.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The batch of resized images.

    )pbdoc");
    m.def("hq_resize_into", NvtxTrace("cvcuda.hq_resize_into", &TensorBatchHQResizeInto), "dst"_a, "src"_a,
          py::kw_only(), "antialias"_a = false, "roi"_a = nullptr, "interpolation"_a = nullptr,
          "min_interpolation"_a = nullptr, "mag_interpolation"_a = nullptr, "stream"_a = nullptr, R"pbdoc(
        Executes the HQ Resize operation on the given cuda stream. The operator
        supports resampling for 2D (images) and 3D volumetric samples.


        Args:
            dst (cvcuda.TensorBatch): Output batch. The layout must match the input batch.
                The size of D, H, and W extents may be different. The dst
                type must match the src's type or be float32.
            src (cvcuda.TensorBatch): Input batch containing one or more tensors of (D)HW(C) layout.
            antialias (bool): If set to true, an antialiasing is enabled for scaling down.
            roi (List[Tuple[int]]): Optional bounding boxes describing the input's region of interest.
                It should be a list of tuples. The list length must match the number
                of input tensors or be 1 (so that the same ROI is used for all samples).
                Each tuple must be of the form:
                * for 2D resampling: (lowH, lowW, highH, highW),
                * for 3D: (lowD, lowH, lowW, highD, highH, highW).
                If, for some axis, the low bound is bigger than the high bound,
                the tensor is flipped across the axis.
            interpolation (cvcuda.Interp): Interpolation type used. Used both for scaling down and up,
                cannot be specified together with (min_interpolation or mag_interpolation).
            min_interpolation (cvcuda.Interp): Interpolation type used for scaling down.
            mag_interpolation (cvcuda.Interp): Interpolation type used for scaling up.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.TensorBatch: The batch of resized tensors.

    )pbdoc");
}

} // namespace cvcudapy
