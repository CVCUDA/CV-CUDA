/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "WorkspaceCache.hpp"

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

namespace cvcudapy {

namespace {

using Roi  = pybind11::tuple;
using Rois = std::vector<Roi>;

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
    auto roiSize = roi.size();
    if (roiSize != static_cast<decltype(roiSize)>(2 * ndim))
    {
        if (ndim == 2)
        {
            throw std::runtime_error(
                "Got wrong number of ROI components. For image resize, 4 integers are expected: "
                "low_height, low_width, high_height, high_width describing the bounding box for "
                "the input.");
        }
        else
        {
            throw std::runtime_error(
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
        int32_t       size = m_rois.size();
        HQResizeRoiF *data = size == 0 ? nullptr : m_rois.data();
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

    char                 shapeArgLayout[4] = "DHW";
    HQResizeTensorShapeI tensorShape;
    for (int d = 0; d < resizeNDim; d++)
    {
        int axis = layout.find(shapeArgLayout[d + 3 - resizeNDim]);
        if (axis < 0)
        {
            throw std::runtime_error(
                "The layout of an input tensor to the resize operator must contain HW extents in the layout (for "
                "images) or DHW extents (for 3D resampling). Some extents are missing in the input tensor.");
        }
        tensorShape.extent[d] = shape[axis];
    }
    int channelAxis         = layout.find('C');
    tensorShape.numChannels = channelAxis < 0 ? 1 : shape[channelAxis];
    tensorShape.ndim        = resizeNDim;
    return tensorShape;
}

class BatchShapesHelper
{
public:
    BatchShapesHelper(const nvcv::ImageBatchVarShape &batch)
    {
        int32_t numSamples = batch.numImages();
        m_shapes.resize(numSamples);
        m_ndim        = 2;
        m_numChannels = batch.uniqueFormat().numChannels();
        for (int i = 0; i < numSamples; i++)
        {
            const auto &imgShape = batch[i].size();
            auto       &shape    = m_shapes[i];
            shape.extent[0]      = imgShape.h;
            shape.extent[1]      = imgShape.w;
        }
    }

    BatchShapesHelper(const TensorBatch &batch)
    {
        int32_t numSamples = batch.numTensors();
        auto    layout     = batch.layout();
        bool    hasDepth   = layout.find('D') >= 0;
        m_ndim             = hasDepth ? 3 : 2;
        m_numChannels      = -1;
        m_shapes.resize(numSamples);
        for (int i = 0; i < numSamples; i++)
        {
            const auto &tensor = batch[i];
            m_shapes[i]        = TensorShape(layout, tensor.shape(), m_ndim);
            if (i == 0)
            {
                m_numChannels = m_shapes[i].numChannels;
            }
            else if (m_numChannels != m_shapes[i].numChannels)
            {
                m_numChannels = -1;
            }
        }
    }

    HQResizeTensorShapesI NonOwningHandle()
    {
        int32_t size = m_shapes.size();
        return {size ? m_shapes.data() : nullptr, size, m_ndim, m_numChannels};
    }

private:
    int32_t                           m_ndim;
    int32_t                           m_numChannels;
    std::vector<HQResizeTensorShapeI> m_shapes;
};

inline Shape ResizedTensorShape(const nvcv::TensorLayout &srcLayout, const nvcv::TensorShape &srcShape,
                                const Shape &outShape)
{
    int resizeNDim = outShape.size();
    if (resizeNDim != 2 && resizeNDim != 3)
    {
        throw std::runtime_error(
            "The `out_shape` must be a tuple of 2 or 3 integers (for 2D or 3D resampling respectively).");
    }

    bool hasDepth     = srcLayout.find('D') >= 0;
    int  expectedNDim = hasDepth ? 3 : 2;

    if (expectedNDim != resizeNDim)
    {
        if (hasDepth)
        {
            throw std::runtime_error(
                "The input tensor contains depth extent (`D`) in the layout. For 3D resize, please specify the resized "
                "shape for 3 extents: depth, height, and width. Got 2 extents.");
        }
        else
        {
            throw std::runtime_error(
                "Expected the resized shape to consists of 2 integers: for resized height and width. Got 3 integers.");
        }
    }

    char shapeArgLayout[4] = "DHW";
    int  shapeArg[3];
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
            throw std::runtime_error(
                "The layout of an input tensor to the resize operator must contain HW extents in the layout (for "
                "images) or DHW extents (for 3D resampling). Some extents are missing in the input tensor.");
        }
        resizedShape[axis] = shapeArg[d];
    }
    return resizedShape;
}

class PyOpHQResize : public nvcvpy::Container
{
public:
    // Define a Key class to be used by the cache to fetch similar items for potential reuse.
    class Key : public nvcvpy::IKey
    {
    public:
        // the filters are generated by the operator constructor for a given device
        Key(int deviceId)
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
            const Key *thatKey = dynamic_cast<const Key *>(&that_);
            return thatKey != nullptr && thatKey->m_deviceId == m_deviceId;
        }

        int m_deviceId;
    };

    PyOpHQResize(int deviceId)
        : m_key(deviceId)
        , m_op()
    {
    }

    void submit(cudaStream_t stream, const Tensor &in, const Tensor &out, const NVCVInterpolationType minInterpolation,
                const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoiF *roi)
    {
        if (in.layout() != out.layout())
        {
            throw std::runtime_error("Input and output tensors must have the same layout");
        }

        int resizeNDim = in.layout().find('D') >= 0 ? 3 : 2;

        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(in.exportData());
        if (!inAccess)
        {
            throw std::runtime_error("Incompatible input tensor layout");
        }

        auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(out.exportData());
        if (!outAccess)
        {
            throw std::runtime_error("Incompatible input tensor layout");
        }

        int                  numSamples = inAccess->numSamples();
        HQResizeTensorShapeI inShape    = TensorShape(in.layout(), in.shape(), resizeNDim);
        HQResizeTensorShapeI outShape   = TensorShape(out.layout(), out.shape(), resizeNDim);

        auto req = m_op.getWorkspaceRequirements(numSamples, inShape, outShape, minInterpolation, magInterpolation,
                                                 antialias, roi);
        auto ws  = WorkspaceCache::instance().get(req, stream);
        m_op(stream, ws.get(), in, out, minInterpolation, magInterpolation, antialias, roi);
    }

    void submit(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation,
                bool antialias, const HQResizeRoisF rois)
    {
        BatchShapesHelper inShapes(in);
        BatchShapesHelper outShapes(out);
        auto              req
            = m_op.getWorkspaceRequirements(in.numImages(), inShapes.NonOwningHandle(), outShapes.NonOwningHandle(),
                                            minInterpolation, magInterpolation, antialias, rois);
        auto ws = WorkspaceCache::instance().get(req, stream);
        m_op(stream, ws.get(), in, out, minInterpolation, magInterpolation, antialias, rois);
    }

    void submit(cudaStream_t stream, const TensorBatch &in, const TensorBatch &out,
                const NVCVInterpolationType minInterpolation, const NVCVInterpolationType magInterpolation,
                bool antialias, const HQResizeRoisF rois)
    {
        if (in.layout() != out.layout())
        {
            throw std::runtime_error("Input and output batches must have the same layout");
        }
        BatchShapesHelper inShapes(in);
        BatchShapesHelper outShapes(out);
        auto              req
            = m_op.getWorkspaceRequirements(in.numTensors(), inShapes.NonOwningHandle(), outShapes.NonOwningHandle(),
                                            minInterpolation, magInterpolation, antialias, rois);
        auto ws = WorkspaceCache::instance().get(req, stream);
        m_op(stream, ws.get(), in, out, minInterpolation, magInterpolation, antialias, rois);
    }

    // Required override to get the py object container.
    py::object container() const override
    {
        return *this;
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
    Key              m_key;
    cvcuda::HQResize m_op;
};

template<typename Op, typename Src, typename Dst, typename Call>
auto RunGuard(Op &op, Src &src, Dst &dst, Stream &stream, Call &&call)
{
    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {src});
    guard.add(LockMode::LOCK_MODE_WRITE, {dst});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    call();
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

    NVCVInterpolationType minInterpolationArg, magInterpolationArg;
    GetMinMagInterpolation(minInterpolationArg, magInterpolationArg, interpolation, minInterpolation, magInterpolation);

    RunGuard(op, src, dst, stream,
             [&]()
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
    NVCVInterpolationType minInterpolationArg, magInterpolationArg;
    GetMinMagInterpolation(minInterpolationArg, magInterpolationArg, interpolation, minInterpolation, magInterpolation);

    RunGuard(op, src, dst, stream,
             [&]()
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
    ImageBatchVarShape out = ImageBatchVarShape::Create(src.capacity());

    int32_t numOutSizes = outShape.size();
    if (numOutSizes != src.numImages() && numOutSizes != 1)
    {
        throw std::runtime_error(
            "The list of output shapes `out_size` must either contain a single shape to be used for all output images "
            "or its length must match the number of input samples.");
    }

    for (int i = 0; i < src.numImages(); ++i)
    {
        auto size  = outShape[numOutSizes == 1 ? 0 : i];
        auto image = Image::Create({std::get<1>(size), std::get<0>(size)}, src[i].format());
        out.pushBack(image);
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

    NVCVInterpolationType minInterpolationArg, magInterpolationArg;
    GetMinMagInterpolation(minInterpolationArg, magInterpolationArg, interpolation, minInterpolation, magInterpolation);

    RunGuard(op, src, dst, stream,
             [&]()
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
    TensorBatch out = TensorBatch::Create(src.numTensors());

    int32_t numOutSizes = outShape.size();
    if (numOutSizes != src.numTensors() && numOutSizes != 1)
    {
        throw std::runtime_error(
            "The list of output shapes `out_size` must either contain a single shape to be used for all output tensors "
            "or its length must match the number of input tensors.");
    }

    for (int i = 0; i < src.numTensors(); ++i)
    {
        auto        sampleShape  = outShape[numOutSizes == 1 ? 0 : i];
        const auto &inSample     = src[i];
        auto        resizedShape = ResizedTensorShape(inSample.layout(), inSample.shape(), sampleShape);
        Tensor      dst          = Tensor::Create(resizedShape, src.dtype(), src.layout());
        out.pushBack(dst);
    }

    return TensorBatchHQResizeInto(out, src, antialias, roi, interpolation, minInterpolation, magInterpolation,
                                   pstream);
}

} // namespace

void ExportOpHQResize(py::module &m)
{
    using namespace pybind11::literals;

    m.def("hq_resize", &TensorHQResize, "src"_a, "out_size"_a, py::kw_only(), "antialias"_a = false, "roi"_a = nullptr,
          "interpolation"_a = nullptr, "min_interpolation"_a = nullptr, "mag_interpolation"_a = nullptr,
          "stream"_a = nullptr, R"pbdoc(
        Executes the HQ Resize operation on the given cuda stream. The operator
        supports resampling for 2D (images) and 3D volumetric samples.

        See also:
            Refer to the CV-CUDA C API reference for the HQ Resize operator
            for more details and usage examples.

        Args:
            src (Tensor): Input tensor containing one or more images.
                          The tensor layout must match: (N)(D)HW(C).
            out_size (Shape): Tuple of 2 or 3 ints describing the output shape in (D)HW layout.
            antialias (bool): If set to true, an antialiasing is enabled for scaling down.
            roi(Tuple): Optional bounding box describing the input's region of interest.
                        For 2D resampling it should be (lowH, lowW, highH, highW),
                        for 3D: (lowD, lowH, lowW, highD, highH, highW).
                        If, for some axis, the low bound is bigger than the high bound,
                        the image is flipped across the axis.
            interpolation(Interp): Interpolation type used. Used both for scaling down and up,
                                   cannot be specified together with (min_interpolation or mag_interpolation).
            min_interpolation(Interp): Interpolation type used for scaling down.
            mag_interpolation(Interp): Interpolation type used for scaling up.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
    m.def("hq_resize", &VarShapeHQResize, "src"_a, "out_size"_a, py::kw_only(), "antialias"_a = false,
          "roi"_a = nullptr, "interpolation"_a = nullptr, "min_interpolation"_a = nullptr,
          "mag_interpolation"_a = nullptr, "stream"_a = nullptr, R"pbdoc(
        Executes the HQ Resize operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the HQ Resize operator
            for more details and usage examples.

        Args:
            src (ImageBatchVarShape): Input batch of images.
            out_size (Shape): Tuple of 2 ints describing the output shape in HW layout.
            antialias (bool): If set to true, an antialiasing is enabled for scaling down.
            roi(List[Tuple]): Optional bounding boxes describing the input's region of interest.
                              It should be a list of tuples. The list length must match the number
                              of input tensors or be 1 (so that the same ROI is used for all samples).
                              Each tuple must be of the form (lowH, lowW, highH, highW).
                              If, for some axis, the low bound is bigger than the high bound,
                              the image is flipped across the axis.
            interpolation(Interp): Interpolation type used. Used both for scaling down and up,
                                   cannot be specified together with (min_interpolation or mag_interpolation).
            min_interpolation(Interp): Interpolation type used for scaling down.
            mag_interpolation(Interp): Interpolation type used for scaling up.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The batch of resized images.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
    m.def("hq_resize", &TensorBatchHQResize, "src"_a, "out_size"_a, py::kw_only(), "antialias"_a = false,
          "roi"_a = nullptr, "interpolation"_a = nullptr, "min_interpolation"_a = nullptr,
          "mag_interpolation"_a = nullptr, "stream"_a = nullptr, R"pbdoc(
        Executes the HQ Resize operation on the given cuda stream. The operator
        supports resampling for 2D (images) and 3D volumetric samples.

        See also:
            Refer to the CV-CUDA C API reference for the HQ Resize operator
            for more details and usage examples.

        Args:
            src (TensorBatch): Input batch containing one or more tensors of (D)HW(C) layout.
            out_size (Shape): Tuple of 2 or 3 ints describing the output shape in (D)HW layout.
            antialias (bool): If set to true, an antialiasing is enabled for scaling down.
            roi(List[Tuple]): Optional bounding boxes describing the input's region of interest.
                              It should be a list of tuples. The list length must match the number
                              of input tensors or be 1 (so that the same ROI is used for all samples).
                              Each tuple must be of the form:
                                  * for 2D resampling: (lowH, lowW, highH, highW),
                                  * for 3D: (lowD, lowH, lowW, highD, highH, highW).
                              If, for some axis, the low bound is bigger than the high bound,
                              the tensor is flipped across the axis.
            interpolation(Interp): Interpolation type used. Used both for scaling down and up,
                                   cannot be specified together with (min_interpolation or mag_interpolation).
            min_interpolation(Interp): Interpolation type used for scaling down.
            mag_interpolation(Interp): Interpolation type used for scaling up.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.TensorBatch: The batch of resized tensors.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
    m.def("hq_resize_into", &TensorHQResizeInto, "dst"_a, "src"_a, py::kw_only(), "antialias"_a = false,
          "roi"_a = nullptr, "interpolation"_a = nullptr, "min_interpolation"_a = nullptr,
          "mag_interpolation"_a = nullptr, "stream"_a = nullptr, R"pbdoc(
        Executes the HQ Resize operation on the given cuda stream. The operator
        supports resampling for 2D (images) and 3D volumetric samples.

        See also:
            Refer to the CV-CUDA C API reference for the HQ Resize operator
            for more details and usage examples.

        Args:
            dst (Tensor): Output tensor. It's layout must match the src tensor.
                          The size of D, H, and W extents may be different. The dst
                          type must match the src's type or be float32.
            src (Tensor): Input tensor containing one or more images.
                          The tensor layout must match: (N)(D)HW(C).
            antialias (bool): If set to true, an antialiasing is enabled for scaling down.
            roi(Tuple): Optional bounding box describing the input's region of interest.
                        For 2D resampling it should be (lowH, lowW, highH, highW),
                        for 3D: (lowD, lowH, lowW, highD, highH, highW).
                        If, for some axis, the low bound is bigger than the high bound,
                        the image is flipped across the axis.
            interpolation(Interp): Interpolation type used. Used both for scaling down and up,
                                   cannot be specified together with (min_interpolation or mag_interpolation).
            min_interpolation(Interp): Interpolation type used for scaling down.
            mag_interpolation(Interp): Interpolation type used for scaling up.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
    m.def("hq_resize_into", &VarShapeHQResizeInto, "dst"_a, "src"_a, py::kw_only(), "antialias"_a = false,
          "roi"_a = nullptr, "interpolation"_a = nullptr, "min_interpolation"_a = nullptr,
          "mag_interpolation"_a = nullptr, "stream"_a = nullptr, R"pbdoc(
        Executes the HQ Resize operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the HQ Resize operator
            for more details and usage examples.

        Args:
            dst (ImageBatchVarShape): Output batch. The layout must match the input batch.
                                      The size of D, H, and W extents may be different. The dst
                                      type must match the src's type or be float32.
            src (ImageBatchVarShape): Input batch of images.
            antialias (bool): If set to true, an antialiasing is enabled for scaling down.
            roi(List[Tuple]): Optional bounding boxes describing the input's region of interest.
                              It should be a list of tuples. The list length must match the number
                              of input tensors or be 1 (so that the same ROI is used for all samples).
                              Each tuple must be of the form (lowH, lowW, highH, highW).
                              If, for some axis, the low bound is bigger than the high bound,
                              the image is flipped across the axis.
            interpolation(Interp): Interpolation type used. Used both for scaling down and up,
                                   cannot be specified together with (min_interpolation or mag_interpolation).
            min_interpolation(Interp): Interpolation type used for scaling down.
            mag_interpolation(Interp): Interpolation type used for scaling up.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.ImageBatchVarShape: The batch of resized images.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
    m.def("hq_resize_into", &TensorBatchHQResizeInto, "dst"_a, "src"_a, py::kw_only(), "antialias"_a = false,
          "roi"_a = nullptr, "interpolation"_a = nullptr, "min_interpolation"_a = nullptr,
          "mag_interpolation"_a = nullptr, "stream"_a = nullptr, R"pbdoc(
        Executes the HQ Resize operation on the given cuda stream. The operator
        supports resampling for 2D (images) and 3D volumetric samples.

        See also:
            Refer to the CV-CUDA C API reference for the HQ Resize operator
            for more details and usage examples.

        Args:
            dst (TensorBatch): Output batch. The layout must match the input batch.
                               The size of D, H, and W extents may be different. The dst
                               type must match the src's type or be float32.
            src (TensorBatch): Input batch containing one or more tensors of (D)HW(C) layout.
            antialias (bool): If set to true, an antialiasing is enabled for scaling down.
            roi(List[Tuple]): Optional bounding boxes describing the input's region of interest.
                              It should be a list of tuples. The list length must match the number
                              of input tensors or be 1 (so that the same ROI is used for all samples).
                              Each tuple must be of the form:
                                  * for 2D resampling: (lowH, lowW, highH, highW),
                                  * for 3D: (lowD, lowH, lowW, highD, highH, highW).
                              If, for some axis, the low bound is bigger than the high bound,
                              the tensor is flipped across the axis.
            interpolation(Interp): Interpolation type used. Used both for scaling down and up,
                                   cannot be specified together with (min_interpolation or mag_interpolation).
            min_interpolation(Interp): Interpolation type used for scaling down.
            mag_interpolation(Interp): Interpolation type used for scaling up.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.TensorBatch: The batch of resized tensors.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
