/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Image.hpp"

#include "Cache.hpp"
#include "DataType.hpp"
#include "ImageFormat.hpp"
#include "Stream.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/TensorShapeInfo.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace nvcvpy::priv {

bool Image::Key::doIsEqual(const IKey &ithat) const
{
    auto &that = static_cast<const Key &>(ithat);

    // Wrapper key's all compare equal, are they can't be used
    // and whenever we query the cache for wrappers, we really
    // want to get them all (as long as they aren't being used).
    if (m_wrapper && that.m_wrapper)
    {
        return true;
    }
    else if (m_wrapper || that.m_wrapper) // xor
    {
        return false;
    }
    else
    {
        return std::tie(m_size, m_format) == std::tie(that.m_size, that.m_format);
    }
}

size_t Image::Key::doGetHash() const
{
    if (m_wrapper)
    {
        return 0; // all wrappers are equal wrt. the cache
    }
    else
    {
        using util::ComputeHash;
        return ComputeHash(m_size, m_format);
    }
}

namespace {

struct BufferImageInfo
{
    int            numPlanes;
    nvcv::Size2D   size;
    int            numChannels;
    bool           isChannelLast;
    int64_t        planeStride, rowStride;
    nvcv::DataType dtype;
    void          *data;
};

std::vector<BufferImageInfo> ExtractBufferImageInfo(const std::vector<py::buffer_info> &buffers,
                                                    const nvcv::ImageFormat            &fmt)
{
    std::vector<BufferImageInfo> bufferInfoList;

    int curChannel = 0;

    // For each buffer,
    for (size_t p = 0; p < buffers.size(); ++p)
    {
        const py::buffer_info &info = buffers[p];

        // Extract 4d shape and layout regardless of rank
        ssize_t            shape[4];
        ssize_t            strides[4];
        nvcv::TensorLayout layout;

        switch (info.ndim)
        {
        case 1:
            layout = nvcv::TensorLayout::NCHW;

            shape[0] = 1;
            shape[1] = 1;
            shape[2] = 1;
            shape[3] = info.shape[0];

            strides[0] = info.strides[0];
            strides[1] = info.strides[0];
            strides[2] = info.strides[0];
            strides[3] = info.strides[0];
            break;

        case 2:
            layout = nvcv::TensorLayout::NCHW;

            shape[0] = 1;
            shape[1] = 1;
            shape[2] = info.shape[0];
            shape[3] = info.shape[1];

            strides[0] = info.shape[0] * info.strides[0];
            strides[1] = strides[0];
            strides[2] = info.strides[0];
            strides[3] = info.strides[1];
            break;

        case 3:
        case 4:
            shape[0] = info.ndim == 3 ? 1 : info.shape[info.ndim - 4];
            shape[1] = info.shape[info.ndim - 3];
            shape[2] = info.shape[info.ndim - 2];
            shape[3] = info.shape[info.ndim - 1];

            // User has specified a format?
            if (fmt != nvcv::FMT_NONE)
            {
                // Use it to disambiguate
                if (fmt.planeNumChannels(p) == shape[3])
                {
                    layout = nvcv::TensorLayout::NHWC;
                }
                else
                {
                    layout = nvcv::TensorLayout::NCHW;
                }
            }
            else
            {
                // Or else,
                if (shape[3] <= 4) // (C<=4)
                {
                    layout = nvcv::TensorLayout::NHWC;
                }
                else
                {
                    layout = nvcv::TensorLayout::NCHW;
                }
            }

            strides[1] = info.strides[info.ndim - 3];
            strides[2] = info.strides[info.ndim - 2];
            strides[3] = info.strides[info.ndim - 1];

            if (info.ndim == 3)
            {
                strides[0] = shape[1] * strides[1];
            }
            else
            {
                strides[0] = info.strides[info.ndim - 4];
            }
            break;

        default:
            throw std::invalid_argument(
                util::FormatString("Number of buffer dimensions must be between 1 and 4, not %ld", info.ndim));
        }

        // Validate strides -----------------------

        if (strides[0] <= 0 || strides[1] <= 0 || strides[2] <= 0)
        {
            throw std::invalid_argument("Buffer strides must be all >= 1");
        }

        NVCV_ASSERT(layout.rank() == 4);

        auto infoShape = nvcv::TensorShapeInfoImagePlanar::Create(nvcv::TensorShape(shape, 4, layout));
        NVCV_ASSERT(infoShape);

        const auto *infoLayout = &infoShape->infoLayout();

        if (strides[3] != info.itemsize)
        {
            throw std::invalid_argument(util::FormatString(
                "Fastest changing dimension must be packed, i.e., have stride equal to %ld byte(s), not %ld",
                info.itemsize, strides[2]));
        }

        ssize_t packedRowStride = info.itemsize * infoShape->numCols();
        ssize_t rowStride       = strides[infoLayout->idxHeight()];
        if (!infoLayout->isChannelLast() && rowStride != packedRowStride)
        {
            throw std::invalid_argument(util::FormatString(
                "Image row must packed, i.e., have stride equal to %ld byte(s), not %ld", packedRowStride, rowStride));
        }

        bufferInfoList.emplace_back();

        BufferImageInfo &bufInfo = bufferInfoList.back();
        bufInfo.isChannelLast    = infoLayout->isChannelLast();
        bufInfo.numPlanes        = bufInfo.isChannelLast ? infoShape->numSamples() : infoShape->numChannels();
        bufInfo.numChannels      = infoShape->numChannels();
        bufInfo.size             = infoShape->size();
        bufInfo.planeStride      = strides[infoLayout->idxSample()];
        bufInfo.rowStride        = strides[infoLayout->idxHeight()];
        bufInfo.data             = info.ptr;
        bufInfo.dtype            = py::cast<nvcv::DataType>(util::ToDType(info));

        curChannel += bufInfo.numPlanes * bufInfo.numChannels;
        if (curChannel > 4)
        {
            throw std::invalid_argument("Number of channels specified in a buffers must be <= 4");
        }

        NVCV_ASSERT(bufInfo.numPlanes <= 4);
        NVCV_ASSERT(bufInfo.numChannels <= 4);
    }

    return bufferInfoList;
}

nvcv::DataType MakePackedType(nvcv::DataType dtype, int numChannels)
{
    if (dtype.numChannels() == numChannels)
    {
        return dtype;
    }
    else if (dtype.numChannels() == 1)
    {
        NVCV_ASSERT(2 <= numChannels && numChannels <= 4);

        nvcv::PackingParams pp = GetParams(dtype.packing());

        switch (numChannels)
        {
        case 2:
            pp.swizzle = nvcv::Swizzle::S_XY00;
            break;
        case 3:
            pp.swizzle = nvcv::Swizzle::S_XYZ0;
            break;
        case 4:
            pp.swizzle = nvcv::Swizzle::S_XYZW;
            break;
        }
        pp.byteOrder = nvcv::ByteOrder::MSB;
        for (int i = 1; i < numChannels; ++i)
        {
            pp.bits[i] = pp.bits[0];
        }

        nvcv::Packing newPacking = MakePacking(pp);
        return nvcv::DataType{dtype.dataKind(), newPacking};
    }
    else
    {
        // in case of complex numbers, the number of channels == 1 but pix has 2 channels.
        return dtype;
    }
}

nvcv::ImageFormat InferImageFormat(const std::vector<nvcv::DataType> &planePixTypes)
{
    if (planePixTypes.empty())
    {
        return nvcv::FMT_NONE;
    }

    static_assert(NVCV_PACKING_0 == 0, "Invalid 0 packing value");
    NVCV_ASSERT(planePixTypes.size() <= 4);

    nvcv::Packing packing[4] = {nvcv::Packing::NONE};

    int numChannels = 0;

    for (size_t p = 0; p < planePixTypes.size(); ++p)
    {
        packing[p] = planePixTypes[p].packing();
        numChannels += planePixTypes[p].numChannels();

        if (planePixTypes[p].dataKind() != planePixTypes[0].dataKind())
        {
            throw std::invalid_argument("Planes must all have the same data type");
        }
    }

    nvcv::DataKind dataKind = planePixTypes[0].dataKind();

    int numPlanes = planePixTypes.size();

    // Planar or packed?
    if (numPlanes == 1 || numChannels == numPlanes)
    {
        nvcv::ImageFormat baseFormatList[4] = {nvcv::FMT_U8, nvcv::FMT_2F32, nvcv::FMT_RGB8, nvcv::FMT_RGBA8};

        NVCV_ASSERT(numChannels <= 4);
        nvcv::ImageFormat baseFormat = baseFormatList[numChannels - 1];

        nvcv::ColorModel model = baseFormat.colorModel();
        switch (model)
        {
        case nvcv::ColorModel::YCbCr:
            return nvcv::ImageFormat(baseFormat.colorSpec(), baseFormat.chromaSubsampling(), baseFormat.memLayout(),
                                     dataKind, baseFormat.swizzle(), packing[0], packing[1], packing[2], packing[3]);

        case nvcv::ColorModel::UNDEFINED:
            return nvcv::ImageFormat(baseFormat.memLayout(), dataKind, baseFormat.swizzle(), packing[0], packing[1],
                                     packing[2], packing[3]);
        case nvcv::ColorModel::RAW:
            return nvcv::ImageFormat(baseFormat.rawPattern(), baseFormat.memLayout(), dataKind, baseFormat.swizzle(),
                                     packing[0], packing[1], packing[2], packing[3]);
        default:
            return nvcv::ImageFormat(model, baseFormat.colorSpec(), baseFormat.memLayout(), dataKind,
                                     baseFormat.swizzle(), packing[0], packing[1], packing[2], packing[3]);
        }
    }
    // semi-planar, NV12-like?
    // TODO: this test is too fragile, must improve
    else if (numPlanes == 2 && numChannels == 3)
    {
        return nvcv::FMT_NV12_ER.dataKind(dataKind).swizzleAndPacking(nvcv::Swizzle::S_XYZ0, packing[0], packing[1],
                                                                      packing[2], packing[3]);
    }
    // Or else, we'll consider it as representing a non-color format
    else
    {
        // clang-format off
        nvcv::Swizzle sw = MakeSwizzle(numChannels >= 1 ? nvcv::Channel::X : nvcv::Channel::NONE,
                                     numChannels >= 2 ? nvcv::Channel::Y : nvcv::Channel::NONE,
                                     numChannels >= 3 ? nvcv::Channel::Z : nvcv::Channel::NONE,
                                     numChannels >= 4 ? nvcv::Channel::W : nvcv::Channel::NONE);
        // clang-format on

        return nvcv::FMT_U8.dataKind(dataKind).swizzleAndPacking(sw, packing[0], packing[1], packing[2], packing[3]);
    }
}

void FillNVCVImageBufferStrided(NVCVImageData &imgData, const std::vector<py::buffer_info> &infos,
                                nvcv::ImageFormat fmt)
{
    // If user passes an image format, we must check if the given buffers are consistent with it.
    // Otherwise, we need to infer the image format from the given buffers.

    // Here's the plan:
    // 1. Loop through all buffers and infer its dimensions, number of channels and data type.
    //    In case of ambiguity in inferring data type for a buffer,
    //    - If available, use given image format for disambiguation
    //    - Otherwise, if number of channels in last dimension is <= 4, treat it as packed, or else it's planar
    // 2. Validate the data collected to see if it represents a real image format
    // 3. If available, compare the given image format with the inferred one, they're data layout must be the same.

    // Let the games begin.

    NVCVImageBufferStrided &dataStrided = imgData.buffer.strided;

    dataStrided = {}; // start anew

    std::vector<BufferImageInfo> bufferInfoList = ExtractBufferImageInfo(infos, fmt);
    std::vector<nvcv::DataType>  planeDataTypes;

    int curPlane = 0;
    for (const BufferImageInfo &b : bufferInfoList)
    {
        for (int p = 0; p < b.numPlanes; ++p, ++curPlane)
        {
            NVCV_ASSERT(curPlane <= 4);

            dataStrided.planes[curPlane].width     = b.size.w;
            dataStrided.planes[curPlane].height    = b.size.h;
            dataStrided.planes[curPlane].rowStride = b.rowStride;
            dataStrided.planes[curPlane].basePtr   = reinterpret_cast<NVCVByte *>(b.data) + b.planeStride * p;

            planeDataTypes.push_back(MakePackedType(b.dtype, b.isChannelLast ? b.numChannels : 1));
        }
    }
    dataStrided.numPlanes = curPlane;

    if (dataStrided.numPlanes == 0)
    {
        throw std::invalid_argument("Number of planes must be >= 1");
    }

    nvcv::ImageFormat inferredFormat = InferImageFormat(planeDataTypes);

    nvcv::ImageFormat finalFormat;

    // User explicitely specifies the image format?
    if (fmt != nvcv::FMT_NONE)
    {
        if (!HasSameDataLayout(fmt, inferredFormat))
        {
            throw std::invalid_argument(
                util::FormatString("Format inferred from buffers %s isn't compatible with given image format %s",
                                   util::ToString(inferredFormat).c_str(), util::ToString(fmt).c_str()));
        }
        finalFormat = fmt;
    }
    else
    {
        finalFormat = inferredFormat;
    }
    imgData.format = finalFormat;

    nvcv::Size2D imgSize = {dataStrided.planes[0].width, dataStrided.planes[0].height};

    // Now do a final check on the expected plane sizes according to the
    // format
    for (int p = 0; p < dataStrided.numPlanes; ++p)
    {
        nvcv::Size2D goldSize = finalFormat.planeSize(imgSize, p);
        nvcv::Size2D plSize{dataStrided.planes[p].width, dataStrided.planes[p].height};

        if (plSize.w != goldSize.w || plSize.h != goldSize.h)
        {
            throw std::invalid_argument(util::FormatString(
                "Plane %d's size %dx%d doesn't correspond to what's expected by %s format %s of image with size %dx%d",
                p, plSize.w, plSize.h, (fmt == nvcv::FMT_NONE ? "inferred" : "given"),
                util::ToString(finalFormat).c_str(), imgSize.w, imgSize.h));
        }
    }
}

nvcv::ImageDataStridedCuda CreateNVCVImageDataDevice(const std::vector<py::buffer_info> &infos, nvcv::ImageFormat fmt)
{
    NVCVImageData imgData;
    FillNVCVImageBufferStrided(imgData, infos, fmt);

    return nvcv::ImageDataStridedCuda(nvcv::ImageFormat{imgData.format}, imgData.buffer.strided);
}

nvcv::ImageDataStridedHost CreateNVCVImageDataHost(const std::vector<py::buffer_info> &infos, nvcv::ImageFormat fmt)
{
    NVCVImageData imgData;
    FillNVCVImageBufferStrided(imgData, infos, fmt);

    return nvcv::ImageDataStridedHost(nvcv::ImageFormat{imgData.format}, imgData.buffer.strided);
}

} // namespace

Image::Image(const Size2D &size, nvcv::ImageFormat fmt)
    : m_impl(std::make_unique<nvcv::Image>(nvcv::Size2D{std::get<0>(size), std::get<1>(size)}, fmt))
    , m_key{size, fmt}
{
}

Image::Image(std::vector<std::shared_ptr<CudaBuffer>> bufs, const nvcv::IImageDataStridedCuda &imgData)
    : m_key{} // it's a wrap!
{
    if (bufs.size() == 1)
    {
        m_wrapped = py::cast(bufs[0]);
    }
    else
    {
        NVCV_ASSERT(bufs.size() >= 2);
        m_wrapped = py::cast(std::move(bufs));
    }

    m_impl = std::make_unique<nvcv::ImageWrapData>(imgData);
}

Image::Image(std::vector<py::buffer> bufs, const nvcv::IImageDataStridedHost &hostData)
{
    // Input buffer is host data.
    // We'll create a regular image and copy the host data into it.

    // Create the image with same size and format as host data
    m_impl = std::make_unique<nvcv::Image>(hostData.size(), hostData.format());

    auto *devData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(m_impl->exportData());
    NVCV_ASSERT(devData != nullptr);
    NVCV_ASSERT(hostData.format() == devData->format());
    NVCV_ASSERT(hostData.numPlanes() == devData->numPlanes());

    // Now copy each plane from host to device
    for (int p = 0; p < devData->numPlanes(); ++p)
    {
        const nvcv::ImagePlaneStrided &devPlane  = devData->plane(p);
        const nvcv::ImagePlaneStrided &hostPlane = hostData.plane(p);

        NVCV_ASSERT(devPlane.width == hostPlane.width);
        NVCV_ASSERT(devPlane.height == hostPlane.height);

        util::CheckThrow(cudaMemcpy2D(devPlane.basePtr, devPlane.rowStride, hostPlane.basePtr, hostPlane.rowStride,
                                      hostPlane.width * hostData.format().planePixelStrideBytes(p), hostPlane.height,
                                      cudaMemcpyHostToDevice));
    }

    m_key = Key{
        {m_impl->size().w, m_impl->size().h},
        m_impl->format()
    };
}

std::shared_ptr<Image> Image::shared_from_this()
{
    return std::static_pointer_cast<Image>(Container::shared_from_this());
}

std::shared_ptr<const Image> Image::shared_from_this() const
{
    return std::static_pointer_cast<const Image>(Container::shared_from_this());
}

std::shared_ptr<Image> Image::Create(const Size2D &size, nvcv::ImageFormat fmt)
{
    std::vector<std::shared_ptr<CacheItem>> vcont = Cache::Instance().fetch(Key{size, fmt});

    // None found?
    if (vcont.empty())
    {
        std::shared_ptr<Image> img(new Image(size, fmt));
        Cache::Instance().add(*img);
        return img;
    }
    else
    {
        // Get the first one
        return std::static_pointer_cast<Image>(vcont[0]);
    }
}

std::shared_ptr<Image> Image::Zeros(const Size2D &size, nvcv::ImageFormat fmt)
{
    auto img = Image::Create(size, fmt);

    auto *data = dynamic_cast<const nvcv::IImageDataStridedCuda *>(img->impl().exportData());
    NVCV_ASSERT(data);

    for (int p = 0; p < data->numPlanes(); ++p)
    {
        const nvcv::ImagePlaneStrided &plane = data->plane(p);

        util::CheckThrow(cudaMemset2D(plane.basePtr, plane.rowStride, 0,
                                      plane.width * data->format().planePixelStrideBytes(p), plane.height));
    }

    return img;
}

std::shared_ptr<Image> Image::WrapCuda(CudaBuffer &buffer, nvcv::ImageFormat fmt)
{
    return WrapCudaVector(std::vector{buffer.shared_from_this()}, fmt);
}

std::shared_ptr<Image> Image::WrapCudaVector(std::vector<std::shared_ptr<CudaBuffer>> buffers, nvcv::ImageFormat fmt)
{
    std::vector<py::buffer_info> bufinfos;
    for (size_t i = 0; i < buffers.size(); ++i)
    {
        bufinfos.emplace_back(buffers[i]->request());
    }

    nvcv::ImageDataStridedCuda imgData = CreateNVCVImageDataDevice(std::move(bufinfos), fmt);

    // This is the key of an image wrapper.
    // All image wrappers have the same key.
    Image::Key key;
    // We take this opportunity to remove from cache all wrappers that aren't
    // being used. They aren't reusable anyway.
    Cache::Instance().removeAllNotInUseMatching(key);

    // Need to add wrappers to cache so that they don't get destroyed by
    // the cuda stream when they're last used, and python script isn't
    // holding a reference to them. If we don't do it, things might break.
    std::shared_ptr<Image> img(new Image(std::move(buffers), imgData));
    Cache::Instance().add(*img);
    return img;
}

std::shared_ptr<Image> Image::CreateHost(py::buffer buffer, nvcv::ImageFormat fmt)
{
    return CreateHostVector(std::vector{buffer}, fmt);
}

std::shared_ptr<Image> Image::CreateHostVector(std::vector<py::buffer> buffers, nvcv::ImageFormat fmt)
{
    std::vector<py::buffer_info> bufinfos;
    for (size_t i = 0; i < buffers.size(); ++i)
    {
        bufinfos.emplace_back(buffers[i].request());
    }

    nvcv::ImageDataStridedHost imgData = CreateNVCVImageDataHost(std::move(bufinfos), fmt);

    // We take this opportunity to remove all wrappers from cache.
    // They aren't reusable anyway.
    Image::Key key;
    Cache::Instance().removeAllNotInUseMatching(key);

    std::shared_ptr<Image> img(new Image(std::move(buffers), imgData));
    Cache::Instance().add(*img);
    return img;
}

Size2D Image::size() const
{
    nvcv::Size2D s = m_impl->size();
    return {s.w, s.h};
}

int32_t Image::width() const
{
    return m_impl->size().w;
}

int32_t Image::height() const
{
    return m_impl->size().h;
}

nvcv::ImageFormat Image::format() const
{
    return m_impl->format();
}

std::ostream &operator<<(std::ostream &out, const Image &img)
{
    return out << "<nvcv.Image " << img.impl().size() << ' ' << img.impl().format() << '>';
}

namespace {

std::vector<std::pair<py::buffer_info, nvcv::TensorLayout>> ToPyBufferInfo(const nvcv::IImageDataStrided    &imgData,
                                                                           std::optional<nvcv::TensorLayout> userLayout)
{
    if (imgData.numPlanes() < 1)
    {
        return {};
    }

    const nvcv::ImagePlaneStrided &firstPlane = imgData.plane(0);

    std::optional<nvcv::TensorLayoutInfoImage> infoLayout;
    if (userLayout)
    {
        if (auto tmp = nvcv::TensorLayoutInfoImage::Create(*userLayout))
        {
            infoLayout.emplace(std::move(*tmp));
        }
        else
        {
            throw std::runtime_error("Layout can't represent the planar images needed");
        }
    }

    bool singleBuffer = true;

    // Let's check if we can return only one buffer, depending
    // on the planes dimensions, pitch and data type.
    for (int p = 1; p < imgData.numPlanes(); ++p)
    {
        const nvcv::ImagePlaneStrided &plane = imgData.plane(p);

        if (plane.width != firstPlane.width || plane.height != firstPlane.height
            || plane.rowStride != firstPlane.rowStride || imgData.format().planeDataType(0).numChannels() >= 2
            || imgData.format().planeDataType(0) != imgData.format().planeDataType(p))
        {
            singleBuffer = false;
            break;
        }

        // check if using the same plane pitch
        if (p >= 2)
        {
            intptr_t goldPlaneStrided = imgData.plane(1).basePtr - imgData.plane(0).basePtr;
            intptr_t curPlaneStrided  = imgData.plane(p).basePtr - imgData.plane(p - 1).basePtr;
            if (curPlaneStrided != goldPlaneStrided)
            {
                singleBuffer = false;
                break;
            }
        }
    }

    std::vector<std::pair<py::buffer_info, nvcv::TensorLayout>> out;

    // If not using a single buffer, we'll forcibly use one buffer per plane.
    int numBuffers = singleBuffer ? 1 : imgData.numPlanes();

    for (int p = 0; p < numBuffers; ++p)
    {
        int planeWidth       = imgData.plane(p).width;
        int planeHeight      = imgData.plane(p).height;
        int planeNumChannels = imgData.format().planeNumChannels(p);
        // bytes per pixel in the plane
        int planeBPP = imgData.format().planeDataType(p).strideBytes();

        switch (imgData.format().planePacking(p))
        {
        // These (YUYV, UYVY, ...) need some special treatment.
        // Although it's 3 channels in the plane, it's actually
        // two channels per pixel.
        case nvcv::Packing::X8_Y8__X8_Z8:
        case nvcv::Packing::Y8_X8__Z8_X8:
            planeNumChannels = 2;
            break;
        default:
            break;
        }

        // Infer the layout and shape of this buffer
        std::vector<ssize_t> inferredShape;
        std::vector<ssize_t> inferredStrides;
        nvcv::TensorLayout   inferredLayout;

        py::dtype inferredDType;

        if (numBuffers == 1)
        {
            if (imgData.format().numChannels() == 1)
            {
                NVCV_ASSERT(imgData.numPlanes() == 1);
                inferredShape   = {planeHeight, planeWidth};
                inferredStrides = {imgData.plane(p).rowStride, planeBPP};
                inferredLayout  = nvcv::TensorLayout{"HW"};
                inferredDType   = py::cast(imgData.format().planeDataType(p));
            }
            else if (imgData.numPlanes() == 1)
            {
                NVCV_ASSERT(planeNumChannels >= 2);
                inferredShape   = {planeHeight, planeWidth, planeNumChannels};
                inferredStrides = {imgData.plane(p).rowStride, planeBPP, planeBPP / planeNumChannels};
                inferredLayout  = nvcv::TensorLayout{"HWC"};
                inferredDType   = py::cast(imgData.format().planeDataType(p).channelType(0));
            }
            else
            {
                NVCV_ASSERT(planeNumChannels == 1);

                intptr_t planeStride = imgData.plane(1).basePtr - imgData.plane(0).basePtr;
                NVCV_ASSERT(planeStride > 0);

                inferredShape   = {imgData.numPlanes(), planeHeight, planeWidth};
                inferredStrides = {planeStride, imgData.plane(p).rowStride, planeBPP};
                inferredLayout  = nvcv::TensorLayout{"CHW"};
                inferredDType   = py::cast(imgData.format().planeDataType(p));
            }
        }
        else
        {
            NVCV_ASSERT(imgData.numPlanes() >= 2);
            NVCV_ASSERT(imgData.numPlanes() == numBuffers);

            inferredShape = {planeHeight, planeWidth, planeNumChannels};
            inferredStrides
                = {(int64_t)imgData.plane(p).rowStride, (int64_t)planeBPP, (int64_t)planeBPP / planeNumChannels};
            inferredLayout = nvcv::TensorLayout{"HWC"};
            inferredDType  = py::cast(imgData.format().planeDataType(p).channelType(0));
        }

        NVCV_ASSERT((ssize_t)inferredShape.size() == inferredLayout.rank());
        NVCV_ASSERT((ssize_t)inferredStrides.size() == inferredLayout.rank());

        std::vector<ssize_t> shape;
        std::vector<ssize_t> strides;
        nvcv::TensorLayout   layout;

        // Do we have to use the layout user has specified?
        if (userLayout)
        {
            layout = *userLayout;

            // Check if user layout has all required dimensions
            for (int i = 0; i < inferredLayout.rank(); ++i)
            {
                if (inferredShape[i] >= 2 && userLayout->find(inferredLayout[i]) < 0)
                {
                    throw std::runtime_error(util::FormatString("Layout need dimension '%c'", inferredLayout[i]));
                }
            }

            int idxLastInferDim = -1;

            // Fill up the final shape and strides according to the user layout
            for (int i = 0; i < userLayout->rank(); ++i)
            {
                int idxInferDim = inferredLayout.find((*userLayout)[i]);

                if (idxInferDim < 0)
                {
                    shape.push_back(1);
                    // TODO: must do better than this
                    strides.push_back(0);
                }
                else
                {
                    // The order of channels must be the same, despite of
                    // user layout having some other channels in the layout
                    // in between the channels in inferredLayout.
                    if (idxLastInferDim >= idxInferDim)
                    {
                        throw std::runtime_error("Layout not compatible with image to be exported");
                    }
                    idxLastInferDim = idxInferDim;

                    shape.push_back(inferredShape[idxInferDim]);
                    strides.push_back(inferredStrides[idxInferDim]);
                }
            }
        }
        else
        {
            layout  = inferredLayout;
            shape   = inferredShape;
            strides = inferredStrides;
        }

        // There's no direct way to construct a py::buffer_info from data together with a py::dtype.
        // To do that, we first construct a py::array (it accepts py::dtype), and use ".request()"
        // to retrieve the corresponding py::buffer_info.
        // To avoid spurious data copies in py::array ctor, we create this dummy owner.
        py::tuple tmpOwner = py::make_tuple();
        py::array tmp(inferredDType, shape, strides, imgData.plane(p).basePtr, tmpOwner);
        out.emplace_back(tmp.request(), layout);
    }

    return out;
}

std::vector<py::object> ToPython(const nvcv::IImageData &imgData, std::optional<nvcv::TensorLayout> userLayout,
                                 py::object owner)
{
    std::vector<py::object> out;

    auto *pitchData = dynamic_cast<const nvcv::IImageDataStrided *>(&imgData);
    if (!pitchData)
    {
        throw std::runtime_error("Only images with pitch-linear formats can be exported");
    }

    for (const auto &[info, layout] : ToPyBufferInfo(*pitchData, userLayout))
    {
        if (dynamic_cast<const nvcv::IImageDataStridedCuda *>(pitchData))
        {
            if (owner)
            {
                out.emplace_back(py::cast(std::make_shared<CudaBuffer>(info, false),
                                          py::return_value_policy::reference_internal, owner));
            }
            else
            {
                out.emplace_back(
                    py::cast(std::make_shared<CudaBuffer>(info, true), py::return_value_policy::take_ownership));
            }
        }
        else if (dynamic_cast<const nvcv::IImageDataStridedHost *>(pitchData))
        {
            // With no owner, python/pybind11 will make a copy of the data
            out.emplace_back(py::array(info, owner));
        }
        else
        {
            throw std::runtime_error("Buffer type not supported");
        }
    }

    return out;
}

} // namespace

py::object Image::cuda(std::optional<nvcv::TensorLayout> layout) const
{
    // Do we need to redefine the cuda object?
    // (not defined yet, or requested layout is different)
    if (!m_cacheCudaObject || layout != m_cacheCudaObjectLayout)
    {
        // No layout requested and we're wrapping external data?
        if (!layout && m_wrapped)
        {
            // That's what we'll return, as m_impl is wrapping it.
            m_cacheCudaObject = m_wrapped;
        }
        else
        {
            const nvcv::IImageData *imgData = m_impl->exportData();
            if (!imgData)
            {
                throw std::runtime_error("Image data can't be exported");
            }

            std::vector<py::object> out = ToPython(*imgData, layout, py::cast(*this));

            m_cacheCudaObjectLayout = layout;
            if (out.size() == 1)
            {
                m_cacheCudaObject = std::move(out[0]);
            }
            else
            {
                m_cacheCudaObject = py::cast(out);
            }
        }
    }

    return m_cacheCudaObject;
}

py::object Image::cpu(std::optional<nvcv::TensorLayout> layout) const
{
    const nvcv::IImageData *devData = m_impl->exportData();
    if (!devData)
    {
        throw std::runtime_error("Image data can't be exported");
    }

    auto *devStrided = dynamic_cast<const nvcv::IImageDataStridedCuda *>(devData);
    if (!devStrided)
    {
        throw std::runtime_error("Only images with pitch-linear formats can be exported");
    }

    std::vector<std::pair<py::buffer_info, nvcv::TensorLayout>> vDevBufInfo = ToPyBufferInfo(*devStrided, layout);

    std::vector<py::object> out;

    for (const auto &[devBufInfo, bufLayout] : vDevBufInfo)
    {
        std::vector<ssize_t> shape      = devBufInfo.shape;
        std::vector<ssize_t> devStrides = devBufInfo.strides;

        py::array hostData(util::ToDType(devBufInfo), shape);

        py::buffer_info      hostBufInfo = hostData.request();
        std::vector<ssize_t> hostStrides = hostBufInfo.strides;

        auto infoShape
            = nvcv::TensorShapeInfoImagePlanar::Create(nvcv::TensorShape(shape.data(), shape.size(), bufLayout));
        NVCV_ASSERT(infoShape);

        int nplanes = infoShape->numPlanes();
        int ncols   = infoShape->numCols();
        int nrows   = infoShape->numRows();

        ssize_t colStride = devStrides[infoShape->infoLayout().idxWidth()];
        NVCV_ASSERT(colStride == hostStrides[infoShape->infoLayout().idxWidth()]); // both must be packed

        ssize_t hostRowStride, devRowStride;
        if (infoShape->infoLayout().idxHeight() >= 0)
        {
            devRowStride  = devStrides[infoShape->infoLayout().idxHeight()];
            hostRowStride = hostStrides[infoShape->infoLayout().idxHeight()];
        }
        else
        {
            devRowStride  = colStride * ncols;
            hostRowStride = colStride * ncols;
        }

        ssize_t hostPlaneStride = hostRowStride * nrows;
        ssize_t devPlaneStride  = devRowStride * nrows;

        for (int p = 0; p < nplanes; ++p)
        {
            util::CheckThrow(cudaMemcpy2D(reinterpret_cast<std::byte *>(hostBufInfo.ptr) + p * hostPlaneStride,
                                          hostRowStride,
                                          reinterpret_cast<std::byte *>(devBufInfo.ptr) + p * devPlaneStride,
                                          devRowStride, ncols * colStride, nrows, cudaMemcpyDeviceToHost));
        }

        out.push_back(std::move(hostData));
    }

    if (out.size() == 1)
    {
        return std::move(out[0]);
    }
    else
    {
        return py::cast(out);
    }
}

void Image::Export(py::module &m)
{
    using namespace py::literals;

    py::class_<Image, std::shared_ptr<Image>, Container>(m, "Image")
        .def(py::init(&Image::Create), "size"_a, "format"_a)
        .def(py::init(&Image::CreateHost), "buffer"_a, "format"_a = nvcv::FMT_NONE)
        .def(py::init(&Image::CreateHostVector), "buffer"_a, "format"_a = nvcv::FMT_NONE)
        .def_static("zeros", &Image::Zeros, "size"_a, "format"_a)
        .def("__repr__", &util::ToString<Image>)
        .def("cuda", &Image::cuda, "layout"_a = std::nullopt)
        .def("cpu", &Image::cpu, "layout"_a = std::nullopt)
        .def_property_readonly("size", &Image::size)
        .def_property_readonly("width", &Image::width)
        .def_property_readonly("height", &Image::height)
        .def_property_readonly("format", &Image::format);

    // Make sure buffer lifetime is tied to image's (keep_alive)
    m.def("as_image", &Image::WrapCuda, "buffer"_a, "format"_a = nvcv::FMT_NONE, py::keep_alive<0, 1>());
    m.def("as_image", &Image::WrapCudaVector, "buffer"_a, "format"_a = nvcv::FMT_NONE, py::keep_alive<0, 1>());
}

} // namespace nvcvpy::priv
