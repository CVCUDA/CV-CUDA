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

#include "Image.hpp"

#include "../NvtxRange.hpp"
#include "Cache.hpp"
#include "CastUtils.hpp"
#include "DataType.hpp"
#include "ImageFormat.hpp"
#include "Stream.hpp"

#include <common/Assert.hpp>
#include <common/CheckError.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <dlpack/dlpack.h>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/TensorShapeInfo.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <stdexcept>

namespace nvcvpy::priv {

bool Image::Key::doIsCompatible(const IKey &ithat) const
{
    auto &that = static_cast<const Key &>(ithat);

    // Wrapper key's all compare equal, are they can't be used
    // and whenever we query the cache for wrappers, we really
    // want to get them all (as long as they aren't being used).
    if (m_isWrapper && that.m_isWrapper)
    {
        return true;
    }
    else if (m_isWrapper || that.m_isWrapper) // xor
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
    if (m_isWrapper)
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

class ImageError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

struct BufferImageInfo
{
    int            numPlanes;
    nvcv::Size2D   size;
    int            numChannels;
    bool           isChannelLast;
    int64_t        planeStride;
    int64_t        rowStride;
    nvcv::DataType dtype;
    NVCVByte      *data;
};

struct BufferTensorInfo
{
    std::array<ssize_t, 4> shape;
    std::array<ssize_t, 4> strides;
    nvcv::TensorLayout     layout;
};

nvcv::TensorLayout SelectBufferLayout(const nvcv::ImageFormat &fmt, size_t plane, ssize_t channels)
{
    if (fmt != nvcv::FMT_NONE)
    {
        return fmt.planeNumChannels(static_cast<int>(plane)) == channels ? nvcv::TENSOR_NHWC : nvcv::TENSOR_NCHW;
    }

    return channels <= 4 ? nvcv::TENSOR_NHWC : nvcv::TENSOR_NCHW;
}

BufferTensorInfo MakeBufferTensorInfo(const DLTensor &tensor, const nvcv::ImageFormat &fmt, size_t plane,
                                      int elemStrideBytes)
{
    BufferTensorInfo info{};

    switch (tensor.ndim)
    {
    case 1:
        info.layout     = nvcv::TENSOR_NCHW;
        info.shape      = {1, 1, 1, tensor.shape[0]};
        info.strides[0] = tensor.strides[0] * elemStrideBytes;
        info.strides[1] = info.strides[0];
        info.strides[2] = info.strides[0];
        info.strides[3] = info.strides[0];
        break;

    case 2:
        info.layout     = nvcv::TENSOR_NCHW;
        info.shape      = {1, 1, tensor.shape[0], tensor.shape[1]};
        info.strides[0] = tensor.shape[0] * tensor.strides[0] * elemStrideBytes;
        info.strides[1] = info.strides[0];
        info.strides[2] = tensor.strides[0] * elemStrideBytes;
        info.strides[3] = tensor.strides[1] * elemStrideBytes;
        break;

    case 3:
    case 4:
        info.shape[0] = tensor.ndim == 3 ? 1 : tensor.shape[tensor.ndim - 4];
        info.shape[1] = tensor.shape[tensor.ndim - 3];
        info.shape[2] = tensor.shape[tensor.ndim - 2];
        info.shape[3] = tensor.shape[tensor.ndim - 1];
        info.layout   = SelectBufferLayout(fmt, plane, info.shape[3]);

        info.strides[1] = tensor.strides[tensor.ndim - 3] * elemStrideBytes;
        info.strides[2] = tensor.strides[tensor.ndim - 2] * elemStrideBytes;
        info.strides[3] = tensor.strides[tensor.ndim - 1] * elemStrideBytes;
        info.strides[0] = tensor.ndim == 3 ? info.shape[1] * info.strides[1] : tensor.strides[tensor.ndim - 4];
        break;

    default:
        throw std::invalid_argument(
            util::ConcatString("Number of buffer dimensions must be between 1 and 4, not ", tensor.ndim));
    }

    return info;
}

void ValidateBufferStrides(const BufferTensorInfo &bufferInfo, int elemStrideBytes,
                           const nvcv::TensorShapeInfoImagePlanar &infoShape)
{
    const auto &strides = bufferInfo.strides;
    if (strides[0] <= 0 || strides[1] <= 0 || strides[2] <= 0)
    {
        throw std::invalid_argument("Buffer strides must be all >= 1");
    }

    const auto &infoLayout = infoShape.infoLayout();

    if (strides[3] != elemStrideBytes)
    {
        throw std::invalid_argument(
            util::ConcatString("Fastest changing dimension must be packed, i.e., have stride equal to ",
                               elemStrideBytes, " byte(s), not ", strides[3]));
    }

    ssize_t packedRowStride = static_cast<ssize_t>(elemStrideBytes) * infoShape.numCols();
    if (ssize_t rowStride = strides[infoLayout.idxHeight()];
        !infoLayout.isChannelLast() && rowStride != packedRowStride)
    {
        throw std::invalid_argument(util::ConcatString("Image row must packed, i.e., have stride equal to ",
                                                       packedRowStride, " byte(s), not ", rowStride));
    }
}

BufferImageInfo MakeBufferImageInfo(const DLTensor &tensor, const nvcv::ImageFormat &fmt, size_t plane)
{
    int  elemStrideBytes = (tensor.dtype.bits * tensor.dtype.lanes + 7) / 8;
    auto tensorInfo      = MakeBufferTensorInfo(tensor, fmt, plane, elemStrideBytes);

    auto infoShape = nvcv::TensorShapeInfoImagePlanar::Create(
        nvcv::TensorShape(tensorInfo.shape.data(), tensorInfo.shape.size(), tensorInfo.layout));
    NVCV_ASSERT(infoShape);

    ValidateBufferStrides(tensorInfo, elemStrideBytes, *infoShape);

    const auto &infoLayout = infoShape->infoLayout();

    BufferImageInfo bufferInfo;
    bufferInfo.isChannelLast = infoLayout.isChannelLast();
    bufferInfo.numPlanes
        = bufferInfo.isChannelLast ? static_cast<int>(infoShape->numSamples()) : infoShape->numChannels();
    bufferInfo.numChannels = infoShape->numChannels();
    bufferInfo.size        = infoShape->size();
    bufferInfo.planeStride
        = tensorInfo.strides[bufferInfo.isChannelLast ? infoLayout.idxSample() : infoLayout.idxChannel()];
    bufferInfo.rowStride = tensorInfo.strides[infoLayout.idxHeight()];
    bufferInfo.dtype     = ToNVCVDataType(tensor.dtype);
    bufferInfo.data      = static_cast<NVCVByte *>(tensor.data);
    return bufferInfo;
}

std::vector<BufferImageInfo> ExtractBufferImageInfo(const std::vector<DLPackTensor> &tensorList,
                                                    const nvcv::ImageFormat         &fmt)
{
    std::vector<BufferImageInfo> bufferInfoList;
    int                          curChannel = 0;

    for (size_t p = 0; p < tensorList.size(); ++p)
    {
        const DLTensor &tensor  = *tensorList[p];
        auto            bufInfo = MakeBufferImageInfo(tensor, fmt, p);

        curChannel += bufInfo.numPlanes * bufInfo.numChannels;
        if (curChannel > 4)
        {
            throw std::invalid_argument("Number of channels specified in a buffers must be <= 4");
        }

        NVCV_ASSERT(bufInfo.numPlanes <= 4);
        NVCV_ASSERT(bufInfo.numChannels <= 4);
        bufferInfoList.push_back(bufInfo);
    }

    return bufferInfoList;
}

nvcv::DataType MakePackedType(nvcv::DataType dtype, int numChannels)
{
    if (dtype.numChannels() == numChannels)
    {
        return dtype;
    }
    else
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
        default:
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
}

nvcv::ImageFormat InferImageFormat(const std::vector<nvcv::DataType> &planePixTypes)
{
    if (planePixTypes.empty())
    {
        return nvcv::FMT_NONE;
    }

    static_assert(NVCV_PACKING_0 == 0, "Invalid 0 packing value");
    NVCV_ASSERT(planePixTypes.size() <= 4);

    std::array<nvcv::Packing, 4> packing = {nvcv::Packing::NONE};

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

    auto numPlanes = static_cast<int>(planePixTypes.size());

    // Planar or packed?
    if (numPlanes == 1 || numChannels == numPlanes)
    {
        static const std::array<nvcv::ImageFormat, 4> baseFormatList
            = {nvcv::FMT_U8, nvcv::FMT_2F32, nvcv::FMT_RGB8, nvcv::FMT_RGBA8};

        // Validate array index to prevent buffer overrun
        if (numChannels < 1 || numChannels > 4)
        {
            throw std::invalid_argument(
                util::ConcatString("Invalid number of channels ", numChannels, ", must be between 1 and 4"));
        }

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
    // REVISIT: this test is too fragile, must improve
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

void FillNVCVImageBufferStrided(NVCVImageData &imgData, const std::vector<DLPackTensor> &infos, nvcv::ImageFormat fmt)
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
            dataStrided.planes[curPlane].rowStride = static_cast<int32_t>(b.rowStride);
            dataStrided.planes[curPlane].basePtr   = b.data + b.planeStride * p;

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
            throw std::invalid_argument(util::ConcatString("Format inferred from buffers ", inferredFormat,
                                                           " isn't compatible with given image format ", fmt));
        }
        finalFormat = fmt;
    }
    else
    {
        finalFormat = inferredFormat;
    }
    imgData.format = static_cast<NVCVImageFormat>(finalFormat);

    nvcv::Size2D imgSize = {dataStrided.planes[0].width, dataStrided.planes[0].height};

    // Now do a final check on the expected plane sizes according to the
    // format
    for (int p = 0; p < dataStrided.numPlanes; ++p)
    {
        nvcv::Size2D goldSize = finalFormat.planeSize(imgSize, p);
        nvcv::Size2D plSize{dataStrided.planes[p].width, dataStrided.planes[p].height};

        if (plSize.w != goldSize.w || plSize.h != goldSize.h)
        {
            throw std::invalid_argument(util::ConcatString(
                "Plane ", p, "'s size ", plSize.w, "x", plSize.h, " doesn't correspond to what's expected by ",
                (fmt == nvcv::FMT_NONE ? "inferred" : "given"), " format ", finalFormat, " of image with size ",
                imgSize.w, "x", imgSize.h));
        }
    }
}

nvcv::ImageDataStridedCuda CreateNVCVImageDataCuda(const std::vector<DLPackTensor> &infos, nvcv::ImageFormat fmt)
{
    NVCVImageData imgData;
    FillNVCVImageBufferStrided(imgData, infos, fmt);

    return nvcv::ImageDataStridedCuda(nvcv::ImageFormat{imgData.format}, imgData.buffer.strided);
}

nvcv::ImageDataStridedHost CreateNVCVImageDataHost(const std::vector<DLPackTensor> &infos, nvcv::ImageFormat fmt)
{
    NVCVImageData imgData;
    FillNVCVImageBufferStrided(imgData, infos, fmt);

    return nvcv::ImageDataStridedHost(nvcv::ImageFormat{imgData.format}, imgData.buffer.strided);
}

} // namespace

Image::Image(const Size2D &size, nvcv::ImageFormat fmt, int rowAlign)
    : m_key{size, fmt}
{
    nvcv::MemAlignment    bufAlign = rowAlign == 0 ? nvcv::MemAlignment{} : nvcv::MemAlignment{}.rowAddr(rowAlign);
    NVCVImageRequirements reqs;

    nvcvImageCalcRequirements(std::get<0>(size), std::get<1>(size), static_cast<NVCVImageFormat>(fmt),
                              bufAlign.baseAddr(), bufAlign.rowAddr(), &reqs);

    m_impl         = nvcv::Image(reqs);
    m_size_inbytes = doComputeSizeInBytes(reqs);
}

Image::Image(std::vector<std::shared_ptr<ExternalBuffer>> bufs, const nvcv::ImageDataStridedCuda &imgData)
    : m_size_inbytes{doComputeSizeInBytes(NVCVImageRequirements())}
{
    m_wrapData.emplace();

    this->setWrapData(std::move(bufs), imgData);
}

Image::Image(const std::vector<py::buffer> &, const nvcv::ImageDataStridedHost &hostData, int rowAlign)
{
    // Input buffer is host data.
    // We'll create a regular image and copy the host data into it.

    // Create the image with same size and format as host data
    nvcv::MemAlignment    bufAlign = nvcv::MemAlignment{}.rowAddr(rowAlign);
    NVCVImageRequirements reqs;

    nvcvImageCalcRequirements(hostData.size().w, hostData.size().h, static_cast<NVCVImageFormat>(hostData.format()),
                              bufAlign.baseAddr(), bufAlign.rowAddr(), &reqs);

    m_impl         = nvcv::Image(reqs);
    m_size_inbytes = doComputeSizeInBytes(reqs);

    auto devData = *m_impl.exportData<nvcv::ImageDataStridedCuda>();
    NVCV_ASSERT(hostData.format() == devData.format());
    NVCV_ASSERT(hostData.numPlanes() == devData.numPlanes());

    // Now copy each plane from host to device
    for (int p = 0; p < devData.numPlanes(); ++p)
    {
        const nvcv::ImagePlaneStrided &devPlane  = devData.plane(p);
        const nvcv::ImagePlaneStrided &hostPlane = hostData.plane(p);

        NVCV_ASSERT(devPlane.width == hostPlane.width);
        NVCV_ASSERT(devPlane.height == hostPlane.height);

        util::CheckThrow(cudaMemcpy2D(devPlane.basePtr, devPlane.rowStride, hostPlane.basePtr, hostPlane.rowStride,
                                      static_cast<size_t>(hostPlane.width) * hostData.format().planePixelStrideBytes(p),
                                      hostPlane.height, cudaMemcpyHostToDevice));
    }

    m_key = Key{
        {m_impl.size().w, m_impl.size().h},
        m_impl.format()
    };
}

int64_t Image::doComputeSizeInBytes(const NVCVImageRequirements &reqs) const
{
    int64_t size_inbytes;
    util::CheckThrow(nvcvMemRequirementsCalcTotalSizeBytes(&(reqs.mem.cudaMem), &size_inbytes));
    return size_inbytes;
}

int64_t Image::GetSizeInBytes() const
{
    // m_size_inbytes == -1 indicates failure case and value has not been computed yet
    NVCV_ASSERT(m_size_inbytes != -1 && "Image has m_size_inbytes == -1, ie m_size_inbytes has not been correctly set");
    return m_size_inbytes;
}

std::shared_ptr<Image> Image::Create(const Size2D &size, nvcv::ImageFormat fmt, int rowAlign)
{
    std::vector<std::shared_ptr<CacheItem>> vcont = Cache::Instance().fetch(Key{size, fmt});

    // None found?
    if (vcont.empty())
    {
        std::shared_ptr<Image> img(new Image(size, fmt, rowAlign)); // NOSONAR: constructor is private.
        Cache::Instance().add(*img);
        return img;
    }
    else
    {
        // Get the first one
        return std::static_pointer_cast<Image>(vcont[0]);
    }
}

std::shared_ptr<Image> Image::Zeros(const Size2D &size, nvcv::ImageFormat fmt, int rowAlign)
{
    auto img = Image::Create(size, fmt, rowAlign);

    auto data = *img->impl().exportData<nvcv::ImageDataStridedCuda>();

    for (int p = 0; p < data.numPlanes(); ++p)
    {
        const nvcv::ImagePlaneStrided &plane = data.plane(p);

        util::CheckThrow(cudaMemset2D(plane.basePtr, plane.rowStride, 0,
                                      static_cast<size_t>(plane.width) * data.format().planePixelStrideBytes(p),
                                      plane.height));
    }

    return img;
}

std::shared_ptr<Image> Image::WrapExternalBuffer(ExternalBuffer &buffer, nvcv::ImageFormat fmt)
{
    py::object obj = py::cast(buffer.shared_from_this());
    return WrapExternalBufferVector({obj}, fmt);
}

// Seed the Image's Resource with the producer stream of its wrapping
// ExternalBuffer so the first cvcuda op reading the image inserts the
// necessary cross-stream wait (CAI v3 `stream` field honoring).
//
// For multi-plane buffers we conservatively seed from the first buffer that
// advertises a (non-synced) stream; common-case cupy/torch interop uses a
// single buffer so this captures the full producer-stream contract.
static void SeedImageFromBuffers(Image &img, const std::vector<std::shared_ptr<ExternalBuffer>> &bufs)
{
    for (const auto &buf : bufs)
    {
        if (!buf || buf->producerIsSynced() || buf->producerStream() == nullptr)
        {
            continue;
        }
        int device = buf->producerDevice();
        if (device < 0)
        {
            util::CheckThrow(cudaGetDevice(&device));
        }
        img.seedLastStream(buf->producerStream(), device);
        return;
    }
}

std::vector<std::shared_ptr<Image>> Image::WrapExternalBufferMany(std::vector<std::shared_ptr<ExternalBuffer>> &buffers,
                                                                  nvcv::ImageFormat                             fmt)
{
    // This is the key of an image wrapper.
    // All image wrappers have the same key.
    Image::Key key;

    std::vector<std::shared_ptr<CacheItem>> items = Cache::Instance().fetch(key);

    std::vector<std::shared_ptr<Image>> out;
    out.reserve(buffers.size());

    for (const auto &buffer : buffers)
    {
        std::vector<std::shared_ptr<ExternalBuffer>> spBuffers;
        spBuffers.push_back(buffer);

        if (!spBuffers.back())
            throw ImageError("Input buffer doesn't provide cuda_array_interface or DLPack interfaces");

        std::vector<DLPackTensor> bufinfos;
        bufinfos.emplace_back(spBuffers[0]->dlTensor());
        nvcv::ImageDataStridedCuda imgData = CreateNVCVImageDataCuda(bufinfos, fmt);

        // None found?
        if (items.empty())
        {
            // Need to add wrappers into cache so that they don't get destroyed by
            // the cuda stream when they're last used, and python script isn't
            // holding a reference to them. If we don't do it, things might break.
            std::shared_ptr<Image> img(new Image(spBuffers, imgData)); // NOSONAR: constructor is private.
            SeedImageFromBuffers(*img, spBuffers);
            Cache::Instance().add(*img);
            out.push_back(img);
        }
        else
        {
            std::shared_ptr<Image> img = std::static_pointer_cast<Image>(items.back());
            items.pop_back();
            img->setWrapData(spBuffers, imgData);
            SeedImageFromBuffers(*img, spBuffers);
            out.push_back(img);
        }
    }

    // Release any over-fetched or pre-existing not-in-use wrappers so their
    // ExternalBuffer references (and thus the wrapped GPU buffers) are freed
    // promptly.  Drop 'items' first so those shared_ptrs no longer count as
    // "in use", then run the cleanup.  Images in 'out' are still in-use and
    // will not be removed.
    items.clear();
    Cache::Instance().removeAllNotInUseMatching(key);

    return out;
}

std::shared_ptr<Image> Image::WrapExternalBufferVector(std::vector<py::object> buffers, nvcv::ImageFormat fmt)
{
    std::vector<std::shared_ptr<ExternalBuffer>> spBuffers;
    for (auto &obj : buffers)
    {
        std::shared_ptr<ExternalBuffer> buffer = cast_py_object_as<ExternalBuffer>(obj);
        if (!buffer)
            throw ImageError("Input buffer doesn't provide cuda_array_interface or DLPack interfaces");
        spBuffers.push_back(std::move(buffer));
    }

    std::vector<DLPackTensor> bufinfos;

    for (const auto &buffer : spBuffers)
    {
        bufinfos.emplace_back(buffer->dlTensor());
    }

    nvcv::ImageDataStridedCuda imgData = CreateNVCVImageDataCuda(bufinfos, fmt);

    // This is the key of an image wrapper.
    // All image wrappers have the same key.
    Image::Key key;

    std::shared_ptr<CacheItem> item = Cache::Instance().fetchOne(key);
    std::shared_ptr<Image>     img;

    // None found?
    if (!item)
    {
        // Need to add wrappers into cache so that they don't get destroyed by
        // the cuda stream when they're last used, and python script isn't
        // holding a reference to them. If we don't do it, things might break.
        img = std::shared_ptr<Image>(new Image(spBuffers, imgData)); // NOSONAR: constructor is private.
        Cache::Instance().add(*img);
    }
    else
    {
        img = std::static_pointer_cast<Image>(item);
        img->setWrapData(spBuffers, imgData);
    }
    SeedImageFromBuffers(*img, spBuffers);

    // Release any other not-in-use wrappers so their ExternalBuffer references
    // (and thus the wrapped GPU buffers) are freed promptly.  The current img
    // is in-use and will not be removed.
    Cache::Instance().removeAllNotInUseMatching(key);

    return img;
}

void Image::setWrapData(std::vector<std::shared_ptr<ExternalBuffer>> bufs, const nvcv::ImageDataStridedCuda &imgData)
{
    NVCV_ASSERT(m_wrapData);

    NVCV_ASSERT(bufs.size() >= 1);
    const DLDeviceType devType = bufs[0]->dlTensor().device.device_type;
    py::object         newObj;

    if (bufs.size() == 1)
    {
        newObj = py::cast(bufs[0]);
    }
    else
    {
        for (size_t i = 1; i < bufs.size(); ++i)
        {
            if (bufs[i]->dlTensor().device.device_type != bufs[0]->dlTensor().device.device_type
                || bufs[i]->dlTensor().device.device_id != bufs[0]->dlTensor().device.device_id)
            {
                throw ImageError("All buffers must belong to the same device, but some don't.");
            }
        }

        newObj = py::cast(std::move(bufs));
    }

    nvcv::Image newImpl = nvcv::ImageWrapData(imgData);

    // Cache::fetch only returns wrappers after their prior stream work has
    // released them, so the old buffer's ordering state is safe to discard.
    resetLastStreamForRebind();
    m_wrapData->devType = devType;
    m_wrapData->obj     = std::move(newObj);

    //We recreate the nvcv::Image wrapper (m_impl) because it's cheap.
    //It's not cheap to create nvcvpy::Image as it might have allocated expensive resources (cudaEvent_t in Resource parent).
    m_impl = std::move(newImpl);
}

std::shared_ptr<Image> Image::CreateHost(py::buffer buffer, nvcv::ImageFormat fmt, int rowAlign)
{
    return CreateHostVector(std::vector{buffer}, fmt, rowAlign);
}

std::shared_ptr<Image> Image::CreateHostVector(const std::vector<py::buffer> &buffers, nvcv::ImageFormat fmt,
                                               int rowAlign)
{
    std::vector<DLPackTensor> dlTensorList;

    for (const auto &buffer : buffers)
    {
        dlTensorList.emplace_back(buffer.request(), DLDevice{kDLCPU, 0});
    }

    nvcv::ImageDataStridedHost imgData = CreateNVCVImageDataHost(dlTensorList, fmt);

    // We take this opportunity to remove all wrappers from cache.
    // They aren't reusable anyway.
    Image::Key key;
    Cache::Instance().removeAllNotInUseMatching(key);

    std::shared_ptr<Image> img(new Image(buffers, imgData, rowAlign)); // NOSONAR: constructor is private.
    Cache::Instance().add(*img);
    return img;
}

Size2D Image::size() const
{
    nvcv::Size2D s = m_impl.size();
    return {s.w, s.h};
}

int32_t Image::width() const
{
    return m_impl.size().w;
}

int32_t Image::height() const
{
    return m_impl.size().h;
}

nvcv::ImageFormat Image::format() const
{
    return m_impl.format();
}

std::ostream &operator<<(std::ostream &out, const Image &img)
{
    std::string size_str = std::to_string(img.width()) + 'x' + std::to_string(img.height());

    return out << "<nvcv.Image " << size_str << ' ' << img.format() << '>';
}

namespace {

struct InferredBufferInfo
{
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;
    nvcv::TensorLayout   layout;
    py::dtype            dtype;
};

void ValidateExportLayout(const std::optional<nvcv::TensorLayout> &userLayout)
{
    if (!userLayout)
    {
        return;
    }

    if (!nvcv::TensorLayoutInfoImage::Create(*userLayout))
    {
        throw ImageError("Layout can't represent the planar images needed");
    }
}

bool PlaneMatchesSingleBuffer(const nvcv::ImageDataStrided &imgData, const nvcv::ImagePlaneStrided &firstPlane, int p)
{
    const nvcv::ImagePlaneStrided &plane = imgData.plane(p);

    return plane.width == firstPlane.width && plane.height == firstPlane.height
        && plane.rowStride == firstPlane.rowStride && imgData.format().planeDataType(0).numChannels() < 2
        && imgData.format().planeDataType(0) == imgData.format().planeDataType(p);
}

bool PlaneStrideMatchesSingleBuffer(const nvcv::ImageDataStrided &imgData, int p)
{
    intptr_t goldPlaneStride = imgData.plane(1).basePtr - imgData.plane(0).basePtr;
    intptr_t curPlaneStride  = imgData.plane(p).basePtr - imgData.plane(p - 1).basePtr;
    return curPlaneStride == goldPlaneStride;
}

bool CanExportAsSingleBuffer(const nvcv::ImageDataStrided &imgData)
{
    const nvcv::ImagePlaneStrided &firstPlane = imgData.plane(0);

    for (int p = 1; p < imgData.numPlanes(); ++p)
    {
        if (!PlaneMatchesSingleBuffer(imgData, firstPlane, p))
        {
            return false;
        }

        if (p >= 2 && !PlaneStrideMatchesSingleBuffer(imgData, p))
        {
            return false;
        }
    }

    return true;
}

int PlaneNumChannelsForExport(const nvcv::ImageFormat &format, int p)
{
    switch (format.planePacking(p))
    {
    // These (YUYV, UYVY, ...) need some special treatment.
    // Although it's 3 channels in the plane, it's actually two channels per pixel.
    case nvcv::Packing::X8_Y8__X8_Z8:
    case nvcv::Packing::Y8_X8__Z8_X8:
        return 2;

    default:
        return format.planeNumChannels(p);
    }
}

InferredBufferInfo InferSingleBufferInfo(const nvcv::ImageDataStrided &imgData, int p)
{
    const nvcv::ImagePlaneStrided &plane    = imgData.plane(p);
    int                            planeBPP = imgData.format().planeDataType(p).strideBytes();

    if (imgData.format().numChannels() == 1)
    {
        NVCV_ASSERT(imgData.numPlanes() == 1);

        InferredBufferInfo info;
        info.shape   = {plane.height, plane.width};
        info.strides = {plane.rowStride, planeBPP};
        info.layout  = nvcv::TensorLayout{"HW"};
        info.dtype   = py::cast(imgData.format().planeDataType(p));
        return info;
    }

    if (imgData.numPlanes() == 1)
    {
        int planeNumChannels = PlaneNumChannelsForExport(imgData.format(), p);
        NVCV_ASSERT(planeNumChannels >= 2);

        InferredBufferInfo info;
        info.shape   = {plane.height, plane.width, planeNumChannels};
        info.strides = {plane.rowStride, planeBPP, planeBPP / planeNumChannels};
        info.layout  = nvcv::TensorLayout{"HWC"};
        info.dtype   = py::cast(imgData.format().planeDataType(p).channelType(0));
        return info;
    }

    NVCV_ASSERT(PlaneNumChannelsForExport(imgData.format(), p) == 1);

    intptr_t planeStride = imgData.plane(1).basePtr - imgData.plane(0).basePtr;
    NVCV_ASSERT(planeStride > 0);

    InferredBufferInfo info;
    info.shape   = {imgData.numPlanes(), plane.height, plane.width};
    info.strides = {planeStride, plane.rowStride, planeBPP};
    info.layout  = nvcv::TensorLayout{"CHW"};
    info.dtype   = py::cast(imgData.format().planeDataType(p));
    return info;
}

InferredBufferInfo InferPlaneBufferInfo(const nvcv::ImageDataStrided &imgData, int p)
{
    const nvcv::ImagePlaneStrided &plane            = imgData.plane(p);
    int                            planeNumChannels = PlaneNumChannelsForExport(imgData.format(), p);
    int                            planeBPP         = imgData.format().planeDataType(p).strideBytes();

    NVCV_ASSERT(imgData.numPlanes() >= 2);

    InferredBufferInfo info;
    info.shape   = {plane.height, plane.width, planeNumChannels};
    info.strides = {static_cast<ssize_t>(plane.rowStride), static_cast<ssize_t>(planeBPP),
                    static_cast<ssize_t>(planeBPP / planeNumChannels)};
    info.layout  = nvcv::TensorLayout{"HWC"};
    info.dtype   = py::cast(imgData.format().planeDataType(p).channelType(0));
    return info;
}

InferredBufferInfo InferBufferInfo(const nvcv::ImageDataStrided &imgData, int p, int numBuffers)
{
    if (numBuffers == 1)
    {
        return InferSingleBufferInfo(imgData, p);
    }

    return InferPlaneBufferInfo(imgData, p);
}

void ValidateRequiredLayoutDimensions(const InferredBufferInfo &inferred, const nvcv::TensorLayout &userLayout)
{
    for (int i = 0; i < inferred.layout.rank(); ++i)
    {
        if (inferred.shape[i] >= 2 && userLayout.find(inferred.layout[i]) < 0)
        {
            throw py::value_error(util::ConcatString("Layout need dimension '", inferred.layout[i], "'"));
        }
    }
}

InferredBufferInfo ApplyUserLayout(const InferredBufferInfo &inferred, const nvcv::TensorLayout &userLayout)
{
    InferredBufferInfo out{{}, {}, userLayout, inferred.dtype};
    int                idxLastInferDim = -1;

    ValidateRequiredLayoutDimensions(inferred, userLayout);

    for (int i = 0; i < userLayout.rank(); ++i)
    {
        int idxInferDim = inferred.layout.find(userLayout[i]);

        if (idxInferDim < 0)
        {
            out.shape.push_back(1);
            // REVISIT: must do better than this
            out.strides.push_back(0);
            continue;
        }

        // The order of channels must be the same, despite of user layout having
        // some other channels in the layout in between the channels in inferredLayout.
        if (idxLastInferDim >= idxInferDim)
        {
            throw ImageError("Layout not compatible with image to be exported");
        }

        idxLastInferDim = idxInferDim;
        out.shape.push_back(inferred.shape[idxInferDim]);
        out.strides.push_back(inferred.strides[idxInferDim]);
    }

    return out;
}

InferredBufferInfo ResolveBufferInfoLayout(const InferredBufferInfo                &inferred,
                                           const std::optional<nvcv::TensorLayout> &userLayout)
{
    if (userLayout)
    {
        return ApplyUserLayout(inferred, *userLayout);
    }

    return inferred;
}

std::vector<std::pair<py::buffer_info, nvcv::TensorLayout>> ToPyBufferInfo(const nvcv::ImageDataStrided     &imgData,
                                                                           std::optional<nvcv::TensorLayout> userLayout)
{
    if (imgData.numPlanes() < 1)
    {
        return {};
    }

    ValidateExportLayout(userLayout);

    std::vector<std::pair<py::buffer_info, nvcv::TensorLayout>> out;

    // If not using a single buffer, we'll forcibly use one buffer per plane.
    int numBuffers = CanExportAsSingleBuffer(imgData) ? 1 : imgData.numPlanes();

    for (int p = 0; p < numBuffers; ++p)
    {
        NVCV_ASSERT(numBuffers == 1 || imgData.numPlanes() == numBuffers);

        InferredBufferInfo inferred = InferBufferInfo(imgData, p, numBuffers);
        NVCV_ASSERT(static_cast<ssize_t>(inferred.shape.size()) == inferred.layout.rank());
        NVCV_ASSERT(static_cast<ssize_t>(inferred.strides.size()) == inferred.layout.rank());

        InferredBufferInfo resolved = ResolveBufferInfoLayout(inferred, userLayout);

        // There's no direct way to construct a py::buffer_info from data together with a py::dtype.
        // To do that, we first construct a py::array (it accepts py::dtype), and use ".request()"
        // to retrieve the corresponding py::buffer_info.
        // To avoid spurious data copies in py::array ctor, we create this dummy owner.
        py::tuple tmpOwner = py::make_tuple();
        py::array tmp(resolved.dtype, resolved.shape, resolved.strides, imgData.plane(p).basePtr, tmpOwner);
        out.emplace_back(tmp.request(), resolved.layout);
    }

    return out;
}

std::vector<py::object> ToPython(const nvcv::ImageData &imgData, std::optional<nvcv::TensorLayout> userLayout,
                                 py::object owner, cudaStream_t exportStream, bool setExportStream)
{
    std::vector<py::object> out;

    auto pitchData = imgData.cast<nvcv::ImageDataStrided>();
    if (!pitchData)
    {
        throw ImageError("Only images with pitch-linear formats can be exported");
    }

    for (const auto &[info, layout] : ToPyBufferInfo(*pitchData, userLayout))
    {
        if (pitchData->cast<nvcv::ImageDataStridedCuda>())
        {
            // REVISIT: set correct device_type and device_id
            out.emplace_back(ExternalBuffer::Create(
                DLPackTensor{
                    info,
                    {kDLCUDA, 0}
            },
                owner, exportStream, setExportStream));
        }
        else if (pitchData->cast<nvcv::ImageDataStridedHost>())
        {
            // With no owner, python/pybind11 will make a copy of the data
            out.emplace_back(py::array(info, owner));
        }
        else
        {
            throw ImageError("Buffer type not supported");
        }
    }

    return out;
}

} // namespace

py::object Image::cuda(std::optional<nvcv::TensorLayout> layout) const
{
    // No layout requested and we're wrapping external data?
    if (!layout && m_wrapData)
    {
        if (!IsCudaAccessible(m_wrapData->devType))
        {
            throw ImageError("Image data can't be exported, it's not cuda-accessible");
        }

        // That's what we'll return, as m_impl is wrapping it.
        return m_wrapData->obj;
    }
    else
    {
        auto imgData = m_impl.exportData<nvcv::ImageDataStridedCuda>();
        if (!imgData)
        {
            throw ImageError("Image data can't be exported, it's not cuda-accessible");
        }

        // Advertise the stream the image's data was last written on so
        // downstream consumers (cupy/torch) can sync via CAI `stream`.
        cudaStream_t lastStream = this->getLastStreamHandle();
        bool         setStream  = lastStream != nullptr;

        std::vector<py::object> out = ToPython(*imgData, layout, py::cast(*this), lastStream, setStream);

        if (out.size() == 1)
        {
            return std::move(out[0]);
        }
        else
        {
            return py::cast(out);
        }
    }
}

py::object Image::cpu(std::optional<nvcv::TensorLayout> layout) const
{
    auto devStrided = m_impl.exportData<nvcv::ImageDataStridedCuda>();
    if (!devStrided)
    {
        throw ImageError("Only images with pitch-linear formats can be exported to CPU");
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

        auto infoShape = nvcv::TensorShapeInfoImagePlanar::Create(
            nvcv::TensorShape(shape.data(), static_cast<int32_t>(shape.size()), bufLayout));
        NVCV_ASSERT(infoShape);

        int nplanes = infoShape->numPlanes();
        int ncols   = infoShape->numCols();
        int nrows   = infoShape->numRows();

        ssize_t colStride = devStrides[infoShape->infoLayout().idxWidth()];
        NVCV_ASSERT(colStride == hostStrides[infoShape->infoLayout().idxWidth()]); // both must be packed

        ssize_t hostRowStride;
        ssize_t devRowStride;
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

    py::class_<Image, std::shared_ptr<Image>, Container>(m, "Image", "Image")
        .def(py::init(&Image::Create), "size"_a, "format"_a, "rowalign"_a = 0,
             "Constructor that takes a size, format and optional row align of the image")
        .def(py::init(&Image::CreateHost), "buffer"_a, "format"_a = nvcv::FMT_NONE, "rowalign"_a = 0,
             "Constructor that takes a host buffer, format and optional row align")
        .def(py::init(&Image::CreateHostVector), "buffer"_a, "format"_a = nvcv::FMT_NONE, "rowalign"_a = 0,
             "Constructor that takes a host buffer vector, format and optional row align")
        .def_static("zeros", &Image::Zeros, "size"_a, "format"_a, "rowalign"_a = 0,
                    "Create an image filled with zeros with a given size, format and optional row align")
        .def("__repr__", &util::ToString<Image>)
        .def("cuda", ::cvcudapy::NvtxTrace("cvcuda.Image.cuda", &Image::cuda), "layout"_a = std::nullopt,
             "The image on the CUDA device")
        .def("cpu", ::cvcudapy::NvtxTrace("cvcuda.Image.cpu", &Image::cpu), "layout"_a = std::nullopt,
             "The image on the CPU")
        .def_property_readonly("size", &Image::size, "Read-only property that returns the size of the image")
        .def_property_readonly("width", &Image::width, "Read-only property that returns the width of the image")
        .def_property_readonly("height", &Image::height, "Read-only property that returns the height of the image")
        .def_property_readonly("format", &Image::format, "Read-only property that returns the format of the image");

    // Make sure buffer lifetime is tied to image's (keep_alive)
    m.def("as_image", ::cvcudapy::NvtxTrace("cvcuda.as_image", &Image::WrapExternalBuffer), "buffer"_a,
          "format"_a = nvcv::FMT_NONE, py::keep_alive<0, 1>(),
          "Wrap an external buffer as an image and tie the buffer lifetime to the image");
    m.def("as_image", ::cvcudapy::NvtxTrace("cvcuda.as_image", &Image::WrapExternalBufferVector),
          py::arg_v("buffer", std::vector<py::object>{}), "format"_a = nvcv::FMT_NONE, py::keep_alive<0, 1>(),
          "Wrap a vector of external buffers as an image and tie the buffer lifetime to the image");
}

} // namespace nvcvpy::priv
