/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "TensorDataUtils.hpp"

#include <cmath>

namespace nvcv::util {

static void printPlane(const uint8_t *data, int width, int height, int rowStride, int bytesPC, int numC)
{
    std::cout << "[";
    printf("%02x", data[0]);
    bool endB = false;
    int  size = height * rowStride;
    int  x    = 1;
    for (int i = 1; i < size; i++)
    {
        if (x % (width * bytesPC * numC) == 0 && !endB)
        {
            std::cout << "]";
            endB = true;
        }
        else if (x % bytesPC == 0)
        {
            if (x % (bytesPC * numC) == 0 && !endB)
            {
                std::cout << " |";
            }
            else
            {
                std::cout << ",";
            }
        }

        if (i % rowStride == 0)
        {
            std::cout << "\n[";
            endB = false;
            x    = 1;
        }
        else
        {
            x++;
            std::cout << " ";
        }
        printf("%02x", data[i]);
    }

    if (!endB)
        std::cout << "]";
    else
        std::cout << ",";

    std::cout << "\n";
}

void PrintImageFromByteVector(const uint8_t *data, int width, int height, int rowStride, int bytesPC, int numC,
                              bool planar)
{
    std::cout << "\n[H = " << height << " W = " << width << " C = " << numC << "]\n[planar = " << planar
              << " bytesPerC = " << bytesPC << " rowStride = " << rowStride << " planeStride = " << rowStride * height
              << " sampleStride = " << (planar ? (rowStride * height * numC) : (rowStride * height)) << "]\n";

    if (!planar)
    {
        printPlane(data, width, height, rowStride, bytesPC, numC);
    }
    else
    {
        for (int i = 0; i < numC; i++)
        {
            std::cout << "\nPlane = " << i << "\n";
            printPlane(&data[rowStride * i], width, height, rowStride, bytesPC, 1);
        }
    }
    std::cout << "\n";
    return;
}

TensorImageData::TensorImageData(const TensorData &tensorData, int sampleIndex)
    : m_planeStride(0)
{
    if (!nvcv::TensorDataAccessStridedImage::IsCompatible(tensorData))
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor Data not compatible with Pitch Access");

    auto tDataAc = nvcv::TensorDataAccessStridedImage::Create(tensorData);

    m_rowStride = tDataAc->rowStride();
    m_layout    = tDataAc->infoLayout().layout().m_layout;
    m_bytesPerC = tDataAc->dtype().bitsPerPixel() / 8;
    m_planar    = m_layout == NVCV_TENSOR_CHW || m_layout == NVCV_TENSOR_NCHW;
    m_size.h    = tDataAc->numRows();
    m_size.w    = tDataAc->numCols();
    m_numC      = tDataAc->numChannels();

    if (m_layout != NVCV_TENSOR_CHW && m_layout != NVCV_TENSOR_NCHW && m_layout != NVCV_TENSOR_HWC
        && m_layout != NVCV_TENSOR_NHWC)
    {
        throw std::runtime_error("Tensor layout unknown");
    }

    long sampleStride = tDataAc->sampleStride();

    if (tensorData.rank() == 3)
    {
        sampleStride = m_planar ? m_numC * m_size.h * m_rowStride : m_size.h * m_rowStride;
    }

    m_data.resize(sampleStride);

    if (cudaSuccess
        != cudaMemcpy(m_data.data(), tDataAc->sampleData(sampleIndex), sampleStride, cudaMemcpyDeviceToHost))
    {
        throw Exception(Status::ERROR_INTERNAL, "cudaMemcpy filed");
    }

    if (m_planar)
    {
        if (!nvcv::TensorDataAccessStridedImagePlanar::IsCompatible(tensorData))
            throw std::runtime_error("Tensor Data not compatible with Pitch Planar Access");

        auto tDataACp = nvcv::TensorDataAccessStridedImagePlanar::Create(tensorData);
        m_planeStride = tDataACp->planeStride();
    }

    return;
}

std::ostream &operator<<(std::ostream &out, const TensorImageData &cvImageData)
{
    out << "\n[H = " << cvImageData.m_size.h << " W = " << cvImageData.m_size.w << " C = " << cvImageData.m_numC
        << "]\n[planar = " << cvImageData.m_planar << " bytesPerC = " << cvImageData.m_bytesPerC
        << " rowStride = " << cvImageData.m_rowStride << " planeStride = " << cvImageData.m_planeStride
        << " sampleStride = " << cvImageData.m_data.size() << "]\n";

    if (!cvImageData.m_planar)
    {
        printPlane(&cvImageData.m_data[0], cvImageData.m_size.w, cvImageData.m_size.h, cvImageData.m_rowStride,
                   cvImageData.m_bytesPerC, cvImageData.m_numC);
    }
    else
    {
        for (int i = 0; i < cvImageData.m_numC; i++)
        {
            out << "\nPlane = " << i << "\n";
            printPlane(&cvImageData.m_data[cvImageData.m_planeStride * i], cvImageData.m_size.w, cvImageData.m_size.h,
                       cvImageData.m_rowStride, cvImageData.m_bytesPerC, 1);
        }
    }
    return out << "\n";
}

bool TensorImageData::operator==(const TensorImageData &that) const
{
    return ((this->m_size == that.m_size) && (this->m_rowStride == that.m_rowStride) && (this->m_numC == that.m_numC)
            && (this->m_planar == that.m_planar) && (this->m_bytesPerC == that.m_bytesPerC)
            && (this->m_layout == that.m_layout) && (this->m_data.size() == that.m_data.size())
            && (memcmp(this->m_data.data(), that.m_data.data(), this->m_data.size()) == 0));
}

bool TensorImageData::operator!=(const TensorImageData &that) const
{
    return !operator==(that);
}

nvcv::Tensor CreateTensor(int numImages, int imgWidth, int imgHeight, const nvcv::ImageFormat &imgFormat)
{
    //semi planar 420 NV12/21 format
    if (imgFormat == NVCV_IMAGE_FORMAT_NV12 || imgFormat == NVCV_IMAGE_FORMAT_NV12_ER
        || imgFormat == NVCV_IMAGE_FORMAT_NV21 || imgFormat == NVCV_IMAGE_FORMAT_NV21_ER)
    {
        // Width and height must be a multiple of 2 (i.e., even).
        if (imgHeight % 2 != 0 || imgWidth % 2 != 0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Invalid height or width: height and width need to be a "
                                  "multiple of 2 for planar and semi-planar YUV420 formats.");
        }

        // Tensor height is 3/2 times the image height to accommodate the half-height chroma planes.
        int height420 = (imgHeight * 3) / 2;

        if (numImages == 1)
        {
            return nvcv::Tensor(
                {
                    {height420, imgWidth, 1},
                    "HWC"
            },
                imgFormat.planeDataType(0).channelType(0));
        }
        else
        {
            // Note this tensor is being passed an YUV8 format, but the tensor is actually YUV420 however the tensor
            // class does not yet understand semi planar formats such as NV12/21 hence we just create an Y8 tensor with
            // modified height.
            return nvcv::Tensor(numImages, {imgWidth, height420}, nvcv::ImageFormat(NVCV_IMAGE_FORMAT_Y8));
        }
    }
    else if (imgFormat == NVCV_IMAGE_FORMAT_UYVY || imgFormat == NVCV_IMAGE_FORMAT_UYVY_ER
             || imgFormat == NVCV_IMAGE_FORMAT_YUYV || imgFormat == NVCV_IMAGE_FORMAT_YUYV_ER)
    {
        if (imgWidth % 2 != 0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Invalid width: width needs to be a multiple of 2 for interleaved YUV422 formats.");
        }

        int wdth422 = 2 * imgWidth; // Tensor width is 2x the image width to accomodate two chromaticity values for
                                    //   each group of two luma values (UYVY or YUYV).
        // clang-format off
        nvcv::DataType    type  = imgFormat.planeDataType(0).channelType(0);
        nvcv::TensorShape shape = numImages > 1 ? nvcv::TensorShape({numImages, imgHeight, wdth422, 1}, "NHWC")
                                                : nvcv::TensorShape(           {imgHeight, wdth422, 1},  "HWC");

        return nvcv::Tensor(shape, type);
        // clang-format on
    }
    if (numImages == 1)
    {
        int numChannels = imgFormat.numPlanes() == 1 ? imgFormat.planeNumChannels(0) : imgFormat.numPlanes();

        if (imgFormat.numPlanes() > 1)
        {
            return nvcv::Tensor(
                {
                    {numChannels, imgHeight, imgWidth},
                    "CHW"
            },
                imgFormat.planeDataType(0).channelType(0));
        }
        else
        {
            return nvcv::Tensor(
                {
                    {imgHeight, imgWidth, numChannels},
                    "HWC"
            },
                imgFormat.planeDataType(0).channelType(0));
        }
    }

    assert(numImages > 1);
    return nvcv::Tensor(numImages, {imgWidth, imgHeight}, imgFormat);
}

static void GetImageByteVectorFromTensorPlanar(const TensorData &tensorData, int sample,
                                               std::vector<nvcv::Byte> &outData)
{
    Optional<TensorDataAccessStridedImagePlanar> tDataAc = nvcv::TensorDataAccessStridedImagePlanar::Create(tensorData);

    if (!tDataAc)
        throw std::runtime_error("Tensor Data not compatible with planar access.");

    if (tDataAc->numSamples() <= sample || sample < 0)
        throw std::runtime_error("Number of samples smaller than requested sample.");

    // in a planar tensor the dtype represents each plane so the total bytes per pixel must be calculated
    int bytesPerC       = tDataAc->dtype().bitsPerPixel() / 8;
    int outputSizeBytes = tDataAc->numRows() * tDataAc->numCols() * bytesPerC * tDataAc->numChannels();

    // Make sure we have the right size.
    outData.resize(outputSizeBytes);
    Byte  *basePtr  = tDataAc->sampleData(sample);
    size_t dstWidth = tDataAc->numCols() * bytesPerC;
    for (int i = 0; i < tDataAc->numChannels(); ++i)
    {
        if (cudaSuccess
            != cudaMemcpy2D(outData.data() + (i * (tDataAc->numCols() * tDataAc->numRows()) * bytesPerC), dstWidth,
                            basePtr, tDataAc->rowStride(), dstWidth, tDataAc->numRows(), cudaMemcpyDeviceToHost))
        {
            throw std::runtime_error("CudaMemcpy failed on copy of channel plane from device to host.");
        }
        basePtr += tDataAc->planeStride();
    }
    return;
}

void GetImageByteVectorFromTensor(const TensorData &tensorData, int sample, std::vector<nvcv::Byte> &outData)
{
    Optional<TensorDataAccessStridedImage> tDataAc = nvcv::TensorDataAccessStridedImage::Create(tensorData);

    if (!tDataAc)
        throw std::runtime_error("Tensor Data not compatible with pitch access.");
    if (tDataAc->infoLayout().isChannelFirst())
        return GetImageByteVectorFromTensorPlanar(tensorData, sample, outData);

    if (tDataAc->numSamples() <= sample || sample < 0)
        throw std::runtime_error("Number of samples smaller than requested sample.");

    int bytesPerPixel   = (tDataAc->dtype().bitsPerPixel() / 8) * tDataAc->numChannels();
    int outputSizeBytes = tDataAc->numRows() * tDataAc->numCols() * bytesPerPixel;

    // Make sure we have the right size.
    outData.resize(outputSizeBytes);

    if (cudaSuccess
        != cudaMemcpy2D(outData.data(), tDataAc->numCols() * bytesPerPixel, tDataAc->sampleData(sample),
                        tDataAc->rowStride(), tDataAc->numCols() * bytesPerPixel, tDataAc->numRows(),
                        cudaMemcpyDeviceToHost))
    {
        throw std::runtime_error("CudaMemcpy failed");
    }
    return;
}

static void SetImageTensorFromByteVectorPlanar(const TensorData &tensorData, std::vector<nvcv::Byte> &data, int sample)
{
    Optional<TensorDataAccessStridedImagePlanar> tDataAc = nvcv::TensorDataAccessStridedImagePlanar::Create(tensorData);

    if (!tDataAc)
        throw std::runtime_error("Tensor Data not compatible with planar image access.");

    if (tDataAc->numSamples() <= sample)
        throw std::runtime_error("Number of samples smaller than requested sample.");

    if ((int64_t)data.size()
        != tDataAc->numCols() * tDataAc->numRows() * (tDataAc->dtype().bitsPerPixel() / 8) * tDataAc->numChannels())
        throw std::runtime_error("Data vector is incorrect size, size must be W*H*bytesPerPixel.");

    int bytesPerC = (tDataAc->dtype().bitsPerPixel() / 8);

    auto copyToGpu = [&](int j)
    {
        Byte *basePtr = tDataAc->sampleData(j);

        for (int i = 0; i < tDataAc->numChannels(); ++i)
        {
            Byte  *srcPtr        = data.data() + (i * (tDataAc->numCols() * tDataAc->numRows() * bytesPerC));
            size_t srcPitch      = tDataAc->numCols() * bytesPerC;
            size_t srcWidthBytes = tDataAc->numCols() * bytesPerC;
            if (cudaSuccess
                != cudaMemcpy2D(basePtr, tDataAc->rowStride(), srcPtr, srcPitch, srcWidthBytes, tDataAc->numRows(),
                                cudaMemcpyHostToDevice))
            {
                throw std::runtime_error("CudaMemcpy failed for channel plane copy from host to device.");
            }
            basePtr += tDataAc->planeStride();
        }
    };

    if (sample < 0)
        for (auto i = 0; i < tDataAc->numSamples(); ++i)
        {
            copyToGpu(i);
        }
    else
        copyToGpu(sample);
}

void SetImageTensorFromByteVector(const TensorData &tensorData, std::vector<nvcv::Byte> &data, int sample)
{
    Optional<TensorDataAccessStridedImage> tDataAc = nvcv::TensorDataAccessStridedImage::Create(tensorData);

    if (!tDataAc)
        throw std::runtime_error("Tensor Data not compatible with pitch access.");

    if (tDataAc->infoLayout().isChannelFirst()) // planar case
        return SetImageTensorFromByteVectorPlanar(tensorData, data, sample);

    if (tDataAc->numSamples() <= sample)
        throw std::runtime_error("Number of samples smaller than requested sample.");

    if ((int64_t)data.size()
        != tDataAc->numCols() * tDataAc->numRows() * (tDataAc->dtype().bitsPerPixel() / 8) * tDataAc->numChannels())
        throw std::runtime_error("Data vector is incorrect size, size must be N*W*sizeof(pixel).");

    int bytesPerC = (tDataAc->dtype().bitsPerPixel() / 8);

    auto copyToGpu = [&](int i)
    {
        Byte  *basePtr       = tDataAc->sampleData(i);
        Byte  *srcPtr        = data.data();
        size_t srcPitch      = tDataAc->numCols() * bytesPerC * tDataAc->numChannels();
        size_t srcWidthBytes = tDataAc->numCols() * bytesPerC * tDataAc->numChannels();

        if (cudaSuccess
            != cudaMemcpy2D(basePtr, tDataAc->rowStride(), srcPtr, srcPitch, srcWidthBytes, tDataAc->numRows(),
                            cudaMemcpyHostToDevice))
        {
            throw std::runtime_error("CudaMemcpy failed on copy of image from host to device.");
        }
    };

    if (sample < 0)
        for (auto i = 0; i < tDataAc->numSamples(); ++i)
        {
            copyToGpu(i);
        }
    else
        copyToGpu(sample);
}

} // namespace nvcv::util
