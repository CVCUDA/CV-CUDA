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

#ifndef NVCV_TEST_COMMON_TENSOR_DATA_UTILS_HPP
#define NVCV_TEST_COMMON_TENSOR_DATA_UTILS_HPP

#include <cuda_runtime.h>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/Assert.h>

#include <array>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

namespace nvcv::util {

class CudaMemcpyError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class TensorDataUtilsError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

enum chflags
{
    C0 = 0x1 << 0,
    C1 = 0x1 << 1,
    C2 = 0x1 << 2,
    C3 = 0x1 << 3
};

// Generic ValueAt can be used with any tensor data with any strides and coordinate to access
// E.g.:
//   std::vector<uint8_t> vec(...);
//   long3 strides{...};
//   int3 coord{...};
//   ValueAt<int4>(vec, strides, coord) = 0;
template<typename T, class VecType, typename ST, typename CT,
         typename RT = std::conditional_t<std::is_const_v<VecType>, const T, T>>
inline RT &ValueAt(VecType &vec, const ST &strides, const CT &coord) requires nvcv::cuda::detail::IsSameCompound<ST, CT>
{
    return *reinterpret_cast<RT *>(&vec[nvcv::cuda::dot(coord, strides)]);
}

// Holds a single image copied from an ITensor in host memory
// (N)CHW and (N)HWC tensors are supported but only 1 sample is held.
class TensorImageData
{
public:
    TensorImageData() = delete;
    explicit TensorImageData(const TensorData &tensorData, int sampleIndex = 0);

    // H/W in logical pixels where byte offset == m_size.x * bytesPerPixel.
    const Size2D &size() const
    {
        return m_size;
    };

    // Returns the size in bytes of the color component.
    const int32_t &bytesPerC() const
    {
        return m_bytesPerC;
    };

    // Returns the number of color components.
    int numC() const
    {
        return m_numC;
    };

    // Returns the row pitch in bytes.
    int64_t rowStride() const
    {
        return m_rowStride;
    };

    // Returns plane pitch in bytes
    int64_t planeStride() const
    {
        return m_planeStride;
    };

    // Returns true if the image is planar, false if the image is HWC.
    bool imageCHW() const
    {
        return m_planar;
    };

    /**
     * Returns the underling data vector representing the bytes copied from the ITensor.
     */
    std::vector<uint8_t> &getVector()
    {
        return m_data;
    };

    /**
     * Compares the two TensorImageData classes including data stored in buffer.
     */
    bool operator==(const TensorImageData &that) const;

    /**
     * Compares if two TensorImageData classes including data stored in buffer are not equal.
     */
    bool operator!=(const TensorImageData &that) const;

    /**
     * Prints out the contents of the byte data contained in the buffer
     * Data is formatted as follows:
     *
     * ex.
     * [H = 2 W = 2 C = 3]
     * [planar = 0 bytesPerC = 1 rowStride = 32 planeStride = 0 sampleStride = 64]
     * [aa, bb, cc | aa, bb, cc] 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
     * [aa, bb, cc | aa, bb, cc] 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
     *
     * KEY:
     * H = Height in Rows (x dim)
     * W = Width in Cols  (y dim)
     * C = Colors in a pixel, H*W define number of pixels
     * planar = if 1 data is CHW if 0 data is HWC
     * bytesPerC = Bytes per Color
     * rowStride = Bytes per Row, HWC rows will include all Color components.
     * planeStride = Number of bytes in a CHW plane will be 0 in a HWC Tensor.
     *
     *
     * [aa, XX, XX | XX, XX, XX] 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
     * [XX, XX, XX | XX, XX, bb] 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
     *
     * aa = val @ C = 0, x = 0, y = 0
     * bb = val @ C = 2, x = 1, y = 1
     * H = 2
     * W = 2
     * C = 3
     * | = Pixel boundary
     * [] = logical image boundary, bytes outside of this are for alignment and considered part of pitchBytes
     * \n in print-out indicates end of rowBytesPitch
     *
     * If CHW, plane = X will indicate the color plane
     *
     */
    friend std::ostream &operator<<(std::ostream &out, const TensorImageData &cvImageData);

    /**
     * Returns a pointer of type T to the CHW/HWC data at location X,Y,C, this pointer can be used to read and write data.
     * Function does not do bounds checking on X/Y/C and TYPE T.
     * However the function will throw if the total offset is past
     * the bounds of the internal buffer holding the data.
     *
     * @param[in] x The column [0 to MAX] of the HWC or CHW data stored from ITensor
     *
     * @param[in] y The column [0 to MAX] of the HWC or CHW data stored from ITensor
     *
     * @param[in] c The color component [0 to MAX], typically [0 ... 3]
     *
     * @retval T* pointer to the type at x,y,c cords of the CHW or HWC buffer.
     *
     *           Function will throw if total offset is outside the bounds of the internal buffer
     */
    template<class T>
    T *item(const int x, const int y, const int c)
    {
        size_t byteIndex = 0;

        if (!m_planar)
            byteIndex = static_cast<size_t>((c * m_bytesPerC) + (x * m_bytesPerC * m_numC) + (y * m_rowStride));
        else
            byteIndex = static_cast<size_t>((c * m_planeStride) + (x * m_bytesPerC) + (y * m_rowStride));

        if (byteIndex >= m_data.size())
            throw TensorDataUtilsError("Requested data out of bounds");

        return reinterpret_cast<T *>(m_data.data() + byteIndex);
    }

private:
    std::vector<uint8_t> m_data;            // pointer to local data
    Size2D               m_size;            // h/w in logical pixels, byte offset == m_size.x * bytesPerPixel.
    int64_t              m_rowStride;       // Row stride in bytes
    int64_t              m_planeStride = 0; // used for (n)CHW Tensors 0 if not CHW
    int                  m_numC;            // Number of color channels usually 1,3,4 (Y, RGB, ARGB)
    bool                 m_planar;          // If true the image is (n)CHW
    int32_t              m_bytesPerC;       // bytes per logical pixels
    NVCVTensorLayout     m_layout;          // layout of originating ITensor NVCV_TENSOR_CHW/NVCV_TENSOR_NHWC/HWC
};

/**
 * Create either HWC or NHWC Tensor with given parameters.
 *
 * @param[in] numImages Number of images, if 1 creates a HWC tensor, else creates a NHWC tensor.
 * @param[in] imgWidth Image width inside the tensor.
 * @param[in] imgHeight Image height inside the tensor.
 * @param[in] imgFormat Image format inside the tensor.
 *
 */
nvcv::Tensor CreateTensor(int numImages, int imgWidth, int imgHeight, const nvcv::ImageFormat &imgFormat);

/**
 * Writes over the Tensor data with type DT and value of @data.
 * Function does not do data type or underflow checking if
 * the passed in type does not match the type the tensor
 * was created with. Writes over all samples stored in the
 * tensor.
 *
 * @param[in,out] tensorData created tensor object.
 *
 * @param[in] data the data to set the tensor to.
 *
 * @param[in] sample optional the sample to write to if -1 all samples are written
 */
template<typename DT>
static void SetTensorTo(const TensorData &tensorData, DT data, int sample = -1);

/**
 * Writes over the Tensor data with type DT and random data values.
 * Function does not do data type or underflow checking if
 * the passed in type does not match the type the tensor
 * was created with. Writes over all samples stored in the
 * tensor.
 *
 * @param[in,out] tensorData created tensor object.
 *
 * @param[in] minVal minimum value of the random generated value (inclusive).
 *
 * @param[in] maxVal maximum value of the random generated value (inclusive).
 *
 * @param[in] sample optional the sample to write to if -1 all samples are written
 */
template<typename DT>
static void SetTensorToRandomValue(const TensorData &tensorData, DT minVal, DT maxVal, int sample = -1);

/**
 * Writes over the Tensor data with an array of type DT array must be
 * the size of H*W*C with DT matching the Tensor datatype. Function does not do data type checking.
 *
 * @param[in,out] tensorData created tensor object.
 *
 * @param[in] data the data to set the tensor to.
 *
 * @param[in] sample optional the sample to write to if -1 all samples are written
 */
template<typename DT>
static void SetTensorFromVector(const TensorData &tensorData, std::vector<DT> &data, int sample = -1);

/**
 * Sets the TensorData to the values contained in the data parameter.
 * The data parameter must contain all of the data for the image, however it should
 * not include any padding. Also the DT must be data type contained in the TensorImageData.
 * Data should be size of Width*Height*NumChannels.
 *
 * @param[in,out] tensorData TensorImageData object.
 *
 * @param[in] data vector of data to set the image to.
 *
 * @param[in] sample sample number to set the vector to, -1 indicates all samples.
 *
 */
template<typename DT>
static void SetImageTensorFromVector(const TensorData &tensorData, std::vector<DT> &data, int sample = -1);

/**
 * Writes over the Tensor data with an byte array, array must be
 * the size of H*W*C*bytesPerC. Function does not do data type checking.
 *
 * @param[in,out] tensorData created tensor object.
 *
 * @param[in] data vector of bytes with the data to set the tensor to.
 *
 * @param[in] sample optional the sample to write to if -1 all samples are written
 */
void SetImageTensorFromByteVector(const TensorData &tensorData, std::vector<nvcv::Byte> &data, int sample = -1);

/**
 * Returns a vector contains the values of the provided sample including any padding. This function assumes that the DT data type
 * matches the datatype in the TensorData.
 *
 * @param[in] tensorData created tensor object.
 *
 * @param[in] sample the sample to copy to vector 0 index.
 *
 * @param[out] outData the data to set the tensor to.
 *
 */
template<typename DT>
static void GetVectorFromTensor(const TensorData &tensorData, int sample, std::vector<DT> &outData);

/**
 * Returns a vector contains the values of the provided sample. This vector will only contain
 * the values of the image and not any padding/stride. This function assumes that the DT data type
 * matches the datatype in the TensorData.
 *
 * @param[in] tensorData created tensor object.
 *
 * @param[in] sample the sample to copy to vector 0 index.
 *
 * @param[out] outData the data to set the tensor to.
 *
 */
template<typename DT>
static void GetImageVectorFromTensor(const TensorData &tensorData, int sample, std::vector<DT> &outData);

/**
 * Returns a byte vector which contains the values of the specified sample. This vector will only contain
 * the values of the image and not any padding/stride. Also this will return a byte array regardless of
 * the DataType of the tensor. The byte vector returned will be the size of H*W*C*bytesPerC.
 *
 * @param[in] tensorData created tensor object.
 *
 * @param[in] sample the sample to copy to vector 0 index.
 *
 * @param[out] outData the data to set the tensor to.
 *
 */
void GetImageByteVectorFromTensor(const TensorData &tensorData, int sample, std::vector<nvcv::Byte> &outData);

/**
 * Sets the TensorImageData to the value set by the data parameter
 * region defines the amount of image to set starting at 0,0
 *
 * @param[in] cvImg TensorImageData object.
 *
 * @param[in] data data to set the image to.
 *
 * @param[in] region region in which to set the data to.
 *
 * @param[out] chFlags bitmask indicating which color channels to set.
 *
 */
template<typename DT>
static void SetCvDataTo(TensorImageData &cvImg, DT data, Size2D region, uint8_t chFlags);

/**
 * @brief Prints a image from a byte vector, useful for debugging does not check bounds on the passed in data.
 *
 * @param[in] data        Pointer to the image data.
 * @param[in] width       Width of the image in pixels.
 * @param[in] height      Height of the image in pixels.
 * @param[in] rowStride   Number of bytes each row of the image uses/
 * @param[in] bytesPC     Number of bytes per channel. i.e 1 for a uint8_t etc
 * @param[in] numC        Number of color channels in the image. i.e 3 for an RGB image.
 * @param[in] planar      If true, data is in planar format; if false, data is in interleaved format.
 */
void PrintImageFromByteVector(const uint8_t *data, int width, int height, int rowStride, int bytesPC, int numC,
                              bool planar);

template<typename DT>
static void SetTensorTo(const TensorData &tensorData, DT data, int sample)
{
    if (!nvcv::TensorDataAccessStrided::IsCompatible(tensorData))
        throw TensorDataUtilsError("Tensor Data is not pitch access capable.");

    auto tDataAc = nvcv::TensorDataAccessStrided::Create(tensorData);

    if (tDataAc->numSamples() <= sample)
        throw TensorDataUtilsError("Number of samples smaller than requested sample.");

    auto            inElements = static_cast<int>(tDataAc->sampleStride() / sizeof(DT));
    std::vector<DT> srcVec(inElements, data);

    int totalSamples;
    if (sample < 0)
    {
        totalSamples = static_cast<int>(tDataAc->numSamples());
        sample       = 0;
    }
    else
    {
        totalSamples = sample + 1;
    }

    for (int i = sample; i < totalSamples; ++i)
    {
        auto  *outSamplePtr = tDataAc->sampleData(i);
        size_t size         = tDataAc->sampleStride();
        if (auto err = cudaMemcpy(outSamplePtr, srcVec.data(), size, cudaMemcpyHostToDevice))
        {
            throw CudaMemcpyError(std::string("CudaMemcpy failed with ") + cudaGetErrorName(err) + " ("
                                  + std::to_string(err) + "): " + cudaGetErrorString(err));
        }
    }

    return;
}

template<typename DT>
static void SetTensorToRandomValueFloat(const TensorData &tensorData, DT minVal, DT maxVal, int sample)
{
    if (!nvcv::TensorDataAccessStrided::IsCompatible(tensorData))
        throw TensorDataUtilsError("Tensor Data is not pitch access capable.");

    auto tDataAc = nvcv::TensorDataAccessStrided::Create(tensorData);

    if (tDataAc->numSamples() <= sample)
        throw TensorDataUtilsError("Number of samples smaller than requested sample.");

    auto                       inElements = static_cast<int>(tDataAc->sampleStride() / sizeof(DT));
    std::vector<DT>            srcVec(inElements);
    std::default_random_engine randEng(0);

    int totalSamples;
    if (sample < 0)
    {
        totalSamples = static_cast<int>(tDataAc->numSamples());
        sample       = 0;
    }
    else
    {
        totalSamples = sample + 1;
    }

    std::uniform_real_distribution<> srcRand(minVal, maxVal);
    for (int i = sample; i < totalSamples; ++i)
    {
        std::generate( // NOSONAR: std::ranges::generate is C++20.
            srcVec.begin(), srcVec.end(), [&srcRand, &randEng]() { return srcRand(randEng); });
        if (cudaSuccess
            != cudaMemcpy(tDataAc->sampleData(i), srcVec.data(), tDataAc->sampleStride(), cudaMemcpyHostToDevice))
        {
            throw TensorDataUtilsError("CudaMemcpy failed");
        }
    }
    return;
}

template<>
inline void SetTensorToRandomValue<float>(const TensorData &tensorData, float minVal, float maxVal, int sample)
{
    SetTensorToRandomValueFloat<float>(tensorData, minVal, maxVal, sample);
}

template<>
inline void SetTensorToRandomValue<double>(const TensorData &tensorData, double minVal, double maxVal, int sample)
{
    SetTensorToRandomValueFloat<double>(tensorData, minVal, maxVal, sample);
}

template<typename DT>
static void SetTensorToRandomValue(const TensorData &tensorData, DT minVal, DT maxVal, int sample)
{
    if (!nvcv::TensorDataAccessStrided::IsCompatible(tensorData))
        throw TensorDataUtilsError("Tensor Data is not pitch access capable.");

    auto tDataAc = nvcv::TensorDataAccessStrided::Create(tensorData);

    if (tDataAc->numSamples() <= sample)
        throw TensorDataUtilsError("Number of samples smaller than requested sample.");

    auto                       inElements = static_cast<int>(tDataAc->sampleStride() / sizeof(DT));
    std::vector<DT>            srcVec(inElements);
    std::default_random_engine randEng(0);

    int totalSamples;
    if (sample < 0)
    {
        totalSamples = static_cast<int>(tDataAc->numSamples());
        sample       = 0;
    }
    else
    {
        totalSamples = sample + 1;
    }
    std::uniform_int_distribution<DT> srcRand{minVal, maxVal};
    for (int i = sample; i < totalSamples; ++i)
    {
        std::generate( // NOSONAR: std::ranges::generate is C++20.
            srcVec.begin(), srcVec.end(), [&srcRand, &randEng]() { return srcRand(randEng); });
        if (cudaSuccess
            != cudaMemcpy(tDataAc->sampleData(i), srcVec.data(), tDataAc->sampleStride(), cudaMemcpyHostToDevice))
        {
            throw TensorDataUtilsError("CudaMemcpy failed");
        }
    }

    return;
}

template<typename DT>
static void SetTensorFromVector(const TensorData &tensorData, std::vector<DT> &data, int sample)
{
    if (!nvcv::TensorDataAccessStrided::IsCompatible(tensorData))
        throw TensorDataUtilsError("Tensor Data is not pitch access capable.");

    auto tDataAc = nvcv::TensorDataAccessStrided::Create(tensorData);

    if ((int64_t)(data.size() * sizeof(DT)) != tDataAc->sampleStride())
        throw TensorDataUtilsError("Data vector is incorrect size.");

    if (tDataAc->numSamples() <= sample)
        throw TensorDataUtilsError("Number of samples smaller than requested sample.");

    if (sample < 0)
    {
        for (int i = 0; i < tDataAc->numSamples(); ++i)
        {
            if (cudaSuccess
                != cudaMemcpy(tDataAc->sampleData(i), data.data(), tDataAc->sampleStride(), cudaMemcpyHostToDevice))
            {
                throw TensorDataUtilsError("CudaMemcpy failed");
            }
        }
    }
    else
    {
        if (cudaSuccess
            != cudaMemcpy(tDataAc->sampleData(sample), data.data(), tDataAc->sampleStride(), cudaMemcpyHostToDevice))
        {
            throw TensorDataUtilsError("CudaMemcpy failed");
        }
    }

    return;
}

template<typename DT>
static void GetVectorFromTensor(const TensorData &tensorData, int sample, std::vector<DT> &outData)
{
    if (!nvcv::TensorDataAccessStrided::IsCompatible(tensorData))
        throw TensorDataUtilsError("Tensor Data is not pitch access capable.");

    auto tDataAc = nvcv::TensorDataAccessStrided::Create(tensorData);

    if (tDataAc->numSamples() <= sample || sample < 0)
        throw TensorDataUtilsError("Number of samples smaller than requested sample.");

    auto elements = static_cast<int>(tDataAc->sampleStride() / sizeof(DT));

    outData.resize(elements);

    if (cudaSuccess
        != cudaMemcpy(outData.data(), tDataAc->sampleData(sample), tDataAc->sampleStride(), cudaMemcpyDeviceToHost))
    {
        throw TensorDataUtilsError("CudaMemcpy failed");
    }

    return;
}

template<typename DT>
static void SetImageTensorFromVectorPlanar(const TensorData &tensorData, std::vector<DT> &data, int sample)
{
    Optional<TensorDataAccessStridedImagePlanar> tDataAc = nvcv::TensorDataAccessStridedImagePlanar::Create(tensorData);

    if (!tDataAc)
        throw TensorDataUtilsError("Tensor Data not compatible with planar image access.");

    if (tDataAc->numSamples() <= sample)
        throw TensorDataUtilsError("Number of samples smaller than requested sample.");

    if ((int64_t)data.size() != tDataAc->numCols() * tDataAc->numRows() * tDataAc->numChannels())
        throw TensorDataUtilsError("Data vector is incorrect size, size must be W*C*sizeof(DT)*channels.");

    auto copyToGpu = [&tDataAc, &data](int j)
    {
        Byte *basePtr = tDataAc->sampleData(j);
        for (int i = 0; i < tDataAc->numChannels(); ++i)
        {
            if (cudaSuccess
                != cudaMemcpy2D(basePtr, tDataAc->rowStride(),
                                data.data() + (i * (tDataAc->numCols() * tDataAc->numRows())),
                                tDataAc->numCols() * sizeof(DT), tDataAc->numCols() * sizeof(DT), tDataAc->numRows(),
                                cudaMemcpyHostToDevice))
            {
                throw TensorDataUtilsError("CudaMemcpy failed for channel plane copy from host to device.");
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

template<typename DT>
static void SetImageTensorFromVector(const TensorData &tensorData, std::vector<DT> &data, int sample)
{
    Optional<TensorDataAccessStridedImage> tDataAc = nvcv::TensorDataAccessStridedImage::Create(tensorData);

    if (!tDataAc)
        throw TensorDataUtilsError("Tensor Data not compatible with pitch access.");

    if (tDataAc->infoLayout().isChannelFirst()) // planar case
        return SetImageTensorFromVectorPlanar<DT>(tensorData, data, sample);

    if (tDataAc->numSamples() <= sample)
        throw TensorDataUtilsError("Number of samples smaller than requested sample.");

    if ((int64_t)data.size() != tDataAc->numCols() * tDataAc->numRows() * tDataAc->numChannels())
        throw TensorDataUtilsError("Data vector is incorrect size, size must be N*W*C*sizeof(DT).");

    auto copyToGpu = [&tDataAc, &data](int i)
    {
        Byte *basePtr = tDataAc->sampleData(i);
        if (cudaSuccess
            != cudaMemcpy2D(
                basePtr, tDataAc->rowStride(), data.data(), tDataAc->numCols() * tDataAc->numChannels() * sizeof(DT),
                tDataAc->numCols() * tDataAc->numChannels() * sizeof(DT), tDataAc->numRows(), cudaMemcpyHostToDevice))
        {
            throw TensorDataUtilsError("CudaMemcpy failed on copy of image from host to device.");
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

template<typename DT>
static void GetImageVectorFromTensorPlanar(const TensorData &tensorData, int sample, std::vector<DT> &outData)
{
    Optional<TensorDataAccessStridedImagePlanar> tDataAc = nvcv::TensorDataAccessStridedImagePlanar::Create(tensorData);

    if (!tDataAc)
        throw TensorDataUtilsError("Tensor Data not compatible with planar access.");

    if (tDataAc->numSamples() <= sample || sample < 0)
        throw TensorDataUtilsError("Number of samples smaller than requested sample.");

    int elements = tDataAc->numRows() * tDataAc->numCols() * tDataAc->numChannels();

    // Make sure we have the right size.
    outData.resize(elements);
    Byte *basePtr = tDataAc->sampleData(sample);
    for (int i = 0; i < tDataAc->numChannels(); ++i)
    {
        if (cudaSuccess
            != cudaMemcpy2D(outData.data() + (i * (tDataAc->numCols() * tDataAc->numRows())),
                            tDataAc->numCols() * sizeof(DT), basePtr, tDataAc->rowStride(),
                            tDataAc->numCols() * sizeof(DT), tDataAc->numRows(), cudaMemcpyDeviceToHost))
        {
            throw TensorDataUtilsError("CudaMemcpy failed on copy of channel plane from device to host.");
        }
        basePtr += tDataAc->planeStride();
    }
    return;
}

// sets the tensor data to the value of data, but honors strides.
template<typename DT>
static void GetImageVectorFromTensor(const TensorData &tensorData, int sample, std::vector<DT> &outData)
{
    Optional<TensorDataAccessStridedImage> tDataAc = nvcv::TensorDataAccessStridedImage::Create(tensorData);

    if (!tDataAc)
        throw TensorDataUtilsError("Tensor Data not compatible with pitch access.");
    if (tDataAc->infoLayout().isChannelFirst())
        return GetImageVectorFromTensorPlanar<DT>(tensorData, sample, outData);

    if (tDataAc->numSamples() <= sample || sample < 0)
        throw TensorDataUtilsError("Number of samples smaller than requested sample.");

    int elements = tDataAc->numRows() * tDataAc->numCols() * tDataAc->numChannels();

    // Make sure we have the right size.
    outData.resize(elements);

    if (cudaSuccess
        != cudaMemcpy2D(outData.data(), tDataAc->numCols() * sizeof(DT) * tDataAc->numChannels(),
                        tDataAc->sampleData(sample), tDataAc->rowStride(),
                        tDataAc->numCols() * sizeof(DT) * tDataAc->numChannels(), tDataAc->numRows(),
                        cudaMemcpyDeviceToHost))
    {
        throw TensorDataUtilsError("CudaMemcpy failed");
    }
    return;
}

template<typename DT>
static void SetCvDataPixelChannels(TensorImageData &cvImg, DT data, int x, int y, uint8_t chFlags)
{
    for (int c = 0; c < 4; c++)
    {
        if (((chFlags >> c) & 0x1) == 0x1) // NOSONAR: chFlags is a channel bitmask, not byte storage.
        {
            *cvImg.item<DT>(x, y, c) = data;
        }
    }
}

template<typename DT>
static void SetCvDataTo(TensorImageData &cvImg, DT data, Size2D region, uint8_t chFlags)
{
    for (int x = 0; x < region.w; x++)
    {
        for (int y = 0; y < region.h; y++)
        {
            SetCvDataPixelChannels(cvImg, data, x, y, chFlags);
        }
    }

    return;
}

// Useful for debugging
template<typename ST>
inline nvcv::cuda::BaseType<ST> BufferShapeElementOrOne(const ST &shape, int element)
{
    using BT = nvcv::cuda::BaseType<ST>;

    return nvcv::cuda::NumElements<ST> > element ? nvcv::cuda::GetElement(shape, element) : BT{1};
}

template<typename VT, typename ST>
inline void PrintBufferValues(const std::vector<uint8_t> &vec, const ST &strides, const ST &shape,
                              nvcv::cuda::BaseType<ST> x, nvcv::cuda::BaseType<ST> y, nvcv::cuda::BaseType<ST> z,
                              uint32_t endls)
{
    using BT  = nvcv::cuda::BaseType<ST>;
    using BT4 = nvcv::cuda::MakeType<BT, 4>;
    using CVT = std::conditional_t<sizeof(VT) == 1, int, VT>;

    std::cout << " " << std::flush;
    for (BT w = 0; w < BufferShapeElementOrOne(shape, 3); ++w)
    {
        ST coord = nvcv::cuda::DropCast<nvcv::cuda::NumElements<ST>>(BT4{x, y, z, w});

        std::cout << " " << static_cast<CVT>(ValueAt<VT>(vec, strides, coord)) << std::flush;
    }

    if (endls & 0b0010)
        std::cout << std::endl;
    else
        std::cout << std::flush;
}

template<typename VT, typename ST>
inline void PrintBufferRows(const std::vector<uint8_t> &vec, const ST &strides, const ST &shape,
                            nvcv::cuda::BaseType<ST> x, nvcv::cuda::BaseType<ST> y, uint32_t endls)
{
    using BT = nvcv::cuda::BaseType<ST>;

    if (endls & 0b0100)
        std::cout << "  [" << std::endl;
    else
        std::cout << "  [" << std::flush;

    for (BT z = 0; z < BufferShapeElementOrOne(shape, 2); ++z)
    {
        PrintBufferValues<VT>(vec, strides, shape, x, y, z, endls);
    }

    if (endls & 0b0001)
        std::cout << "  ]" << std::endl;
    else
        std::cout << "  ]" << std::flush;
}

template<typename VT, typename ST>
inline void PrintBuffer(const std::vector<uint8_t> &vec, const ST &strides, const ST &shape, const char *name = "",
                        uint32_t endls = 0b1111)
{
    using BT = nvcv::cuda::BaseType<ST>;

    std::cout << "I Printing buffer " << name << " with:\nI\tSize = " << vec.size() << " Bytes\nI\tShape = " << shape
              << "\nI\tStrides = " << strides << "\nI\tValues = " << std::endl;

    for (BT x = 0; x < BufferShapeElementOrOne(shape, 0); ++x)
    {
        if (endls & 0b1000)
            std::cout << "{" << std::endl;
        else
            std::cout << "{" << std::flush;
        for (BT y = 0; y < BufferShapeElementOrOne(shape, 1); ++y)
        {
            PrintBufferRows<VT>(vec, strides, shape, x, y, endls);
        }
        std::cout << "}" << std::endl;
    }
}

// Write images in *HW tensor buffer vec to PGM files.
// The file name provided should have two (one) "%ld" format substr to place the first two (one) indices.
// The value type VT is converted to U8 when writing to each PGM file.
template<typename ST>
inline ST StripPGMCoord(long4_16a coord)
{
    if constexpr (nvcv::cuda::NumElements<ST> == 4)
        return ST{coord};
    else if constexpr (nvcv::cuda::NumElements<ST> == 3)
        return ST{coord.y, coord.z, coord.w};
    return ST{coord.z, coord.w};
}

template<typename VT>
inline VT ConvertPGMValue(VT val)
{
    if constexpr (std::is_same_v<VT, uint8_t>)
        return val;
    else if constexpr (std::is_integral_v<VT> && !std::is_signed_v<VT>)
        return std::min((VT)255, std::max((VT)0, val));
    else if constexpr (std::is_integral_v<VT> && std::is_signed_v<VT>)
        return std::min((VT)255, std::max((VT)0, (VT)std::abs(val)));
    else
        return std::min((VT)255, std::max((VT)0, (VT)std::round(std::abs(val))));
}

template<typename VT, typename ST>
inline void WritePGMValues(std::ofstream &ofs, const std::vector<uint8_t> &vec, const ST &strides, long c0, long c1,
                           int width, int height)
{
    for (long i = 0; i < height; ++i)
    {
        for (long j = 0; j < width; ++j)
        {
            ST coord = StripPGMCoord<ST>(long4_16a{c0, c1, i, j});
            VT val   = util::ValueAt<VT>(vec, strides, coord);

            ofs << ConvertPGMValue(val) << ((j == width - 1) ? "\n" : " ");
        }
    }
}

template<typename VT, typename ST>
inline void WriteOnePGMImage(const char *filename, const std::vector<uint8_t> &vec, const ST &strides, long c0, long c1,
                             int width, int height)
{
    std::array<char, 256> fn;
    int                   numChars = std::snprintf(fn.data(), fn.size(), filename, c1, c0);
    NVCV_ASSERT(numChars >= 0 && static_cast<size_t>(numChars) < fn.size());

    std::ofstream ofs(fn.data());

    ofs << "P2\n" << width << " " << height << " 255\n";

    WritePGMValues<VT>(ofs, vec, strides, c0, c1, width, height);

    ofs.close();
}

template<typename VT, typename ST>
inline void WriteImagesToPGM(const char *filename, const std::vector<uint8_t> &vec, const ST &strides, const ST &shape)
{
    static_assert(nvcv::cuda::NumElements<ST> >= 2);

    int widthIdx  = nvcv::cuda::NumElements<ST> - 1;
    int heightIdx = nvcv::cuda::NumElements<ST> - 2;
    int width     = nvcv::cuda::GetElement(shape, widthIdx);
    int height    = nvcv::cuda::GetElement(shape, heightIdx);

    int c0size = 1;
    int c1size = 1;
    if constexpr (nvcv::cuda::NumElements<ST> == 4)
    {
        c0size = nvcv::cuda::GetElement(shape, 0);
        c1size = nvcv::cuda::GetElement(shape, 1);
    }
    else if constexpr (nvcv::cuda::NumElements<ST> == 3)
    {
        c1size = nvcv::cuda::GetElement(shape, 0);
    }

    for (long c0 = 0; c0 < c0size; ++c0)
    {
        for (long c1 = 0; c1 < c1size; ++c1)
        {
            WriteOnePGMImage<VT>(filename, vec, strides, c0, c1, width, height);
        }
    }
}

// Write pyramid to PGM files, a pyramid is a list of a list of tensors
template<typename VT>
inline void WritePyramidToPGM(const char *header, const std::vector<std::vector<std::vector<uint8_t>>> &pyr,
                              const std::vector<long3> &strides, const std::vector<long3> &shape)
{
    std::string h = std::string(header) + std::string("_%ld_");

    for (int o = 0; o < (int)pyr.size(); ++o)
    {
        for (int l = 0; l < (int)pyr[o].size(); ++l)
        {
            std::string filename
                = h + std::to_string(o) + "_" + std::to_string(l) + ".pgm"; // NOSONAR: std::format is C++20.

            WriteImagesToPGM<VT>(filename.c_str(), pyr[o][l], strides[o], shape[o]);
        }
    }
}

// Write "fat" pyramid to PGM files, a pyramid is list of octave LNHWC tensors
template<typename VT>
inline void WritePyramidToPGM(const char *header, const std::vector<std::vector<uint8_t>> &pyr,
                              const std::vector<long4_16a> &strides, const std::vector<long4_16a> &shape)
{
    std::string h = std::string(header) + std::string("_%ld_");

    for (int o = 0; o < (int)pyr.size(); ++o)
    {
        std::string filename = h + std::to_string(o) + "_%ld" + ".pgm"; // NOSONAR: std::format is C++20.

        WriteImagesToPGM<VT>(filename.c_str(), pyr[o], strides[o], shape[o]);
    }
}

} // namespace nvcv::util

#endif // NVCV_TEST_COMMON_TENSOR_DATA_UTILS_HPP
