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

#ifndef NVCV_TEST_COMMON_TENSOR_DATA_UTILS_HPP
#define NVCV_TEST_COMMON_TENSOR_DATA_UTILS_HPP

#include <cuda_runtime.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <random>

namespace nvcv::test {

enum chflags
{
    C0 = 0x1 << 0,
    C1 = 0x1 << 1,
    C2 = 0x1 << 2,
    C3 = 0x1 << 3
};

// Holds a single image copied from an ITensor in host memory
// (N)CHW and (N)HWC tensors are supported but only 1 sample is held.
class TensorImageData
{
public:
    TensorImageData() = delete;
    explicit TensorImageData(const ITensorData *tensorData, int sampleIndex = 0);

    // H/W in logical pixels where byte offset == m_size.x * bytesPerPixel.
    const Size2D &size() const
    {
        return m_size;
    };

    // Returns the size in bytes of the color component.
    const int32_t bytesPerC() const
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
        uint32_t byteIndex = 0;

        if (!m_planar)
            byteIndex = (c * m_bytesPerC) + (x * m_bytesPerC * m_numC) + (y * m_rowStride);
        else
            byteIndex = (c * m_planeStride) + (x * m_bytesPerC) + (y * m_rowStride);

        if (byteIndex >= m_data.size())
            throw std::runtime_error("Requested data out of bounds");

        return reinterpret_cast<T *>(reinterpret_cast<unsigned char *>(m_data.data()) + byteIndex);
    }

private:
    std::vector<uint8_t> m_data;        // pointer to local data
    Size2D               m_size;        // h/w in logical pixels, byte offset == m_size.x * bytesPerPixel.
    int64_t              m_rowStride;   // Row stride in bytes
    int64_t              m_planeStride; // used for (n)CHW Tensors 0 if not CHW
    int                  m_numC;        // Number of color channels usually 1,3,4 (Y, RGB, ARGB)
    bool                 m_planar;      // If true the image is (n)CHW
    int32_t              m_bytesPerC;   // bytes per logical pixels
    NVCVTensorLayout     m_layout;      // layout of originating ITensor NVCV_TENSOR_CHW/NVCV_TENSOR_NHWC/HWC
};

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
static void SetTensorTo(const ITensorData *tensorData, DT data, int sample = -1);

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
static void SetTensorToRandomValue(const ITensorData *tensorData, DT minVal, DT maxVal, int sample = -1);

/**
 * Writes over the Tensor data with an array of type DT array must be
 * the size of sampleStride(). All samples will be overriden.
 * Function does not do data type checking
 *
 * @param[in,out] tensorData created tensor object.
 *
 * @param[in] data the data to set the tensor to.
 *
 * @param[in] sample optional the sample to write to if -1 all samples are written
 */
template<typename DT>
static void SetTensorFromVector(const ITensorData *tensorData, std::vector<DT> &data, int sample = -1);

/**
 * Returns a vector contains the values of the provided sample.
 *
 * @param[in] tensorData created tensor object.
 *
 * @param[in] sample the sample to copy to vector 0 index.
 *
 * @param[out] outData the data to set the tensor to.
 *
 */
template<typename DT>
static void GetVectorFromTensor(const ITensorData *tensorData, int sample, std::vector<DT> &outData);

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

template<typename DT>
void SetTensorTo(const ITensorData *tensorData, DT data, int sample)
{
    assert(tensorData);
    if (!nvcv::TensorDataAccessStrided::IsCompatible(*tensorData))
        throw std::runtime_error("Tensor Data is not pitch access capable.");

    auto tDataAc = nvcv::TensorDataAccessStrided::Create(*tensorData);

    if (tDataAc->numSamples() <= sample)
        throw std::runtime_error("Number of samples smaller than requested sample.");

    int             inElements = (tDataAc->sampleStride() / sizeof(DT));
    std::vector<DT> srcVec(inElements, data);

    int totalSamples;
    if (sample < 0)
    {
        totalSamples = tDataAc->numSamples();
        sample       = 0;
    }
    else
    {
        totalSamples = sample + 1;
    }

    for (int i = sample; i < totalSamples; ++i)
    {
        if (cudaSuccess
            != cudaMemcpy(tDataAc->sampleData(i), srcVec.data(), tDataAc->sampleStride(), cudaMemcpyHostToDevice))
        {
            throw std::runtime_error("CudaMemcpy failed");
        }
    }

    return;
}

template<typename DT>
static void SetTensorToRandomValueFloat(const ITensorData *tensorData, DT minVal, DT maxVal, int sample)
{
    assert(tensorData);
    if (!nvcv::TensorDataAccessStrided::IsCompatible(*tensorData))
        throw std::runtime_error("Tensor Data is not pitch access capable.");

    auto tDataAc = nvcv::TensorDataAccessStrided::Create(*tensorData);

    if (tDataAc->numSamples() <= sample)
        throw std::runtime_error("Number of samples smaller than requested sample.");

    int                        inElements = (tDataAc->sampleStride() / sizeof(DT));
    std::vector<DT>            srcVec(inElements);
    std::default_random_engine randEng(0);

    int totalSamples;
    if (sample < 0)
    {
        totalSamples = tDataAc->numSamples();
        sample       = 0;
    }
    else
    {
        totalSamples = sample + 1;
    }

    std::uniform_real_distribution<> srcRand(minVal, maxVal);
    for (int i = sample; i < totalSamples; ++i)
    {
        std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng); });
        if (cudaSuccess
            != cudaMemcpy(tDataAc->sampleData(i), srcVec.data(), tDataAc->sampleStride(), cudaMemcpyHostToDevice))
        {
            throw std::runtime_error("CudaMemcpy failed");
        }
    }
    return;
}

template<>
inline void SetTensorToRandomValue<float>(const ITensorData *tensorData, float minVal, float maxVal, int sample)
{
    SetTensorToRandomValueFloat<float>(tensorData, minVal, maxVal, sample);
}

template<>
inline void SetTensorToRandomValue<double>(const ITensorData *tensorData, double minVal, double maxVal, int sample)
{
    SetTensorToRandomValueFloat<double>(tensorData, minVal, maxVal, sample);
}

template<typename DT>
static void SetTensorToRandomValue(const ITensorData *tensorData, DT minVal, DT maxVal, int sample)
{
    assert(tensorData);
    if (!nvcv::TensorDataAccessStrided::IsCompatible(*tensorData))
        throw std::runtime_error("Tensor Data is not pitch access capable.");

    auto tDataAc = nvcv::TensorDataAccessStrided::Create(*tensorData);

    if (tDataAc->numSamples() <= sample)
        throw std::runtime_error("Number of samples smaller than requested sample.");

    int                        inElements = (tDataAc->sampleStride() / sizeof(DT));
    std::vector<DT>            srcVec(inElements);
    std::default_random_engine randEng(0);

    int totalSamples;
    if (sample < 0)
    {
        totalSamples = tDataAc->numSamples();
        sample       = 0;
    }
    else
    {
        totalSamples = sample + 1;
    }
    std::uniform_int_distribution<DT> srcRand{minVal, maxVal};
    for (int i = sample; i < totalSamples; ++i)
    {
        std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng); });
        if (cudaSuccess
            != cudaMemcpy(tDataAc->sampleData(i), srcVec.data(), tDataAc->sampleStride(), cudaMemcpyHostToDevice))
        {
            throw std::runtime_error("CudaMemcpy failed");
        }
    }

    return;
}

template<typename DT>
void SetTensorFromVector(const ITensorData *tensorData, std::vector<DT> &data, int sample)
{
    assert(tensorData);
    if (!nvcv::TensorDataAccessStrided::IsCompatible(*tensorData))
        throw std::runtime_error("Tensor Data is not pitch access capable.");

    auto tDataAc = nvcv::TensorDataAccessStrided::Create(*tensorData);

    if ((int64_t)(data.size() * sizeof(DT)) != tDataAc->sampleStride())
        throw std::runtime_error("Data vector is incorrect size.");

    if (tDataAc->numSamples() <= sample)
        throw std::runtime_error("Number of samples smaller than requested sample.");

    if (sample < 0)
    {
        for (int i = 0; i < tDataAc->numSamples(); ++i)
        {
            if (cudaSuccess
                != cudaMemcpy(tDataAc->sampleData(i), (char8_t *)data.data(), tDataAc->sampleStride(),
                              cudaMemcpyHostToDevice))
            {
                throw std::runtime_error("CudaMemcpy failed");
            }
        }
    }
    else
    {
        if (cudaSuccess
            != cudaMemcpy(tDataAc->sampleData(sample), (char8_t *)data.data(), tDataAc->sampleStride(),
                          cudaMemcpyHostToDevice))
        {
            throw std::runtime_error("CudaMemcpy failed");
        }
    }

    return;
}

template<typename DT>
void GetVectorFromTensor(const ITensorData *tensorData, int sample, std::vector<DT> &outData)
{
    assert(tensorData);
    if (!nvcv::TensorDataAccessStrided::IsCompatible(*tensorData))
        throw std::runtime_error("Tensor Data is not pitch access capable.");

    auto tDataAc = nvcv::TensorDataAccessStrided::Create(*tensorData);

    if (tDataAc->numSamples() <= sample)
        throw std::runtime_error("Number of samples smaller than requested sample.");

    int elements = (tDataAc->sampleStride() / sizeof(DT));

    outData.resize(elements);

    if (cudaSuccess
        != cudaMemcpy(outData.data(), tDataAc->sampleData(sample), tDataAc->sampleStride(), cudaMemcpyDeviceToHost))
    {
        throw std::runtime_error("CudaMemcpy failed");
    }

    return;
}

template<typename DT>
void SetCvDataTo(TensorImageData &cvImg, DT data, Size2D region, uint8_t chFlags)
{
    for (int x = 0; x < region.w; x++)
        for (int y = 0; y < region.h; y++)
            for (int c = 0; c < 4; c++)
                if (((chFlags >> c) & 0x1) == 0x1)
                    *cvImg.item<DT>(x, y, c) = data;

    return;
}

} // namespace nvcv::test

#endif // NVCV_TEST_COMMON_TENSOR_DATA_UTILS_HPP
