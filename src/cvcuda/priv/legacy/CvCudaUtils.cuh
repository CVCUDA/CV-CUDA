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

#ifndef CV_CUDA_UTILS_CUH
#define CV_CUDA_UTILS_CUH

#include <nvcv/Exception.hpp>
#include <nvcv/IImageBatchData.hpp>
#include <nvcv/IImageData.hpp>  // for IImageDataStridedCuda, etc.
#include <nvcv/ITensorData.hpp> // for ITensorDataStridedCuda, etc.
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/BorderWrap.hpp>   // for BorderWrap, etc.
#include <nvcv/cuda/DropCast.hpp>     // for DropCast, etc.
#include <nvcv/cuda/MathOps.hpp>      // for math operators
#include <nvcv/cuda/MathWrappers.hpp> // for sqrt, etc.
#include <nvcv/cuda/SaturateCast.hpp> // for SaturateCast, etc.
#include <nvcv/cuda/StaticCast.hpp>   // for StaticCast, etc.
#include <nvcv/cuda/TensorWrap.hpp>   // for TensorWrap, etc.
#include <nvcv/cuda/TypeTraits.hpp>   // for BaseType, etc.
#include <nvcv/cuda/math/LinAlg.hpp>  // for Vector, etc.
#include <util/Assert.h>              // for NVCV_ASSERT, etc.
#include <util/CheckError.hpp>        // for NVCV_CHECK_LOG, etc.

#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>

namespace nvcv::legacy::cuda_op {

typedef unsigned char uchar;
typedef signed char   schar;

#define get_batch_idx() (blockIdx.z)
#define get_lid()       (threadIdx.y * blockDim.x + threadIdx.x)

inline int divUp(int a, int b)
{
    assert(b > 0);
    return ceil((float)a / b);
};

struct DefaultTransformPolicy
{
    enum
    {
        block_size_x = 32,
        block_size_y = 8,
        shift        = 4
    };
};

template<class T> // base type
__host__ __device__ int32_t CalcNCHWImageStride(int rows, int cols, int channels)
{
    return rows * cols * channels * sizeof(T);
}

template<class T> // base type
__host__ __device__ int32_t CalcNCHWRowStride(int cols, int channels)
{
    return cols * sizeof(T);
}

template<class T> // base type
__host__ __device__ int32_t CalcNHWCImageStride(int rows, int cols, int channels)
{
    return rows * cols * channels * sizeof(T);
}

template<class T> // base type
__host__ __device__ int32_t CalcNHWCRowStride(int cols, int channels)
{
    return cols * channels * sizeof(T);
}

// Used to disambiguate between the constructors that accept legacy memory buffers,
// and the ones that accept the new ones. Just pass NewAPI as first parameter.
struct NewAPITag
{
};

constexpr NewAPITag NewAPI = {};

template<typename T>
struct Ptr2dNCHW
{
    typedef T value_type;

    __host__ __device__ __forceinline__ Ptr2dNCHW()
        : batches(0)
        , rows(0)
        , cols(0)
        , ch(0)
        , imgStride(0)
        , chStride(0)
        , data{}
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNCHW(int rows_, int cols_, int ch_, T *data_)
        : batches(1)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgStride(0)
        , rowStride(CalcNCHWRowStride<T>(cols, ch_))
        , data(data_)
    {
        chStride = rowStride * rows_;
    }

    __host__ __device__ __forceinline__ Ptr2dNCHW(int batches_, int rows_, int cols_, int ch_, T *data_)
        : batches(batches_)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgStride(CalcNCHWImageStride<T>(rows_, cols_, ch_))
        , rowStride(CalcNCHWRowStride<T>(cols, ch_))
        , data(data_)
    {
        chStride = rowStride * rows_;
    }

    __host__ __forceinline__ Ptr2dNCHW(const IImageDataStridedCuda &inData)
        : batches(1)
        , rows(inData.size().h)
        , cols(inData.size().w)
        , ch(inData.format().numPlanes())
        , imgStride(0)
    {
        if (inData.format().numPlanes() != inData.format().numChannels())
        {
            throw nvcv::Exception(Status::ERROR_INVALID_ARGUMENT, "Image must be planar");
        }

        rowStride = inData.plane(0).rowStride;
        chStride  = rowStride * inData.plane(0).height;
        data      = reinterpret_cast<T *>(inData.plane(0).basePtr);

        for (int i = 0; i < ch; ++i)
        {
            const ImagePlaneStrided &plane = inData.plane(i);

            if (i > 0)
            {
                if (plane.rowStride != rowStride)
                {
                    throw nvcv::Exception(Status::ERROR_INVALID_ARGUMENT,
                                          "All image planes' row pitch must be the same");
                }

                if (plane.basePtr != reinterpret_cast<const NVCVByte *>(data) + rowStride * plane.height * i)
                {
                    throw nvcv::Exception(Status::ERROR_INVALID_ARGUMENT, "All image buffer must be packed");
                }

                if (inData.format().planeDataType(i) != inData.format().planeDataType(0))
                {
                    throw nvcv::Exception(Status::ERROR_INVALID_ARGUMENT,
                                          "All image planes must have the same data type");
                }

                if (plane.width != inData.plane(0).width || plane.height != inData.plane(0).height)
                {
                    throw nvcv::Exception(Status::ERROR_INVALID_ARGUMENT, "All image planes must have the same size");
                }
            }
        }
    }

    __host__ __forceinline__ Ptr2dNCHW(const TensorDataAccessStridedImagePlanar &tensor)
    {
        batches = tensor.numSamples();
        rows    = tensor.numRows();
        cols    = tensor.numCols();
        ch      = tensor.numChannels();

        imgStride = tensor.sampleStride();
        chStride  = tensor.planeStride();
        rowStride = tensor.rowStride();
        data      = tensor.sampleData(0);
    }

    // ptr for uchar, ushort, float, typename T -> uchar etc.
    // each fetch operation get a single channel element
    __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c)
    {
        //return (T *)(data + b * ch * rows * cols + c * rows * cols + y * cols + x);
        return (T *)(reinterpret_cast<std::byte *>(data) + b * imgStride + c * chStride + y * rowStride
                     + x * sizeof(T));
    }

    const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c) const
    {
        //return (const T *)(data + b * ch * rows * cols + c * rows * cols + y * cols + x);
        return (const T *)(reinterpret_cast<const std::byte *>(data) + b * imgStride + c * chStride + y * rowStride
                           + x * sizeof(T));
    }

    int   batches;
    int   rows;
    int   cols;
    int   ch;
    int   imgStride;
    int   rowStride;
    int   chStride;
    void *data;
};

template<typename T>
struct Ptr2dNHWC
{
    typedef T value_type;

    __host__ __device__ __forceinline__ Ptr2dNHWC()
        : batches(0)
        , rows(0)
        , cols(0)
        , imgStride(0)
        , rowStride(0)
        , ch(0)
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNHWC(int rows_, int cols_, int ch_, T *data_)
        : batches(1)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgStride(0)
        , rowStride(CalcNHWCRowStride<T>(cols_, ch_))
        , data(data_)
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNHWC(int batches_, int rows_, int cols_, int ch_, T *data_)
        : batches(batches_)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgStride(CalcNHWCImageStride<T>(rows_, cols_, ch_))
        , rowStride(CalcNHWCRowStride<T>(cols_, ch_))
        , data(data_)
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNHWC(NewAPITag, int rows_, int cols_, int ch_, int rowStride_, T *data_)
        : batches(1)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgStride(0)
        , rowStride(rowStride_)
        , data(data_)
    {
    }

    __host__ __forceinline__ Ptr2dNHWC(const IImageDataStridedCuda &inData)
        : batches(1)
        , rows(inData.size().h)
        , cols(inData.size().w)
        , ch(inData.format().numChannels())
        , imgStride(0)
    {
        if (inData.format().numPlanes() != 1)
        {
            throw nvcv::Exception(Status::ERROR_INVALID_ARGUMENT, "Image must have only one plane");
        }

        const ImagePlaneStrided &plane = inData.plane(0);

        rowStride = inData.plane(0).rowStride;
        data      = reinterpret_cast<T *>(inData.plane(0).basePtr);
    }

    __host__ __forceinline__ Ptr2dNHWC(const TensorDataAccessStridedImagePlanar &tensor, int cols_, int rows_)
    {
        batches = tensor.numSamples();
        rows    = rows_; // allow override of rows and cols with smaller crop rect
        cols    = cols_;
        ch      = tensor.numChannels();

        imgStride = tensor.sampleStride();
        rowStride = tensor.rowStride();
        data      = reinterpret_cast<T *>(tensor.sampleData(0));
    }

    __host__ __forceinline__ Ptr2dNHWC(const TensorDataAccessStridedImagePlanar &tensor)
    {
        batches = tensor.numSamples();
        rows    = tensor.numRows();
        cols    = tensor.numCols();
        ch      = tensor.numChannels();

        imgStride = tensor.sampleStride();
        rowStride = tensor.rowStride();
        data      = reinterpret_cast<T *>(tensor.sampleData(0));
    }

    // ptr for uchar1/3/4, ushort1/3/4, float1/3/4, typename T -> uchar3 etc.
    // each fetch operation get a x-channel elements
    __host__ __device__ __forceinline__ T *ptr(int b, int y, int x)
    {
        //return (T *)(data + b * rows * cols + y * cols + x);
        return (T *)(reinterpret_cast<std::byte *>(data) + b * imgStride + y * rowStride + x * sizeof(T));
    }

    const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x) const
    {
        //return (const T *)(data + b * rows * cols + y * cols + x);
        return (const T *)(reinterpret_cast<const std::byte *>(data) + b * imgStride + y * rowStride + x * sizeof(T));
    }

    // ptr for uchar, ushort, float, typename T -> uchar etc.
    // each fetch operation get a single channel element
    __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c)
    {
        //return (T *)(data + b * rows * cols * ch + y * cols * ch + x * ch + c);
        return (T *)(reinterpret_cast<std::byte *>(data) + b * imgStride + y * rowStride + (x * ch + c) * sizeof(T));
    }

    const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c) const
    {
        //return (const T *)(data + b * rows * cols * ch + y * cols * ch + x * ch + c);
        return (const T *)(reinterpret_cast<const std::byte *>(data) + b * imgStride + y * rowStride
                           + (x * ch + c) * sizeof(T));
    }

    __host__ __device__ __forceinline__ int at_rows(int b)
    {
        return rows;
    }

    __host__ __device__ __forceinline__ int at_rows(int b) const
    {
        return rows;
    }

    __host__ __device__ __forceinline__ int at_cols(int b)
    {
        return cols;
    }

    __host__ __device__ __forceinline__ int at_cols(int b) const
    {
        return cols;
    }

    int batches;
    int rows;
    int cols;
    int ch;
    int imgStride;
    int rowStride;
    T  *data;
};

template<typename T>
struct Ptr2dVarShapeNHWC
{
    typedef T value_type;

    __host__ __device__ __forceinline__ Ptr2dVarShapeNHWC()
        : batches(0)
        , imgList(NULL)
        , nch(0)
    {
    }

    __host__ __forceinline__ Ptr2dVarShapeNHWC(const nvcv::IImageBatchVarShapeDataStridedCuda &data, int nch_ = -1)
        : batches(data.numImages())
        , imgList(data.imageList())
        , nch(
              [&]
              {
                  // If not using number of channels,
                  if (nch_ < 0)
                  {
                      // Require that all images have the same format (it'd be better if we had data.uniqueDataType)
                      if (!data.uniqueFormat())
                      {
                          throw std::runtime_error("Images in a batch must all have the same format");
                      }

                      assert(1 == data.uniqueFormat().numPlanes() && "This class is only for NHWC");

                      return data.uniqueFormat().numChannels();
                  }
                  else
                  {
                      // Use the given number of channels.
                      // The assumption here is that all formats have one plane and this number of channels,
                      // although they can be swizzled.
                      return nch_;
                  }
              }())
    {
    }

    // ptr for uchar1/3/4, ushort1/3/4, float1/3/4, typename T -> uchar3 etc.
    // each fetch operation get a x-channel elements
    __host__ __device__ __forceinline__ T *ptr(int b, int y, int x)
    {
        return reinterpret_cast<T *>(imgList[b].planes[0].basePtr + imgList[b].planes[0].rowStride * y) + x;
    }

    const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x) const
    {
        return reinterpret_cast<const T *>(imgList[b].planes[0].basePtr + imgList[b].planes[0].rowStride * y) + x;
    }

    // ptr for uchar, ushort, float, typename T -> uchar etc.
    // each fetch operation get a single channel element
    __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c)
    {
        return reinterpret_cast<T *>(imgList[b].planes[0].basePtr + imgList[b].planes[0].rowStride * y) + (x * nch + c);
    }

    const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c) const
    {
        return reinterpret_cast<const T *>(imgList[b].planes[0].basePtr + imgList[b].planes[0].rowStride * y)
             + (x * nch + c);
    }

    // Commented out, "offset" doesn't take into account row pitch, but this info is needed.
    // If this function is actually needed, the kernel that calls it has to take into account
    // the pitch info. Use "at_strides" to get it.
#if 0
    // ptr for direct offset, less duplicated computation for higher performance
    __host__ __device__ __forceinline__ T *ptr(int b, int offset)
    {
        return (T *)(data[b] + offset);
    }
    __host__ __device__ __forceinline__ const T *ptr(int b, int offset) const
    {
        return (const T *)(data[b] + offset);
    }
#endif

    __host__ __device__ __forceinline__ int at_strides(int b) const
    {
        return imgList[b].planes[0].rowStride;
    }

    __host__ __device__ __forceinline__ int at_rows(int b) const
    {
        return imgList[b].planes[0].height;
    }

    __host__ __device__ __forceinline__ int at_cols(int b) const
    {
        return imgList[b].planes[0].width;
    }

    const int                     batches;
    const NVCVImageBufferStrided *imgList;
    const int                     nch;
};

template<typename D>
struct BrdConstant
{
    typedef D result_type;

    __host__ __device__ __forceinline__ BrdConstant(int height_, int width_, const D &val_ = nvcv::cuda::SetAll<D>(0))
        : height(height_)
        , width(width_)
        , val(val_)
    {
    }

    template<typename Ptr2D>
    __device__ __forceinline__ D at(int b, int y, int x, const Ptr2D &src) const
    {
        return ((float)x >= 0 && x < src.at_cols(b) && (float)y >= 0 && y < src.at_rows(b))
                 ? nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<D>>(*src.ptr(b, y, x))
                 : val;
    }

    int height;
    int width;
    D   val;
};

template<typename D>
struct BrdReplicate
{
    typedef D result_type;

    __host__ __device__ __forceinline__ BrdReplicate(int height, int width)
        : last_row(height - 1)
        , last_col(width - 1)
    {
    }

    template<typename U>
    __host__ __device__ __forceinline__ BrdReplicate(int height, int width, U)
        : last_row(height - 1)
        , last_col(width - 1)
    {
    }

    __device__ __forceinline__ int idx_row_low(int y) const
    {
        return cuda::max(y, 0);
    }

    __device__ __forceinline__ int idx_row_high(int y, int last_row_) const
    {
        return cuda::min(y, last_row_);
    }

    __device__ __forceinline__ int idx_row(int y, int last_row_) const
    {
        return idx_row_low(idx_row_high(y, last_row_));
    }

    __device__ __forceinline__ int idx_col_low(int x) const
    {
        return cuda::max(x, 0);
    }

    __device__ __forceinline__ int idx_col_high(int x, int last_col_) const
    {
        return cuda::min(x, last_col_);
    }

    __device__ __forceinline__ int idx_col(int x, int last_col_) const
    {
        return idx_col_low(idx_col_high(x, last_col_));
    }

    template<typename Ptr2D>
    __device__ __forceinline__ D at(int b, int y, int x, const Ptr2D &src) const
    {
        return nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<D>>(
            *src.ptr(b, idx_row(y, src.at_rows(b) - 1), idx_col(x, src.at_cols(b) - 1)));
    }

    int last_row;
    int last_col;
};

template<typename D>
struct BrdReflect101
{
    typedef D result_type;

    __host__ __device__ __forceinline__ BrdReflect101(int height, int width)
        : last_row(height - 1)
        , last_col(width - 1)
    {
    }

    template<typename U>
    __host__ __device__ __forceinline__ BrdReflect101(int height, int width, U)
        : last_row(height - 1)
        , last_col(width - 1)
    {
    }

    __device__ __forceinline__ int idx_row_low(int y, int last_row_) const
    {
        return ::abs(y) % (last_row_ + 1);
    }

    __device__ __forceinline__ int idx_row_high(int y, int last_row_) const
    {
        return ::abs(last_row_ - ::abs(last_row_ - y)) % (last_row_ + 1);
    }

    __device__ __forceinline__ int idx_row(int y, int last_row_) const
    {
        return idx_row_low(idx_row_high(y, last_row_), last_row_);
    }

    __device__ __forceinline__ int idx_col_low(int x, int last_col_) const
    {
        return ::abs(x) % (last_col_ + 1);
    }

    __device__ __forceinline__ int idx_col_high(int x, int last_col_) const
    {
        return ::abs(last_col_ - ::abs(last_col_ - x)) % (last_col_ + 1);
    }

    __device__ __forceinline__ int idx_col(int x, int last_col_) const
    {
        return idx_col_low(idx_col_high(x, last_col_), last_col_);
    }

    template<typename Ptr2D>
    __device__ __forceinline__ D at(int b, int y, int x, const Ptr2D &src) const
    {
        return nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<D>>(
            *src.ptr(b, idx_row(y, src.at_rows(b) - 1), idx_col(x, src.at_cols(b) - 1)));
    }

    int last_row;
    int last_col;
};

template<typename D>
struct BrdReflect
{
    typedef D result_type;

    __host__ __device__ __forceinline__ BrdReflect(int height, int width)
        : last_row(height - 1)
        , last_col(width - 1)
    {
    }

    template<typename U>
    __host__ __device__ __forceinline__ BrdReflect(int height, int width, U)
        : last_row(height - 1)
        , last_col(width - 1)
    {
    }

    __device__ __forceinline__ int idx_row_low(int y, int last_row_) const
    {
        return (::abs(y) - (y < 0)) % (last_row_ + 1);
    }

    __device__ __forceinline__ int idx_row_high(int y, int last_row_) const
    {
        return /*::abs*/ (last_row_ - ::abs(last_row_ - y) + (y > last_row_)) /*% (last_row + 1)*/;
    }

    __device__ __forceinline__ int idx_row(int y, int last_row_) const
    {
        return idx_row_low(idx_row_high(y, last_row_), last_row_);
    }

    __device__ __forceinline__ int idx_col_low(int x, int last_col_) const
    {
        return (::abs(x) - (x < 0)) % (last_col_ + 1);
    }

    __device__ __forceinline__ int idx_col_high(int x, int last_col_) const
    {
        return (last_col_ - ::abs(last_col_ - x) + (x > last_col_));
    }

    __device__ __forceinline__ int idx_col(int x, int last_col_) const
    {
        return idx_col_low(idx_col_high(x, last_col_), last_col_);
    }

    template<typename Ptr2D>
    __device__ __forceinline__ D at(int b, int y, int x, const Ptr2D &src) const
    {
        return nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<D>>(
            *src.ptr(b, idx_row(y, src.at_rows(b) - 1), idx_col(x, src.at_cols(b) - 1)));
    }

    int last_row;
    int last_col;
};

template<typename D>
struct BrdWrap
{
    typedef D result_type;

    __host__ __device__ __forceinline__ BrdWrap(int height_, int width_)
        : height(height_)
        , width(width_)
    {
    }

    template<typename U>
    __host__ __device__ __forceinline__ BrdWrap(int height_, int width_, U)
        : height(height_)
        , width(width_)
    {
    }

    __device__ __forceinline__ int idx_row_low(int y, int height_) const
    {
        return (y >= 0) ? y : (y - ((y - height_ + 1) / height_) * height_);
    }

    __device__ __forceinline__ int idx_row_high(int y, int height_) const
    {
        return (y < height_) ? y : (y % height_);
    }

    __device__ __forceinline__ int idx_row(int y, int height_) const
    {
        return idx_row_high(idx_row_low(y, height_), height_);
    }

    __device__ __forceinline__ int idx_col_low(int x, int width_) const
    {
        return (x >= 0) ? x : (x - ((x - width_ + 1) / width_) * width_);
    }

    __device__ __forceinline__ int idx_col_high(int x, int width_) const
    {
        return (x < width_) ? x : (x % width_);
    }

    __device__ __forceinline__ int idx_col(int x, int width_) const
    {
        return idx_col_high(idx_col_low(x, width_), width_);
    }

    template<typename Ptr2D>
    __device__ __forceinline__ D at(int b, int y, int x, const Ptr2D &src) const
    {
        return nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<D>>(
            *src.ptr(b, idx_row(y, src.at_rows(b)), idx_col(x, src.at_cols(b))));
    }

    int height;
    int width;
};

template<typename Ptr2D, typename B>
struct BorderReader
{
    typedef typename B::result_type elem_type;

    __host__ __device__ __forceinline__ BorderReader(const Ptr2D &ptr_, const B &b_)
        : ptr(ptr_)
        , b(b_)
    {
    }

    __device__ __forceinline__ elem_type operator()(int bidx, int y, int x) const
    {
        return b.at(bidx, y, x, ptr);
    }

    __host__ __device__ __forceinline__ int at_rows(int b)
    {
        return ptr.at_rows(b);
    }

    __host__ __device__ __forceinline__ int at_rows(int b) const
    {
        return ptr.at_rows(b);
    }

    __host__ __device__ __forceinline__ int at_cols(int b)
    {
        return ptr.at_cols(b);
    }

    __host__ __device__ __forceinline__ int at_cols(int b) const
    {
        return ptr.at_cols(b);
    }

    Ptr2D ptr;
    B     b;
};

template<typename Ptr2D, typename D>
struct BorderReader<Ptr2D, BrdConstant<D>>
{
    typedef typename BrdConstant<D>::result_type elem_type;

    __host__ __device__ __forceinline__ BorderReader(const Ptr2D &ptr_, const BrdConstant<D> &b)
        : ptr(ptr_)
        , height(b.height)
        , width(b.width)
        , val(b.val)
    {
    }

    __device__ __forceinline__ D operator()(int bidx, int y, int x) const
    {
        return ((float)x >= 0 && x < ptr.at_cols(bidx) && (float)y >= 0 && y < ptr.at_rows(bidx))
                 ? nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<D>>(*ptr.ptr(bidx, y, x))
                 : val;
    }

    __host__ __device__ __forceinline__ int at_rows(int b)
    {
        return ptr.at_rows(b);
    }

    __host__ __device__ __forceinline__ int at_rows(int b) const
    {
        return ptr.at_rows(b);
    }

    __host__ __device__ __forceinline__ int at_cols(int b)
    {
        return ptr.at_cols(b);
    }

    __host__ __device__ __forceinline__ int at_cols(int b) const
    {
        return ptr.at_cols(b);
    }

    Ptr2D ptr;
    int   height;
    int   width;
    D     val;
};

template<typename BrdReader>
struct PointFilter
{
    typedef typename BrdReader::elem_type elem_type;

    explicit __host__ __device__ __forceinline__ PointFilter(const BrdReader &src_, float fx = 0.f, float fy = 0.f)
        : src(src_)
    {
    }

    __device__ __forceinline__ elem_type operator()(int bidx, float y, float x) const
    {
        return src(bidx, __float2int_rz(y), __float2int_rz(x));
    }

    BrdReader src;
};

template<typename BrdReader>
struct LinearFilter
{
    typedef typename BrdReader::elem_type elem_type;

    explicit __host__ __device__ __forceinline__ LinearFilter(const BrdReader &src_, float fx = 0.f, float fy = 0.f)
        : src(src_)
    {
    }

    __device__ __forceinline__ elem_type operator()(int bidx, float y, float x) const
    {
        using work_type = nvcv::cuda::ConvertBaseTypeTo<float, elem_type>;
        work_type out   = nvcv::cuda::SetAll<work_type>(0);

        // to prevent -2147483648 > 0 in border
        // float x_float = x >= std::numeric_limits<int>::max() ? ((float) std::numeric_limits<int>::max() - 1):
        //     ((x <= std::numeric_limits<int>::min() + 1) ? ((float) std::numeric_limits<int>::min() + 1): x);
        // float y_float = y >= std::numeric_limits<int>::max() ? ((float) std::numeric_limits<int>::max() - 1):
        //     ((y <= std::numeric_limits<int>::min() + 1) ? ((float) std::numeric_limits<int>::min() + 1): y);

        const int x1 = __float2int_rd(x);
        const int y1 = __float2int_rd(y);
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;

        elem_type src_reg = src(bidx, y1, x1);
        out               = out + src_reg * ((x2 - x) * (y2 - y));

        src_reg = src(bidx, y1, x2);
        out     = out + src_reg * ((x - x1) * (y2 - y));

        src_reg = src(bidx, y2, x1);
        out     = out + src_reg * ((x2 - x) * (y - y1));

        src_reg = src(bidx, y2, x2);
        out     = out + src_reg * ((x - x1) * (y - y1));

        return nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<elem_type>>(out);
    }

    BrdReader src;
};

template<typename BrdReader>
struct CubicFilter
{
    typedef typename BrdReader::elem_type elem_type;
    using work_type = nvcv::cuda::ConvertBaseTypeTo<float, elem_type>;

    explicit __host__ __device__ __forceinline__ CubicFilter(const BrdReader &src_, float fx = 0.f, float fy = 0.f)
        : src(src_)
    {
    }

    static __device__ __forceinline__ float bicubicCoeff(float x_)
    {
        float x = fabsf(x_);
        if (x <= 1.0f)
        {
            return x * x * (1.5f * x - 2.5f) + 1.0f;
        }
        else if (x < 2.0f)
        {
            return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
        }
        else
        {
            return 0.0f;
        }
    }

    __device__ elem_type operator()(int bidx, float y, float x) const
    {
        const float xmin = ceilf(x - 2.0f);
        const float xmax = floorf(x + 2.0f);

        const float ymin = ceilf(y - 2.0f);
        const float ymax = floorf(y + 2.0f);

        work_type sum  = nvcv::cuda::SetAll<work_type>(0);
        float     wsum = 0.0f;

        for (float cy = ymin; cy <= ymax; cy += 1.0f)
        {
            for (float cx = xmin; cx <= xmax; cx += 1.0f)
            {
                const float w = bicubicCoeff(x - cx) * bicubicCoeff(y - cy);
                sum           = sum + w * src(bidx, __float2int_rd(cy), __float2int_rd(cx));
                wsum += w;
            }
        }

        work_type res = (!wsum) ? nvcv::cuda::SetAll<work_type>(0) : sum / wsum;

        return nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<elem_type>>(res);
    }

    BrdReader src;
};

// for integer scaling
template<typename BrdReader>
struct IntegerAreaFilter
{
    typedef typename BrdReader::elem_type elem_type;

    explicit __host__ __device__ __forceinline__ IntegerAreaFilter(const BrdReader &src_, float scale_x_,
                                                                   float scale_y_)
        : src(src_)
        , scale_x(scale_x_)
        , scale_y(scale_y_)
        , scale(1.f / (scale_x * scale_y))
    {
    }

    __device__ __forceinline__ elem_type operator()(int bidx, float y, float x) const
    {
        float fsx1 = x * scale_x;
        float fsx2 = fsx1 + scale_x;

        int sx1 = __float2int_ru(fsx1);
        int sx2 = __float2int_rd(fsx2);

        float fsy1 = y * scale_y;
        float fsy2 = fsy1 + scale_y;

        int sy1 = __float2int_ru(fsy1);
        int sy2 = __float2int_rd(fsy2);

        using work_type = nvcv::cuda::ConvertBaseTypeTo<float, elem_type>;
        work_type out   = nvcv::cuda::SetAll<work_type>(0.f);

        for (int dy = sy1; dy < sy2; ++dy)
            for (int dx = sx1; dx < sx2; ++dx)
            {
                out = out + src(bidx, dy, dx) * scale;
            }

        return nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<elem_type>>(out);
    }

    BrdReader src;
    float     scale_x, scale_y, scale;
};

template<typename BrdReader>
struct AreaFilter
{
    typedef typename BrdReader::elem_type elem_type;

    explicit __host__ __device__ __forceinline__ AreaFilter(const BrdReader &src_, float scale_x_, float scale_y_)
        : src(src_)
        , scale_x(scale_x_)
        , scale_y(scale_y_)
    {
    }

    __device__ __forceinline__ elem_type operator()(int bidx, float y, float x) const
    {
        float fsx1 = x * scale_x;
        float fsx2 = fsx1 + scale_x;

        int sx1 = __float2int_ru(fsx1);
        int sx2 = __float2int_rd(fsx2);

        float fsy1 = y * scale_y;
        float fsy2 = fsy1 + scale_y;

        int sy1 = __float2int_ru(fsy1);
        int sy2 = __float2int_rd(fsy2);

        float scale = 1.f / (fminf(scale_x, src.at_cols(bidx) - fsx1) * fminf(scale_y, src.at_rows(bidx) - fsy1));

        using work_type = nvcv::cuda::ConvertBaseTypeTo<float, elem_type>;
        work_type out   = nvcv::cuda::SetAll<work_type>(0.f);

        for (int dy = sy1; dy < sy2; ++dy)
        {
            for (int dx = sx1; dx < sx2; ++dx) out = out + src(bidx, dy, dx) * scale;

            if (sx1 > fsx1)
                out = out + src(bidx, dy, (sx1 - 1)) * ((sx1 - fsx1) * scale);

            if (sx2 < fsx2)
                out = out + src(bidx, dy, sx2) * ((fsx2 - sx2) * scale);
        }

        if (sy1 > fsy1)
            for (int dx = sx1; dx < sx2; ++dx) out = out + src(bidx, (sy1 - 1), dx) * ((sy1 - fsy1) * scale);

        if (sy2 < fsy2)
            for (int dx = sx1; dx < sx2; ++dx) out = out + src(bidx, sy2, dx) * ((fsy2 - sy2) * scale);

        if ((sy1 > fsy1) && (sx1 > fsx1))
            out = out + src(bidx, (sy1 - 1), (sx1 - 1)) * ((sy1 - fsy1) * (sx1 - fsx1) * scale);

        if ((sy1 > fsy1) && (sx2 < fsx2))
            out = out + src(bidx, (sy1 - 1), sx2) * ((sy1 - fsy1) * (fsx2 - sx2) * scale);

        if ((sy2 < fsy2) && (sx2 < fsx2))
            out = out + src(bidx, sy2, sx2) * ((fsy2 - sy2) * (fsx2 - sx2) * scale);

        if ((sy2 < fsy2) && (sx1 > fsx1))
            out = out + src(bidx, sy2, (sx1 - 1)) * ((fsy2 - sy2) * (sx1 - fsx1) * scale);

        return nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<elem_type>>(out);
    }

    BrdReader src;
    float     scale_x, scale_y;
    int       width, haight;
};

inline void normalizeAnchor(int &anchor, int ksize)
{
    if (anchor < 0)
        anchor = ksize >> 1;
    NVCV_ASSERT(0 <= anchor && anchor < ksize);
}

inline void normalizeAnchor(int2 &anchor, Size2D ksize)
{
    normalizeAnchor(anchor.x, ksize.w);
    normalizeAnchor(anchor.y, ksize.h);
}

#ifndef checkKernelErrors
#    define checkKernelErrors(expr)                                                               \
        do                                                                                        \
        {                                                                                         \
            expr;                                                                                 \
                                                                                                  \
            cudaError_t __err = cudaGetLastError();                                               \
            if (__err != cudaSuccess)                                                             \
            {                                                                                     \
                printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
                abort();                                                                          \
            }                                                                                     \
        }                                                                                         \
        while (0)
#endif

#ifndef checkCudaErrors
#    define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        const char *errorStr = NULL;
        errorStr             = cudaGetErrorString(err);
        fprintf(stderr,
                "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
                "line %i.\n",
                err, errorStr, file, line);
        exit(1);
    }
}
#endif

#define TERM_NORMAL  "\033[0m"
#define TERM_RED     "\033[0;31m"
#define TERM_YELLOW  "\033[0;33m"
#define TERM_GREEN   "\033[0;32m"
#define TERM_MAGENTA "\033[1;35m"

enum class LogLevel : uint8_t
{
    kERROR   = 0,
    kWARNING = 1,
    kINFO    = 2,
    kDEBUG   = 3,
};

inline void log(LogLevel log_level, LogLevel reportable_severity, std::string msg)
{
    // suppress messages with severity enum value greater than the reportable
    if (log_level > reportable_severity)
    {
        return;
    }

    switch (log_level)
    {
    case LogLevel::kERROR:
        std::cerr << TERM_RED;
        break;
    case LogLevel::kWARNING:
        std::cerr << TERM_YELLOW;
        break;
    case LogLevel::kINFO:
        std::cerr << TERM_GREEN;
        break;
    case LogLevel::kDEBUG:
        std::cerr << TERM_MAGENTA;
        break;
    default:
        break;
    }

    switch (log_level)
    {
    case LogLevel::kERROR:
        std::cerr << "ERROR: ";
        break;
    case LogLevel::kWARNING:
        std::cerr << "WARNING: ";
        break;
    case LogLevel::kINFO:
        std::cerr << "INFO: ";
        break;
    case LogLevel::kDEBUG:
        std::cerr << "DEBUG: ";
        break;
    default:
        std::cerr << "UNKNOWN: ";
        break;
    }

    std::cerr << TERM_NORMAL;
    std::cerr << msg << std::endl;
}

#ifdef CUDA_DEBUG_LOG
#    define SEVERITY LogLevel::kDEBUG
#else
#    define SEVERITY LogLevel::kINFO
#endif

#define GET_MACRO(NAME, ...) NAME

#define CUDA_LOG(l, sev, msg)   \
    do                          \
    {                           \
        std::stringstream ss{}; \
        ss << msg;              \
        log(l, sev, ss.str());  \
    }                           \
    while (0)

#define LOG_DEBUG_GLOBAL(s)   CUDA_LOG(LogLevel::kDEBUG, SEVERITY, s)
#define LOG_INFO_GLOBAL(s)    CUDA_LOG(LogLevel::kINFO, SEVERITY, s)
#define LOG_WARNING_GLOBAL(s) CUDA_LOG(LogLevel::kWARNING, SEVERITY, s)
#define LOG_ERROR_GLOBAL(s)   CUDA_LOG(LogLevel::kERROR, SEVERITY, s)

#define LOG_DEBUG(...)   GET_MACRO(LOG_DEBUG_GLOBAL, __VA_ARGS__)(__VA_ARGS__)
#define LOG_INFO(...)    GET_MACRO(LOG_INFO_GLOBAL, __VA_ARGS__)(__VA_ARGS__)
#define LOG_WARNING(...) GET_MACRO(LOG_WARNING_GLOBAL, __VA_ARGS__)(__VA_ARGS__)
#define LOG_ERROR(...)   GET_MACRO(LOG_ERROR_GLOBAL, __VA_ARGS__)(__VA_ARGS__)

} // namespace nvcv::legacy::cuda_op

#endif // CV_CUDA_UTILS_CUH
