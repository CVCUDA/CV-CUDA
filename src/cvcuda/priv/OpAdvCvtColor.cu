/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OpAdvCvtColor.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"
#include "nvcv/TensorDataAccess.hpp"

#include "legacy/CvCudaUtils.cuh"

#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <nvcv/ColorSpec.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>

#define BLOCK 32

namespace legacy = nvcv::legacy::cuda_op;
namespace cuda   = nvcv::cuda;

struct RGB2YUVConstants
{
    const int R2Y;
    const int G2Y;
    const int B2Y;
    const int B2U;
    const int R2V;
};

struct YUV2RGBConstants
{
    const int U2B;
    const int U2G;
    const int V2G;
    const int V2R;
};

/*
 * YUV conversion formula
 * Y  = R2YF * r * + G2YF * g + B2YF * b
 * U = (B - Y) * B2UF
 * V = (R - Y) * R2VF
 *
 */
// Values per: https://www.itu.int/rec/R-REC-BT.601
// RGB->YUV conversion constants
// static constexpr float R2YF_601 = 0.299f;       //Kr 601
// static constexpr float G2YF_601 = 0.587f;       //Kg 601
// static constexpr float B2YF_601 = 0.114f;       //Kb 601
// static constexpr float B2UF_601 = 0.564334086f; // 1.0f/1.772f Cb/U
// static constexpr float R2VF_601 = 0.713266762;  // 1.0f/1.402f Cr/V

static constexpr int R2Y_601 = 4899;  // = R2YF*16384
static constexpr int G2Y_601 = 9671;  // = G2YF*16384
static constexpr int B2Y_601 = 1868;  // = B2YF*16384
static constexpr int B2U_601 = 9246;  // = B2UF*16384 Cb
static constexpr int R2V_601 = 11686; // = R2VF*16384 Cr

// YUV > RGB conversion (inverse of the forward matrix)
// static constexpr float V2RF_601 = 1.402f;
// static constexpr float U2GF_601 = -.344f;
// static constexpr float V2GF_601 = -.714f;
// static constexpr float U2BF_601 = 1.772f;

static constexpr int V2R_601 = 22970;  // = V2RF*16384
static constexpr int U2G_601 = -5636;  // = U2GF*16384
static constexpr int V2G_601 = -11698; // = V2GF*16384
static constexpr int U2B_601 = 29032;  // = U2BF*16384

// Values per: https://www.itu.int/rec/R-REC-BT.709
// static constexpr float R2YF_709 = 0.2126f;      //Kr 601
// static constexpr float G2YF_709 = 0.7152f;      //Kg 601
// static constexpr float B2YF_709 = 0.0772f;      //Kb 601
// static constexpr float B2UF_709 = 0.538909248f; //1.0f/1.8556f Cb/U
// static constexpr float R2VF_709 = 0.63500127f;  // 1.0f/1.5748f Cr/V

static constexpr int R2Y_709 = 3483;  // = R2YF*16384
static constexpr int G2Y_709 = 11718; // = G2YF*16384
static constexpr int B2Y_709 = 1265;  // = B2YF*16384
static constexpr int B2U_709 = 8829;  // = B2UF*16384 Cb
static constexpr int R2V_709 = 10404; // = R2VF*16384 Cr

// YUV > RGB conversion (inverse of the forward matrix)
// static constexpr float V2RF_709 = 1.5748f;
// static constexpr float U2GF_709 = -.187324f;
// static constexpr float V2GF_709 = -.468124f;
// static constexpr float U2BF_709 = 1.8556f;

static constexpr int V2R_709 = 25802; // = V2RF*16384
static constexpr int U2G_709 = -3069; // = U2GF*16384
static constexpr int V2G_709 = -7670; // = V2GF*16384
static constexpr int U2B_709 = 30402; // = U2BF*16384

// Values per: https://www.itu.int/rec/R-REC-BT.2020
// static constexpr float R2YF_2020 = 0.2627f;      //Kr 601
// static constexpr float G2YF_2020 = 0.6780f;      //Kg 601
// static constexpr float B2YF_2020 = 0.0593f;      //Kb 601
// static constexpr float B2UF_2020 = 0.531519082f; //1.0f/1.8814f Cb/U
// static constexpr float R2VF_2020 = 0.678150007f; //1.0f/1.4746f // Cr/V

static constexpr int R2Y_2020 = 4304;  // = R2YF*16384
static constexpr int G2Y_2020 = 11108; // = G2YF*16384
static constexpr int B2Y_2020 = 972;   // = B2YF*16384
static constexpr int B2U_2020 = 8708;  // = B2UF*16384 Cb
static constexpr int R2V_2020 = 11111; // = R2VF*16384 Cr

// YUV > RGB conversion (inverse of the forward matrix)
// static constexpr float V2RF_2020 = 1.4746f;
// static constexpr float U2GF_2020 = -0.16455;
// static constexpr float V2GF_2020 = -0.57135f;
// static constexpr float U2BF_2020 = 1.8814f;

static constexpr int V2R_2020 = 24160; // = V2RF*16384
static constexpr int U2G_2020 = -2696; // = U2GF*16384
static constexpr int V2G_2020 = -9361; // = V2GF*16384
static constexpr int U2B_2020 = 30825; // = U2BF*16384

// struct of conversion constants
RGB2YUVConstants rgb2yuv_601 = {R2Y_601, G2Y_601, B2Y_601, B2U_601, R2V_601};
YUV2RGBConstants yuv2rgb_601 = {U2B_601, U2G_601, V2G_601, V2R_601};

RGB2YUVConstants rgb2yuv_709 = {R2Y_709, G2Y_709, B2Y_709, B2U_709, R2V_709};
YUV2RGBConstants yuv2rgb_709 = {U2B_709, U2G_709, V2G_709, V2R_709};

RGB2YUVConstants rgb2yuv_2020 = {R2Y_2020, G2Y_2020, B2Y_2020, B2U_2020, R2V_2020};
YUV2RGBConstants yuv2rgb_2020 = {U2B_2020, U2G_2020, V2G_2020, V2R_2020};

static constexpr int yuv_shift = 14;
#define CV_DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))

template<class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void yuv_to_bgr_char_nhwc(SrcWrapper src, DstWrapper dst, int2 dstSize, int bidx,
                                     const YUV2RGBConstants cooef)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dstSize.x || dst_y >= dstSize.y)
        return;
    const int batch_idx = get_batch_idx();
    T         Y         = *src.ptr(batch_idx, dst_y, dst_x, 0);
    T         Cb        = *src.ptr(batch_idx, dst_y, dst_x, 1);
    T         Cr        = *src.ptr(batch_idx, dst_y, dst_x, 2);

    int C0 = cooef.V2R, C1 = cooef.V2G, C2 = cooef.U2G, C3 = cooef.U2B;
    int delta = ((T)(cuda::TypeTraits<T>::max / 2 + 1));

    int b = Y + CV_DESCALE((Cb - delta) * C3, yuv_shift);
    int g = Y + CV_DESCALE((Cb - delta) * C2 + (Cr - delta) * C1, yuv_shift);
    int r = Y + CV_DESCALE((Cr - delta) * C0, yuv_shift);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = cuda::SaturateCast<T>(b);
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = cuda::SaturateCast<T>(g);
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = cuda::SaturateCast<T>(r);
}

template<class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void bgr_to_yuv_char_nhwc(SrcWrapper src, DstWrapper dst, int2 dstSize, int bidx,
                                     const RGB2YUVConstants cooef)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dstSize.x || dst_y >= dstSize.y)
        return;
    const int batch_idx = get_batch_idx();
    int       B         = *src.ptr(batch_idx, dst_y, dst_x, bidx);
    int       G         = *src.ptr(batch_idx, dst_y, dst_x, 1);
    int       R         = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);

    int C0 = cooef.R2Y, C1 = cooef.G2Y, C2 = cooef.B2Y, C3 = cooef.R2V, C4 = cooef.B2U;
    int delta = ((T)(cuda::TypeTraits<T>::max / 2 + 1)) * (1 << yuv_shift);
    int Y     = CV_DESCALE(R * C0 + G * C1 + B * C2, yuv_shift);
    int V     = CV_DESCALE((R - Y) * C3 + delta, yuv_shift); //Cr
    int U     = CV_DESCALE((B - Y) * C4 + delta, yuv_shift); //Cb

    *dst.ptr(batch_idx, dst_y, dst_x, 0) = cuda::SaturateCast<T>(Y);
    *dst.ptr(batch_idx, dst_y, dst_x, 1) = cuda::SaturateCast<T>(U);
    *dst.ptr(batch_idx, dst_y, dst_x, 2) = cuda::SaturateCast<T>(V);
}

template<typename T>
__device__ __forceinline__ void yuv_to_bgr_kernel(const T &Y, const T &U, const T &V, T &r, T &g, T &b,
                                                  const YUV2RGBConstants cooef)
{
    int C0 = cooef.V2R, C1 = cooef.V2G, C2 = cooef.U2G, C3 = cooef.U2B;
    int delta = ((T)(cuda::TypeTraits<T>::max / 2 + 1));
    int B     = Y + CV_DESCALE((U - delta) * C3, yuv_shift);
    int G     = Y + CV_DESCALE((U - delta) * C2 + (V - delta) * C1, yuv_shift);
    int R     = Y + CV_DESCALE((V - delta) * C0, yuv_shift);

    b = cuda::SaturateCast<T>(B);
    g = cuda::SaturateCast<T>(G);
    r = cuda::SaturateCast<T>(R);
}

template<typename T>
__device__ __forceinline__ void bgr_to_yuv420_kernel(const T &r, const T &g, const T &b, int &y, int &u, int &v,
                                                     const RGB2YUVConstants cooef)
{
    int C0 = cooef.R2Y, C1 = cooef.G2Y, C2 = cooef.B2Y, C3 = cooef.R2V, C4 = cooef.B2U;

    y = CV_DESCALE((r * C0) + (g * C1) + (b * C2), yuv_shift);
    v = CV_DESCALE((r - y) * C3, yuv_shift); //Cr
    u = CV_DESCALE((b - y) * C4, yuv_shift); //Cb
}

template<class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void yuv420sp_to_bgr_char_nhwc(SrcWrapper src, DstWrapper dst, int2 dstSize, int dcn, int bidx, int uidx,
                                          const YUV2RGBConstants cooef)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dstSize.x || dst_y >= dstSize.y)
        return;
    const int batch_idx = get_batch_idx();
    int       uv_x      = (dst_x % 2 == 0) ? dst_x : (dst_x - 1);

    T Y = *src.ptr(batch_idx, dst_y, dst_x, 0);
    T U = *src.ptr(batch_idx, dstSize.y + dst_y / 2, uv_x + uidx);
    T V = *src.ptr(batch_idx, dstSize.y + dst_y / 2, uv_x + 1 - uidx);

    uint8_t r{0}, g{0}, b{0}, a{0xff};
    yuv_to_bgr_kernel<T>(Y, U, V, r, g, b, cooef);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = g;
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
    if (dcn == 4)
    {
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = a;
    }
}

template<class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void bgr_to_yuv420sp_char_nhwc(SrcWrapper src, DstWrapper dst, int2 srcSize, int scn, int bidx, int uidx,
                                          const RGB2YUVConstants cooef)
{
    int src_x = blockIdx.x * blockDim.x + threadIdx.x;
    int src_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (src_x >= srcSize.x || src_y >= srcSize.y)
        return;
    const int batch_idx = get_batch_idx();
    int       uv_x      = (src_x % 2 == 0) ? src_x : (src_x - 1);

    uint8_t b0 = static_cast<uint8_t>(*src.ptr(batch_idx, src_y, src_x, bidx));
    uint8_t g0 = static_cast<uint8_t>(*src.ptr(batch_idx, src_y, src_x, 1));
    uint8_t r0 = static_cast<uint8_t>(*src.ptr(batch_idx, src_y, src_x, bidx ^ 2));

    // compute Y for every pixel
    int Y0{0}, U0{0}, V0{0};
    bgr_to_yuv420_kernel<T>(r0, g0, b0, Y0, U0, V0, cooef);

    // Write the Y plane
    *dst.ptr(batch_idx, src_y, src_x, 0) = cuda::SaturateCast<T>(Y0);

    // compute U and V for every 2x2 block
    if (src_x >= srcSize.x - 1 || src_y >= srcSize.y - 1)
        return; //bail out since we need 2x2 block to compute U and V

    if (src_x % 2 || src_y % 2)
        return; //bail out since we need 2x2 block to compute U and V skip all odd pixels for u and v

    uint8_t b1 = static_cast<uint8_t>(*src.ptr(batch_idx, src_y + 0, src_x + 1, bidx));
    uint8_t g1 = static_cast<uint8_t>(*src.ptr(batch_idx, src_y + 0, src_x + 1, 1));
    uint8_t r1 = static_cast<uint8_t>(*src.ptr(batch_idx, src_y + 0, src_x + 1, bidx ^ 2));

    int Y1{0}, U1{0}, V1{0};
    bgr_to_yuv420_kernel<T>(r1, g1, b1, Y1, U1, V1, cooef);

    uint8_t b2 = static_cast<uint8_t>(*src.ptr(batch_idx, src_y + 1, src_x + 0, bidx));
    uint8_t g2 = static_cast<uint8_t>(*src.ptr(batch_idx, src_y + 1, src_x + 0, 1));
    uint8_t r2 = static_cast<uint8_t>(*src.ptr(batch_idx, src_y + 1, src_x + 0, bidx ^ 2));

    int Y2{0}, U2{0}, V2{0};
    bgr_to_yuv420_kernel<T>(r2, g2, b2, Y2, U2, V2, cooef);

    uint8_t b3 = static_cast<uint8_t>(*src.ptr(batch_idx, src_y + 1, src_x + 1, bidx));
    uint8_t g3 = static_cast<uint8_t>(*src.ptr(batch_idx, src_y + 1, src_x + 1, 1));
    uint8_t r3 = static_cast<uint8_t>(*src.ptr(batch_idx, src_y + 1, src_x + 1, bidx ^ 2));

    int Y3{0}, U3{0}, V3{0};
    bgr_to_yuv420_kernel<T>(r3, g3, b3, Y3, U3, V3, cooef);

    int delta = ((T)(cuda::TypeTraits<T>::max / 2 + 1)); // non scaled delta in this kernel

    *dst.ptr(batch_idx, srcSize.y + src_y / 2, uv_x + uidx) = cuda::SaturateCast<T>((U0 + U1 + U2 + U3) / 4 + delta);
    *dst.ptr(batch_idx, srcSize.y + src_y / 2, uv_x + (1 - uidx))
        = cuda::SaturateCast<T>((V0 + V1 + V2 + V3) / 4 + delta);
}

static bool isSemiPlanarToInterleaved(NVCVColorConversionCode code)
{
    switch (code)
    {
    case NVCV_COLOR_YUV2RGB_NV12:
    case NVCV_COLOR_YUV2BGR_NV12:
    case NVCV_COLOR_YUV2RGB_NV21:
    case NVCV_COLOR_YUV2BGR_NV21:
        return true;
    default:
        return false;
    }
}

static bool isInterleavedToSemiPlanar(NVCVColorConversionCode code)
{
    switch (code)
    {
    case NVCV_COLOR_RGB2YUV_NV12:
    case NVCV_COLOR_BGR2YUV_NV12:
    case NVCV_COLOR_RGB2YUV_NV21:
    case NVCV_COLOR_BGR2YUV_NV21:
        return true;
    default:
        return false;
    }
}

static bool isSupportedConversionCode(NVCVColorConversionCode code)
{
    switch (code)
    {
    case NVCV_COLOR_YUV2RGB_NV12: //420
    case NVCV_COLOR_YUV2BGR_NV12:
    case NVCV_COLOR_RGB2YUV_NV12:
    case NVCV_COLOR_BGR2YUV_NV12:
    case NVCV_COLOR_YUV2RGB_NV21:
    case NVCV_COLOR_YUV2BGR_NV21:
    case NVCV_COLOR_RGB2YUV_NV21:
    case NVCV_COLOR_BGR2YUV_NV21:
    case NVCV_COLOR_BGR2YUV: //444 types
    case NVCV_COLOR_RGB2YUV:
    case NVCV_COLOR_YUV2BGR:
    case NVCV_COLOR_YUV2RGB:
        return true;
    default:
        return false;
    }
}

static bool isSupportedColorSpec(nvcv::ColorSpec spec)
{
    //may need to be extended to check for conversion type
    switch (spec)
    {
    case NVCV_COLOR_SPEC_BT601:
    case NVCV_COLOR_SPEC_BT709:
    case NVCV_COLOR_SPEC_BT2020:
        return true;
    default:
        return false;
    }
}

static bool areCorrectSizes(const NVCVColorConversionCode code, const nvcv::TensorDataStridedCuda &t1,
                            const nvcv::TensorDataStridedCuda &t2)
{
    auto at1 = nvcv::TensorDataAccessStridedImagePlanar::Create(t1);
    auto at2 = nvcv::TensorDataAccessStridedImagePlanar::Create(t2);
    if (!at1 || !at2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    if (at1->numSamples() != at2->numSamples())
    {
        return false;
    }

    if (at1->numCols() != at2->numCols())
    {
        return false;
    }
    // NV12/21 are semi-planar, so the number of rows is different in the tensor by 1.5.
    if (at1->numRows() != at2->numRows() && !isSemiPlanarToInterleaved(code) && !isInterleavedToSemiPlanar(code))
    {
        return false;
    }

    // NV12/21 are semi-planar, so C is 1 for nHWC tensors.
    if (at1->numChannels() != at2->numChannels() && !isSemiPlanarToInterleaved(code)
        && !isInterleavedToSemiPlanar(code))
    {
        return false;
    }
    return true;
}

static bool checkInputOutputTensors(NVCVColorConversionCode code, const nvcv::TensorDataStridedCuda &in,
                                    const nvcv::TensorDataStridedCuda &out)
{
    if ((in.layout() == nvcv::TENSOR_NHWC || in.layout() == nvcv::TENSOR_HWC)
        && (out.layout() == nvcv::TENSOR_HWC || out.layout() == nvcv::TENSOR_NHWC))
    {
        return true;
    }
    return false;
}

static const RGB2YUVConstants &getRGB2YUVCooef(nvcv::ColorSpec spec)
{
    switch (spec)
    {
    case NVCV_COLOR_SPEC_BT601:
        return rgb2yuv_601;
    case NVCV_COLOR_SPEC_BT709:
        return rgb2yuv_709;
    case NVCV_COLOR_SPEC_BT2020:
        return rgb2yuv_2020;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unknown color spec");
    }
}

static const YUV2RGBConstants &getYUV2RGBCooef(nvcv::ColorSpec spec)
{
    switch (spec)
    {
    case NVCV_COLOR_SPEC_BT601:
        return yuv2rgb_601;
    case NVCV_COLOR_SPEC_BT709:
        return yuv2rgb_709;
    case NVCV_COLOR_SPEC_BT2020:
        return yuv2rgb_2020;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unknown color spec");
    }
}

namespace cvcuda::priv {

AdvCvtColor::AdvCvtColor() {}

void AdvCvtColor::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                             NVCVColorConversionCode code, nvcv::ColorSpec spec) const
{
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    //check compatibility
    if (isSupportedConversionCode(code) == false)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported conversion code");
    }

    if (isSupportedColorSpec(spec) == false)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported color spec");
    }

    if (checkInputOutputTensors(code, *inData, *outData) == false)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output tensors are not compatible");
    }

    if (areCorrectSizes(code, *inData, *outData) == false)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output tensors are not correct dims for conversion");
    }

    switch (code)
    {
    case NVCV_COLOR_YUV2BGR:
    case NVCV_COLOR_YUV2RGB:
        Yuv2Bgr(stream, *inData, *outData, code, spec);
        break;
    case NVCV_COLOR_BGR2YUV:
    case NVCV_COLOR_RGB2YUV:
        Bgr2Yuv(stream, *inData, *outData, code, spec);
        break;
    case NVCV_COLOR_YUV2RGB_NV12:
    case NVCV_COLOR_YUV2BGR_NV12:
    case NVCV_COLOR_YUV2RGB_NV21:
    case NVCV_COLOR_YUV2BGR_NV21:
        NvYuv2Bgr(stream, *inData, *outData, code, spec);
        break;
    case NVCV_COLOR_RGB2YUV_NV12:
    case NVCV_COLOR_BGR2YUV_NV12:
    case NVCV_COLOR_RGB2YUV_NV21:
    case NVCV_COLOR_BGR2YUV_NV21:
        Bgr2NvYuv(stream, *inData, *outData, code, spec);
        break;
    default:
        break;
    }
    return;
}

void AdvCvtColor::Yuv2Bgr(cudaStream_t stream, const nvcv::TensorDataStridedCuda &in,
                          const nvcv::TensorDataStridedCuda &out, NVCVColorConversionCode code,
                          nvcv::ColorSpec spec) const
{
    int bidx = code == NVCV_COLOR_YUV2BGR ? 0 : 2; // 0 for BGR, 2 for RGB

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(in);
    NVCV_ASSERT(inAccess);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(out);
    NVCV_ASSERT(outAccess);

    nvcv::legacy::cuda_op::DataType  inDataType  = nvcv::legacy::helpers::GetLegacyDataType(inAccess->dtype());
    nvcv::legacy::cuda_op::DataShape inputShape  = nvcv::legacy::helpers::GetLegacyDataShape(inAccess->infoShape());
    nvcv::legacy::cuda_op::DataShape outputShape = nvcv::legacy::helpers::GetLegacyDataShape(outAccess->infoShape());
    int2                             dstSize{outputShape.W, outputShape.H};

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(legacy::divUp(inputShape.W, blockSize.x), legacy::divUp(inputShape.H, blockSize.y), inputShape.N);

    switch (inDataType)
    {
    case legacy::kCV_8U:
    {
        const YUV2RGBConstants &cooef        = getYUV2RGBCooef(spec);
        auto                    outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
        auto                    inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
        if (std::max(outMaxStride, inMaxStride) <= cuda::TypeTraits<int32_t>::max)
        {
            auto srcWrap = cuda::CreateTensorWrapNHWC<uint8_t, int32_t>(in);
            auto dstWrap = cuda::CreateTensorWrapNHWC<uint8_t, int32_t>(out);
            yuv_to_bgr_char_nhwc<<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, cooef);
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                                  cuda::TypeTraits<int32_t>::max);
        }
        checkKernelErrors();
    }
    break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
    }
    return;
}

void AdvCvtColor::Bgr2Yuv(cudaStream_t stream, const nvcv::TensorDataStridedCuda &in,
                          const nvcv::TensorDataStridedCuda &out, NVCVColorConversionCode code,
                          nvcv::ColorSpec spec) const
{
    int bidx = code == NVCV_COLOR_BGR2YUV ? 0 : 2; // 0 for BGR, 2 for RGB

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(in);
    NVCV_ASSERT(inAccess);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(out);
    NVCV_ASSERT(outAccess);

    nvcv::legacy::cuda_op::DataType  inDataType  = nvcv::legacy::helpers::GetLegacyDataType(inAccess->dtype());
    nvcv::legacy::cuda_op::DataShape inputShape  = nvcv::legacy::helpers::GetLegacyDataShape(inAccess->infoShape());
    nvcv::legacy::cuda_op::DataShape outputShape = nvcv::legacy::helpers::GetLegacyDataShape(outAccess->infoShape());

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(legacy::divUp(inputShape.W, blockSize.x), legacy::divUp(inputShape.H, blockSize.y), inputShape.N);

    int2 dstSize{outputShape.W, outputShape.H};

    switch (inDataType)
    {
    case legacy::kCV_8U:
    {
        const RGB2YUVConstants &cooef        = getRGB2YUVCooef(spec);
        auto                    outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
        auto                    inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
        if (std::max(outMaxStride, inMaxStride) <= cuda::TypeTraits<int32_t>::max)
        {
            auto srcWrap = cuda::CreateTensorWrapNHWC<uint8_t, int32_t>(in);
            auto dstWrap = cuda::CreateTensorWrapNHWC<uint8_t, int32_t>(out);
            bgr_to_yuv_char_nhwc<<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, cooef);
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                                  cuda::TypeTraits<int32_t>::max);
        }
        checkKernelErrors();
    }
    break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
    }
    return;
}

void AdvCvtColor::NvYuv2Bgr(cudaStream_t stream, const nvcv::TensorDataStridedCuda &in,
                            const nvcv::TensorDataStridedCuda &out, NVCVColorConversionCode code,
                            nvcv::ColorSpec spec) const
{
    // blue index
    int bidx = (code == NVCV_COLOR_YUV2BGR_NV12 || code == NVCV_COLOR_YUV2BGR_NV21) ? 0 : 2;

    // u index
    int uidx = (code == NVCV_COLOR_YUV2BGR_NV12 || code == NVCV_COLOR_YUV2RGB_NV12) ? 0 : 1;

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(in);
    NVCV_ASSERT(inAccess);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(out);
    NVCV_ASSERT(outAccess);

    nvcv::legacy::cuda_op::DataType  inDataType  = nvcv::legacy::helpers::GetLegacyDataType(inAccess->dtype());
    nvcv::legacy::cuda_op::DataShape inputShape  = nvcv::legacy::helpers::GetLegacyDataShape(inAccess->infoShape());
    nvcv::legacy::cuda_op::DataShape outputShape = nvcv::legacy::helpers::GetLegacyDataShape(outAccess->infoShape());

    if (inputShape.H % 3 != 0 || inputShape.W % 2 != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input shape");
    }
    if (outputShape.C != 3 && outputShape.C != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported output shape channels");
    }
    if (inputShape.C != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input shape channels");
    }

    int rgb_width  = inputShape.W;
    int rgb_height = inputShape.H * 2 / 3;

    if (outputShape.H != rgb_height || outputShape.W != rgb_width || outputShape.N != inputShape.N)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "invalid output shape given input");
    }

    dim3 blockSize(BLOCK, BLOCK / 1, 1);
    dim3 gridSize(legacy::divUp(rgb_width, blockSize.x), legacy::divUp(rgb_height, blockSize.y), inputShape.N);
    int2 dstSize{outputShape.W, outputShape.H};
    int  dcn = outputShape.C;

    switch (inDataType)
    {
    case legacy::kCV_8U:
    {
        const YUV2RGBConstants &cooef        = getYUV2RGBCooef(spec);
        auto                    outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
        auto                    inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
        if (std::max(outMaxStride, inMaxStride) <= cuda::TypeTraits<int32_t>::max)
        {
            auto srcWrap = cuda::CreateTensorWrapNHWC<uint8_t, int32_t>(in);
            auto dstWrap = cuda::CreateTensorWrapNHWC<uint8_t, int32_t>(out);
            yuv420sp_to_bgr_char_nhwc<<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, dcn, bidx, uidx,
                                                                          cooef);
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                                  cuda::TypeTraits<int32_t>::max);
        }
        checkKernelErrors();
    }
    break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
    }
    return;
}

void AdvCvtColor::Bgr2NvYuv(cudaStream_t stream, const nvcv::TensorDataStridedCuda &in,
                            const nvcv::TensorDataStridedCuda &out, NVCVColorConversionCode code,
                            nvcv::ColorSpec spec) const
{
    // blue index
    int bidx = (code == NVCV_COLOR_BGR2YUV_NV12 || code == NVCV_COLOR_BGR2YUV_NV21) ? 0 : 2;

    // u index
    int uidx = (code == NVCV_COLOR_BGR2YUV_NV12 || code == NVCV_COLOR_RGB2YUV_NV12) ? 0 : 1;

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(in);
    NVCV_ASSERT(inAccess);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(out);
    NVCV_ASSERT(outAccess);

    nvcv::legacy::cuda_op::DataType  inDataType  = nvcv::legacy::helpers::GetLegacyDataType(inAccess->dtype());
    nvcv::legacy::cuda_op::DataShape inputShape  = nvcv::legacy::helpers::GetLegacyDataShape(inAccess->infoShape());
    nvcv::legacy::cuda_op::DataShape outputShape = nvcv::legacy::helpers::GetLegacyDataShape(outAccess->infoShape());

    if (inputShape.C != 3 && inputShape.C != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input shape");
    }
    if (inputShape.H % 2 != 0 || inputShape.W % 2 != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input shape");
    }

    int yuv420_width  = inputShape.W;
    int yuv420_height = inputShape.H / 2 * 3;

    if (outputShape.H != yuv420_height || outputShape.W != yuv420_width || outputShape.N != inputShape.N)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported output shape given input");
    }

    int2 srcSize{inputShape.W, inputShape.H};
    dim3 blockSize(BLOCK, BLOCK / 1, 1);
    dim3 gridSize(legacy::divUp(inputShape.W, blockSize.x), legacy::divUp(inputShape.H, blockSize.y), inputShape.N);

    switch (inDataType)
    {
    case legacy::kCV_8U:
    {
        const RGB2YUVConstants &cooef        = getRGB2YUVCooef(spec);
        auto                    outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
        auto                    inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
        if (std::max(outMaxStride, inMaxStride) <= cuda::TypeTraits<int32_t>::max)
        {
            auto srcWrap = cuda::CreateTensorWrapNHWC<uint8_t, int32_t>(in);
            auto dstWrap = cuda::CreateTensorWrapNHWC<uint8_t, int32_t>(out);
            bgr_to_yuv420sp_char_nhwc<<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, srcSize, inputShape.C, bidx,
                                                                          uidx, cooef);
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                                  cuda::TypeTraits<int32_t>::max);
        }
        checkKernelErrors();
    }
    break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported data type");
    }
    return;
}

} // namespace cvcuda::priv
