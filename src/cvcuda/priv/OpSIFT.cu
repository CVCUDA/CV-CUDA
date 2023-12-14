/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OpSIFT.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/InterpolationWrap.hpp>
#include <nvcv/cuda/MathOps.hpp>
#include <nvcv/cuda/MathWrappers.hpp>
#include <nvcv/cuda/StaticCast.hpp>
#include <nvcv/cuda/TensorWrap.hpp>
#include <nvcv/cuda/math/LinAlg.hpp>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

#include <cmath>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

// Compile-time definitions ----------------------------------------------------

// Macro used to check if an internal object (payload) data is valid
#define CVCUDA_SIFT_INTERNAL_DATA_CHECK(data)                                              \
    if (!data)                                                                             \
    {                                                                                      \
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "Invalid SIFT internal data"); \
    }

constexpr NVCVBorderType        kBorderGauss{NVCV_BORDER_REFLECT101}; // default border for Gaussian operation
constexpr NVCVBorderType        kBorderInterp{NVCV_BORDER_REPLICATE}; // default border for interpolation
constexpr NVCVInterpolationType kInterpUp{NVCV_INTERP_LINEAR};        // default interpolation for up scaling

constexpr float kMinSigma              = 0.01f;      // minimum sigma to be used as base sigma
constexpr float kPrevSigma             = 0.5f;       // previous sigma before base image
constexpr int   kMaxKernelSize         = 59;         // maximum Gaussian kernel size
constexpr int2  kBorderSize            = int2{5, 5}; // ignore keypoints close to the border
constexpr int   kMaxInterpolationSteps = 5;          // max. steps of keypoint interpolation before failure

constexpr float kRadiansToDegrees   = 180.f / M_PI;           // convert from radians to degrees
constexpr float kDegreesToRadians   = M_PI / 180.f;           // convert from degrees to radians
constexpr float kOrientationSigma   = 1.5f;                   // sigma used for angle orientation computation
constexpr float kOrientationRadius  = 3 * kOrientationSigma;  // radius of the region to compute angle orientation
constexpr float kHistogramPeakRatio = 0.8f;                   // histogram peak ratio to create new feature
constexpr int   kHistogramBins      = 36;                     // number of histogram bins, impacts register pressure
constexpr float kAngleToBin         = kHistogramBins / 360.f; // angle in degrees to a bin index in histogram
constexpr float kBinToAngle         = 360.f / kHistogramBins; // bin index to angle in degrees

constexpr int   kDescOriRadius     = 3;                                // radius for orientation in descriptor hist.
constexpr int   kDescWidth         = 4;                                // width of the descriptor
constexpr int   kDescHistBins      = 8;                                // number of bins per histogram in descriptor
constexpr float kAngleToDescBin    = kDescHistBins / 360.f;            // angle in degrees to a desc. hist. bin index
constexpr float kDescWeightScale   = -2.f / (kDescWidth * kDescWidth); // weight scale in descriptor histogram
constexpr float kDescWidthToRadius = M_SQRT2 * (kDescWidth + 1) * .5f; // convert from histogram width to radius
constexpr int   kDescMaxRadius     = 51;                               // maximum radius considering max. initSigma=2.4
constexpr float kDescHistPeakRatio = .2f;                              // maximum allowed magnitude in descriptor
constexpr int   kDescF32toU8Ratio  = 512;                              // convert desc. value from float to uint8_t

constexpr int kDescSize          = kDescWidth * kDescWidth * kDescHistBins;                   // desc. size (=128B)
constexpr int kDescHistTotalBins = (kDescWidth + 2) * (kDescWidth + 2) * (kDescHistBins + 2); // desc. hist. size

// Tensor wrap for LNHWC tensors with C inside its type
template<typename T>
using TensorWrapLNHW = cuda::TensorWrap<T, -1, -1, -1, sizeof(T)>;

// Border wrap for LNHWC tensors using the corresponding tensor wrap
template<typename T>
using BorderWrapLNHW = cuda::BorderWrap<TensorWrapLNHW<T>, kBorderGauss, false, false, true, true>;

// Tensor wrap for descriptor has 2 compile-time strides: per 128B descriptor and per 1B within descriptor
using TensorWrapForDescriptor = cuda::TensorWrap<uint8_t, -1, kDescSize * sizeof(uint8_t), sizeof(uint8_t)>;

// CPU functions ---------------------------------------------------------------

// Function to get a view from a tensor parent data with given shape
inline __host__ nvcv::Tensor GetViewFrom(const nvcv::TensorDataStridedCuda  &parentData,
                                         const nvcv::TensorShape::ShapeType &viewShape)
{
    // Get tensor data and create a view of it with given shape instead of its original shape
    NVCVTensorData viewData = parentData.cdata();

    NVCV_ASSERT(viewData.rank == viewShape.rank());

    for (int r = 0; r < viewData.rank; r++)
    {
        NVCV_ASSERT(viewData.shape[r] >= viewShape[r]);

        viewData.shape[r] = viewShape[r];
    }

    return nvcv::TensorWrapData(viewData);
}

// Function to compute number of octaves for an WxH image with W=width and H=height
inline __host__ int ComputeNumberOfOctaves(int width, int height)
{
    return std::round((std::log2(std::min(width, height))) - 2) + 1;
}

// Compute Gaussian filter kernel size for a given sigma
inline __host__ int ComputeGaussianKernelSize(float sigma)
{
    return cuda::min(cuda::round<int>(sigma * 8 + 1) | 1, kMaxKernelSize);
}

// CUDA functions --------------------------------------------------------------

// Direct (no scale) copy a source 3-rank NHW tensor into a destination 4-rank LNHW tensor
template<class SrcWrapper>
__global__ void Copy(TensorWrapLNHW<float> dst, SrcWrapper src, int3 srcShape)
{
    int3 srcCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (srcCoord.x < srcShape.x && srcCoord.y < srcShape.y && srcCoord.z < srcShape.z)
    {
        int4 dstCoord{srcCoord.x, srcCoord.y, srcCoord.z, 0};

        dst[dstCoord] = src[srcCoord];
    }
}

// Expand (2x upscale) copy a source 3-rank NHW tensor into a destination 4-rank LNHW tensor
template<class SrcWrapper>
__global__ void UpCopy(TensorWrapLNHW<float> dst, SrcWrapper src, int3 dstShape)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (coord.x < dstShape.x && coord.y < dstShape.y && coord.z < dstShape.z)
    {
        int4   dstCoord{coord.x, coord.y, coord.z, 0};
        float3 srcCoord{0, 0, (float)coord.z};

        srcCoord.x = (coord.x + .5f) * .5f - .5f;
        srcCoord.y = (coord.y + .5f) * .5f - .5f;

        dst[dstCoord] = src[srcCoord];
    }
}

// Contract (2x downscale) copy a source 4-rank LNHW tensor into a destination 4-rank LNHW tensor
template<class SrcWrapper>
__global__ void DownCopy(TensorWrapLNHW<float> dst, SrcWrapper src, int3 dstShape, int srcLayer)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (coord.x < dstShape.x && coord.y < dstShape.y && coord.z < dstShape.z)
    {
        int4 dstCoord{coord.x, coord.y, coord.z, 0};
        int4 srcCoord{coord.x * 2, coord.y * 2, coord.z, srcLayer};

        dst[dstCoord] = src[srcCoord];
    }
}

// Compute Gaussian and DoG (Difference of Gaussians) SIFT pyramids of current octave and layer
template<int BW, int BH>
__global__ void DoComputePyramids(BorderWrapLNHW<float> prevGauss, TensorWrapLNHW<float> currGauss,
                                  TensorWrapLNHW<float> currDoG, int3 currShape, int currLayer, int currKernelSize,
                                  cuda::math::Vector<float, kMaxKernelSize> gaussSepKernel)
{
    constexpr int SW = BW + kMaxKernelSize; // block width with kernel maximum support (halo) for Gaussian filtering
    constexpr int SH = BH + kMaxKernelSize; // block height with kernel maximum support (halo) for Gaussian filtering

    // Based on kMaxKernelSize = 59 and kDataTile = 32x32x1
    // Registers used: 40
    // SMEM usage (max. 48KB): 44772 B = ((32 + 59) * (32 + 59)) * 4 + (32 * (32 + 59)) * 4
    // CMEM usage (max. 4KB): 688 B = 59 * 4 + 4 + 4 + 4 + 4 * 3 + (4 * 3 + 8) * 2 + (4 * 3 + 4 * 2 + 8) + (360)

    __shared__ float gaussInData[SH * SW];  // plain 1D Gaussian input data in shared memory (SMEM)
    __shared__ float gaussOutData[SH * BW]; // plain 1D Gaussian output (intermediary) data in SMEM

    // Using TensorWrap with compile-time strides for easy multi-dimensional access of Gaussian data in SMEM
    cuda::TensorWrap<float, SW * sizeof(float), sizeof(float)> gaussIn(&gaussInData[0]);
    cuda::TensorWrap<float, BW * sizeof(float), sizeof(float)> gaussOut(&gaussOutData[0]);

    int half = currKernelSize / 2; // i.e. the halo or support data outside block to compute Gaussian filter

    int2 tc; // thread (local) coordinates to access SMEM data
    int2 pc; // previous (local) coordinates to access SMEM data (previous of thread)
    int2 mc; // mirrored (local) coordinates to access SMEM data (mirror of previous)

    int4 gc{0, 0, (int)blockIdx.z, cuda::max(0, currLayer - 1)}; // current (global) coordinates for GMEM data

    float result; // temporary Gaussian filtering result

    // Threads are defined in a 1D block dimension to be just a count of threads to run this kernel
    // First loop uses all threads reshaping them to 2D to load block with halo from global memory (GMEM) into SMEM
    for (int ti = threadIdx.x; ti < (BW + half * 2) * (BH + half * 2); ti += blockDim.x)
    {
        tc.x = ti % (BW + half * 2);
        tc.y = ti / (BW + half * 2);
        gc.x = blockIdx.x * BW - half + tc.x;
        gc.y = blockIdx.y * BH - half + tc.y;

        // The prevGauss is a border wrap allowing access to halo outside image with proper border treatment
        gaussIn[tc] = prevGauss[gc];
    }

    // Second loop runs all threads in 2D to zero out the block with halo in height for intermediary Gaussian output
    for (int ti = threadIdx.x; ti < BW * (BH + half * 2); ti += blockDim.x)
    {
        tc.x = ti % BW;
        tc.y = ti / BW;

        gaussOut[tc] = 0.f;
    }

    __syncthreads(); // wait for the input Gaussian data to be loaded and intermediary Gaussian output to be zeroed

    // SIFT Gaussian pyramid has resolution levels as octaves with scale-space layers, where the first octave
    // only stores one layer as the base image (octave=0, layer=0) and all next octaves do Gaussian filtering.
    // [ O=0: [ L=0: base image ] O=1: [ L=0: Gaussian of O=0 L=0; L=1: Gaussian of O=1 L=0; ... ]
    //   O=2: [ L=0: copy of O=1 L=numOctaveLayers; L=1: Gaussian of O=2 L=0; ... ] ... ] <- Gaussian pyramid

    // Third loop runs all threads in 2D to compute block with halo in height with intermediary Gaussian output
    for (int ti = threadIdx.x; ti < BW * (BH + half * 2); ti += blockDim.x)
    {
        tc.x = ti % BW;
        tc.y = ti / BW;
        pc.y = tc.y;
        pc.x = tc.x + half;
        mc.x = 0;
        mc.y = tc.y;

        result = gaussIn[pc] * gaussSepKernel[half];

        for (int kx = 0; kx < half; ++kx)
        {
            pc.x = tc.x + kx;
            mc.x = tc.x + 2 * half - kx;

            result += (gaussIn[pc] + gaussIn[mc]) * gaussSepKernel[kx]; // first separable convolution
        }

        gaussOut[tc] = result;
    }

    __syncthreads(); // wait for the intermediary (separable) Gaussian filtering output data to be ready

    // SIFT DoG pyramid is the Difference of Gaussians: next layer - current layer; it uses next octave from
    // the Gaussian pyramid since it stores the base image without Gaussian in its first octave.
    // [ O=0: [ L=0: DoG of GaussianPyr (O=1 L=1) - (O=1 L=0); L=1: DoG of GaussianPyr (O=1 L=2) - (O=1 L=1); ... ]
    //   O=1: [ L=0: DoG of GaussianPyr (O=2 L=1) - (O=2 L=0); ... ] ] <- DoG pyramid

    // Fourth loop runs threads in 2D to compute final block of Gaussian filter and DoG
    for (int ti = threadIdx.x; ti < BW * BH; ti += blockDim.x)
    {
        tc.x = ti % BW;
        tc.y = ti / BW;
        gc.x = blockIdx.x * BW + tc.x;
        gc.y = blockIdx.y * BH + tc.y;

        if (gc.x >= currShape.x || gc.y >= currShape.y)
        {
            continue;
        }

        gc.w = currLayer;
        pc.x = tc.x;
        pc.y = tc.y + half;
        mc.x = tc.x;
        mc.y = 0;

        result = gaussOut[pc] * gaussSepKernel[half];

        for (int ky = 0; ky < half; ++ky)
        {
            pc.y = tc.y + ky;
            mc.y = tc.y + 2 * half - ky;

            result += (gaussOut[pc] + gaussOut[mc]) * gaussSepKernel[ky]; // second separable convolution
        }

        currGauss[gc] = result;

        if (gc.w > 0)
        {
            gc.w -= 1;
            pc.x = half + tc.x;
            pc.y = half + tc.y;

            currDoG[gc] = result - gaussIn[pc];
        }
    }
}

// Get adjacent neighbor coordinates of center coordinates and deltas d* with x=col, y=row
__forceinline__ __device__ int2 adj(const int2 &center, int dRow, int dCol)
{
    return int2{center.x + dCol, center.y + dRow};
}

// Get adjacent neighbor coordinates of center coordinates and deltas d* with x=col, y=row, z=batch, w=layer
__forceinline__ __device__ int4 adj(const int4 &center, int dLayer, int dRow, int dCol)
{
    return int4{center.x + dCol, center.y + dRow, center.z, center.w + dLayer};
}

// Compute descriptors, using the Gaussian pyramid, the previously computed angle and feature radius and coordinates
__global__ void DoComputeDescriptors(TensorWrapForDescriptor featDescriptors, cuda::Tensor2DWrap<float4> featCoords,
                                     cuda::Tensor2DWrap<float3> featMetadata, cuda::Tensor1DWrap<int> numFeatures,
                                     TensorWrapLNHW<const float> currGauss, int3 currShape, int featOctave,
                                     float unscaleOctave)
{
    constexpr int BW = (kDescMaxRadius + 1) * 2 + 1; // block width with maximum support radius for descriptor
    constexpr int BH = (kDescMaxRadius + 1) * 2 + 1; // block height with maximum support radius for descriptor

    // Based on kDescMaxRadius = 51
    // Registers usage: 40
    // SMEM usage (max. 48KB): 45540 B = ((51 + 1) * 2 + 1) * ((51 + 1) * 2 + 1) * 4 + 360 * 4
    // CMEM usage (max. 4KB): 452 B = (4 * 1 + 8) * 3 + 8 + (4 * 3 + 8) * 1 + 4 * 3 + 4 + 4 + (368)
    // ! 32 Bytes stack frame

    __shared__ float gaussInData[BH * BW];          // Gaussian input data to compute descriptors
    __shared__ float histogram[kDescHistTotalBins]; // Histogram with intermediary output for descriptors

    // Using TensorWrap with compile-time strides for easy multi-dimensional access of Gaussian data in SMEM
    cuda::TensorWrap<float, BW * sizeof(float), sizeof(float)> gaussIn(&gaussInData[0]);

    int featIdx   = blockIdx.x; // each block thru x computes one feature descriptor
    int sampleIdx = blockIdx.z; // each block thru z computes one image sample

    if (featIdx >= numFeatures[sampleIdx])
    {
        return; // each kernel invocation handles only valid features
    }

    float *pFeatCoords = reinterpret_cast<float *>(featCoords.ptr(sampleIdx, featIdx));
    if (pFeatCoords[2] != featOctave)
    {
        return; // each kernel invocation handles only features of the current octave
    }

    // The coordinate (x, y) and layer of the feature to compute descriptor from
    int2 featCoord{cuda::round<int>(pFeatCoords[0] * unscaleOctave), cuda::round<int>(pFeatCoords[1] * unscaleOctave)};
    int  featLayer{cuda::round<int>(pFeatCoords[3])};

    // Feature metadata at 2 provides size as diameter that has to be halved and uncaled to get descriptor radius
    float *pFeatMetadata = reinterpret_cast<float *>(featMetadata.ptr(sampleIdx, featIdx));
    float  histWidth     = kDescOriRadius * (pFeatMetadata[2] * .5f * unscaleOctave);
    int    radius        = cuda::round<int>(histWidth * kDescWidthToRadius);

    // Clamp radius by maximum allowed radius (impacts SMEM resource usage)
    if (radius > kDescMaxRadius)
    {
        radius    = kDescMaxRadius;
        histWidth = kDescMaxRadius / kDescWidthToRadius;
    }

    // Threads are defined in a 1D block dimension to be just a count of threads to run this kernel
    // First loop runs threads reshaping them to 2D block of radius + 1-halo around current feature
    for (int ti = threadIdx.x; ti < ((radius + 1) * 2 + 1) * ((radius + 1) * 2 + 1); ti += blockDim.x)
    {
        int2 tc{ti % ((radius + 1) * 2 + 1), ti / ((radius + 1) * 2 + 1)};
        int4 gc{featCoord.x - radius - 1 + tc.x, featCoord.y - radius - 1 + tc.y, sampleIdx, featLayer};

        if (gc.x < 0 || gc.x >= currShape.x || gc.y < 0 || gc.y >= currShape.y)
        {
            continue;
        }

        gaussIn[tc] = currGauss[gc]; // load from global memory (GMEM) into shared memory (SMEM)
    }

    // Second loop runs threads to zero out descriptor histogram
    for (int ti = threadIdx.x; ti < kDescHistTotalBins; ti += blockDim.x)
    {
        histogram[ti] = 0.f;
    }

    float featAngle = 360.f - pFeatMetadata[0]; // feature metadata at zero provides angle

    if (cuda::abs(featAngle - 360.f) < 1e-5)
    {
        featAngle = 0.f;
    }

    float cos_a = ::cosf(featAngle * kDegreesToRadians) / histWidth;
    float sin_a = ::sinf(featAngle * kDegreesToRadians) / histWidth;

    __syncthreads(); // wait for the histogram to be zeroed and for the R-tile Gaussian data with 1-halo to be loaded

    // Third loop runs threads over radius around feature without 1-halo
    for (int ti = threadIdx.x; ti < (radius * 2 + 1) * (radius * 2 + 1); ti += blockDim.x)
    {
        int j = ti % (radius * 2 + 1) - radius;
        int i = ti / (radius * 2 + 1) - radius;

        int2 tc{1 + radius + j, 1 + radius + i}; // thread (local) coordinate to access SMEM

        if (featCoord.x + j <= 0 || featCoord.x + j >= currShape.x - 1 || featCoord.y + i <= 0
            || featCoord.y + i >= currShape.y - 1)
        {
            continue; // if any neighbor is outside current shape
        }

        float r_rot  = j * sin_a + i * cos_a;         // rotate row index
        float c_rot  = j * cos_a - i * sin_a;         // rotate column index
        float weight = r_rot * r_rot + c_rot * c_rot; // weight (squared distance) for vector magnitude

        r_rot += kDescWidth / 2 - .5f;
        c_rot += kDescWidth / 2 - .5f;
        if (r_rot <= -1 || r_rot >= kDescWidth || c_rot <= -1 || c_rot >= kDescWidth)
        {
            continue; // if normalized row or column index is outside descriptor width
        }

        int iHist = cuda::round<cuda::RoundMode::DOWN, int>(r_rot); // i index in histogram
        int jHist = cuda::round<cuda::RoundMode::DOWN, int>(c_rot); // j index in histogram
        r_rot -= iHist;
        c_rot -= jHist;

        float dx = gaussIn[adj(tc, +0, +1)] - gaussIn[adj(tc, +0, -1)]; // Gaussian delta x
        float dy = gaussIn[adj(tc, -1, +0)] - gaussIn[adj(tc, +1, +0)]; // Gaussian delta y

        float o_rot = ::atan2f(dy, dx) * kRadiansToDegrees; // rotate orientation in degrees
        if (o_rot < 0.f)
            o_rot += 360.f;
        if (o_rot >= 360.f)
            o_rot -= 360.f;
        o_rot = (o_rot - featAngle) * kAngleToDescBin;

        int bin = cuda::round<cuda::RoundMode::DOWN, int>(o_rot); // rotate index for histogram bin
        o_rot -= bin;
        if (bin < 0)
            bin += kDescHistBins;
        if (bin >= kDescHistBins)
            bin -= kDescHistBins;

        weight = ::exp2f(weight * kDescWeightScale); // weighted scale for vector magnitude

        float magnitude = ::sqrtf(dx * dx + dy * dy) * weight; // neighbor vector magnitude

        // Vector values to be added to the descriptor histogram
        float v_r1     = magnitude * r_rot;
        float v_r0     = magnitude - v_r1;
        float v_rc11   = v_r1 * c_rot;
        float v_rc10   = v_r1 - v_rc11;
        float v_rc01   = v_r0 * c_rot;
        float v_rc00   = v_r0 - v_rc01;
        float v_rco111 = v_rc11 * o_rot;
        float v_rco110 = v_rc11 - v_rco111;
        float v_rco101 = v_rc10 * o_rot;
        float v_rco100 = v_rc10 - v_rco101;
        float v_rco011 = v_rc01 * o_rot;
        float v_rco010 = v_rc01 - v_rco011;
        float v_rco001 = v_rc00 * o_rot;
        float v_rco000 = v_rc00 - v_rco001;

        int idx = ((iHist + 1) * (kDescWidth + 2) + (jHist + 1)) * (kDescHistBins + 2) + bin;

        // Atomic adding to the histogram as other threads may hit same (rotated) location
        atomicAdd(&histogram[idx], v_rco000);
        atomicAdd(&histogram[idx + 1], v_rco001);
        atomicAdd(&histogram[idx + (kDescHistBins + 2)], v_rco010);
        atomicAdd(&histogram[idx + (kDescHistBins + 3)], v_rco011);
        atomicAdd(&histogram[idx + (kDescWidth + 2) * (kDescHistBins + 2)], v_rco100);
        atomicAdd(&histogram[idx + (kDescWidth + 2) * (kDescHistBins + 2) + 1], v_rco101);
        atomicAdd(&histogram[idx + (kDescWidth + 3) * (kDescHistBins + 2)], v_rco110);
        atomicAdd(&histogram[idx + (kDescWidth + 3) * (kDescHistBins + 2) + 1], v_rco111);
    }

    __syncthreads(); // wait for the histogram to be written with initial vector values

    if (threadIdx.x != 0)
    {
        return; // at this point only one thread updates and uses the histogram to compute the feature descriptor
    }

    float norm = 0.f;

    // Complete histogram as it is a circular histogram and compute its norm
    for (int i = 0; i < kDescWidth; i++)
    {
        for (int j = 0; j < kDescWidth; j++)
        {
            int histIdx = ((i + 1) * (kDescWidth + 2) + (j + 1)) * (kDescHistBins + 2);

            histogram[histIdx] += histogram[histIdx + kDescHistBins];
            histogram[histIdx + 1] += histogram[histIdx + kDescHistBins + 1];

            for (int bin = 0; bin < kDescHistBins; bin++)
            {
                float v = histogram[histIdx + bin];

                norm += v * v;
            }
        }
    }

    // Hysteresis threshold the histogram by its norm

    float histMax = ::sqrtf(norm) * kDescHistPeakRatio;

    norm = 0.f;

    for (int i = 0; i < kDescWidth; i++)
    {
        for (int j = 0; j < kDescWidth; j++)
        {
            int histIdx = ((i + 1) * (kDescWidth + 2) + (j + 1)) * (kDescHistBins + 2);

            for (int bin = 0; bin < kDescHistBins; bin++)
            {
                float v = histogram[histIdx + bin];

                v = cuda::min(v, histMax);

                norm += v * v;

                histogram[histIdx + bin] = v;
            }
        }
    }

    // Compute the feature descriptor by using its histogram

    norm = kDescF32toU8Ratio / cuda::max(::sqrtf(norm), 1e-5f);

    for (int i = 0; i < kDescWidth; i++)
    {
        for (int j = 0; j < kDescWidth; j++)
        {
            int histIdx = ((i + 1) * (kDescWidth + 2) + (j + 1)) * (kDescHistBins + 2);

            for (int bin = 0; bin < kDescHistBins; bin++)
            {
                float v = histogram[histIdx + bin];

                int descIdx = (i * kDescWidth + j) * kDescHistBins + bin;

                *featDescriptors.ptr(sampleIdx, featIdx, descIdx) = cuda::SaturateCast<uint8_t>(v * norm);
            }
        }
    }
}

// Find extrema (feature coordinates + metadata) using Gaussian + DoG (Difference of Gaussians) pyramids
template<int BW, int BH, typename DT>
__global__ void DoFindExtrema(cuda::Tensor2DWrap<float4> featCoords, cuda::Tensor2DWrap<float3> featMetadata,
                              int maxCapacity, cuda::Tensor1DWrap<int> numFeatures,
                              TensorWrapLNHW<const float> currGauss, TensorWrapLNHW<const float> currDoG,
                              int3 currShape, int featOctave, float scaleOctave, int numOctaveLayers, int thr,
                              float contrastThreshold, float edgeThreshold, float initSigma)
{
    constexpr float kImageScale = 1.f / cuda::TypeTraits<DT>::max; // source images data type scale
    constexpr float kDScale1    = kImageScale * .5f;               // first derivative scale
    constexpr float kDScale2    = kImageScale;                     // second derivative scale
    constexpr float kDScaleC    = kImageScale * .25f;              // cross derivative scale

    // Registers usage: 68
    // SMEM usage (max. 48KB): 0 B
    // CMEM usage (max. 4KB): 488 B = (4 * 1 + 8) * 2 + 4 + 8 + (4 * 3 + 8) * 2 + 4 * 3 + 4 * 7 + (372)
    // ! 218072 bytes gmem ! 208 Bytes stack frame

    float                           cv;      // central value
    cuda::math::Vector<float, 3>    dD, sol; // derivative distances and solver solution
    cuda::math::Matrix<float, 3, 3> H;       // Hessian matrix

    float hist[kHistogramBins]; // histogram for angle computation

    int4 gc{0, 0, (int)blockIdx.z, 0}; // global (at tensor level) coordinate to access global memory (GMEM)

    // Threads are defined in a 1D block dimension to be just a count of threads to run this kernel
    // A single loop uses all threads reshaping them to 2D block to find extrema in GMEM
    for (int ti = threadIdx.x; ti < BW * BH; ti += blockDim.x)
    {
        gc.x = blockIdx.x * BW + (ti % BW);
        gc.y = blockIdx.y * BH + (ti / BW);

        if (gc.x < kBorderSize.x || gc.x >= currShape.x - kBorderSize.x || gc.y < kBorderSize.y
            || gc.y >= currShape.y - kBorderSize.y)
        {
            continue;
        }

        // Loop over central layers within an octave, layer 0 and numOctaveLayers + 1 are for top/bottom neighborhood
        for (gc.w = 1; gc.w <= numOctaveLayers; gc.w++)
        {
            cv = currDoG[gc]; // load central DoG value to test if it is an extremum point

            // An extremum point is one that is greater or lower than all its 26 neighbors around it (in a 1-halo):
            // 26 = 8 in the same layer + 9 in the layer above + 9 in the layer below
            // clang-format off
            if (cuda::abs(cv) > thr &&
                ((cv > 0 &&
                  cv >= currDoG[adj(gc, +0, +0, -1)] && cv >= currDoG[adj(gc, +0, +0, +1)] &&
                  cv >= currDoG[adj(gc, +0, -1, -1)] && cv >= currDoG[adj(gc, +0, -1, +0)] &&
                  cv >= currDoG[adj(gc, +0, -1, +1)] && cv >= currDoG[adj(gc, +0, +1, -1)] &&
                  cv >= currDoG[adj(gc, +0, +1, +0)] && cv >= currDoG[adj(gc, +0, +1, +1)] &&
                  cv >= currDoG[adj(gc, +1, +0, +0)] &&
                  cv >= currDoG[adj(gc, +1, +0, -1)] && cv >= currDoG[adj(gc, +1, +0, +1)] &&
                  cv >= currDoG[adj(gc, +1, -1, -1)] && cv >= currDoG[adj(gc, +1, -1, +0)] &&
                  cv >= currDoG[adj(gc, +1, -1, +1)] && cv >= currDoG[adj(gc, +1, +1, -1)] &&
                  cv >= currDoG[adj(gc, +1, +1, +0)] && cv >= currDoG[adj(gc, +1, +1, +1)] &&
                  cv >= currDoG[adj(gc, -1, +0, +0)] &&
                  cv >= currDoG[adj(gc, -1, +0, -1)] && cv >= currDoG[adj(gc, -1, +0, +1)] &&
                  cv >= currDoG[adj(gc, -1, -1, -1)] && cv >= currDoG[adj(gc, -1, -1, +0)] &&
                  cv >= currDoG[adj(gc, -1, -1, +1)] && cv >= currDoG[adj(gc, -1, +1, -1)] &&
                  cv >= currDoG[adj(gc, -1, +1, +0)] && cv >= currDoG[adj(gc, -1, +1, +1)]) ||
                 (cv < 0 &&
                  cv <= currDoG[adj(gc, +0, +0, -1)] && cv <= currDoG[adj(gc, +0, +0, +1)] &&
                  cv <= currDoG[adj(gc, +0, -1, -1)] && cv <= currDoG[adj(gc, +0, -1, +0)] &&
                  cv <= currDoG[adj(gc, +0, -1, +1)] && cv <= currDoG[adj(gc, +0, +1, -1)] &&
                  cv <= currDoG[adj(gc, +0, +1, +0)] && cv <= currDoG[adj(gc, +0, +1, +1)] &&
                  cv <= currDoG[adj(gc, +1, +0, +0)] &&
                  cv <= currDoG[adj(gc, +1, +0, -1)] && cv <= currDoG[adj(gc, +1, +0, +1)] &&
                  cv <= currDoG[adj(gc, +1, -1, -1)] && cv <= currDoG[adj(gc, +1, -1, +0)] &&
                  cv <= currDoG[adj(gc, +1, -1, +1)] && cv <= currDoG[adj(gc, +1, +1, -1)] &&
                  cv <= currDoG[adj(gc, +1, +1, +0)] && cv <= currDoG[adj(gc, +1, +1, +1)] &&
                  cv <= currDoG[adj(gc, -1, +0, +0)] &&
                  cv <= currDoG[adj(gc, -1, +0, -1)] && cv <= currDoG[adj(gc, -1, +0, +1)] &&
                  cv <= currDoG[adj(gc, -1, -1, -1)] && cv <= currDoG[adj(gc, -1, -1, +0)] &&
                  cv <= currDoG[adj(gc, -1, -1, +1)] && cv <= currDoG[adj(gc, -1, +1, -1)] &&
                  cv <= currDoG[adj(gc, -1, +1, +0)] && cv <= currDoG[adj(gc, -1, +1, +1)])))
            {
                // clang-format on

                // If the point at gc is an extrema point, fit (in a least-square sense) a quadratic polynomial to
                // the magnitude value around it to localize the scale-space extremum with a resolution higher than
                // the sampling density over space and scale.  The fit (done below) uses the derivatives of
                // sample-point neighbors and the principal curvatures (Hessian matrix) to solve a linear system
                // (using LU decomposition).  It converges when the interpolation step is less than half a pixel.

                int4 ic = gc; // to-be interpolated global coordinate

                bool converged = false;

                for (int i = 0; i < kMaxInterpolationSteps; i++)
                {
                    // clang-format off
                    dD[0] = (currDoG[adj(ic, +0, +0, +1)] - currDoG[adj(ic, +0, +0, -1)]) * kDScale1;
                    dD[1] = (currDoG[adj(ic, +0, +1, +0)] - currDoG[adj(ic, +0, -1, +0)]) * kDScale1;
                    dD[2] = (currDoG[adj(ic, +1, +0, +0)] - currDoG[adj(ic, -1, +0, +0)]) * kDScale1;

                    cv = currDoG[ic];

                    H[0][0] = (currDoG[adj(ic, +0, +0, +1)] + currDoG[adj(ic, +0, +0, -1)] - 2 * cv) * kDScale2;
                    H[1][1] = (currDoG[adj(ic, +0, +1, +0)] + currDoG[adj(ic, +0, -1, +0)] - 2 * cv) * kDScale2;
                    H[2][2] = (currDoG[adj(ic, +1, +0, +0)] + currDoG[adj(ic, -1, +0, +0)] - 2 * cv) * kDScale2;

                    H[0][1] = H[1][0] = (currDoG[adj(ic, +0, +1, +1)] - currDoG[adj(ic, +0, +1, -1)] -
                                         currDoG[adj(ic, +0, -1, +1)] + currDoG[adj(ic, +0, -1, -1)]) * kDScaleC;
                    H[0][2] = H[2][0] = (currDoG[adj(ic, +1, +0, +1)] - currDoG[adj(ic, +1, +0, -1)] -
                                         currDoG[adj(ic, -1, +0, +1)] + currDoG[adj(ic, -1, +0, -1)]) * kDScaleC;
                    H[1][2] = H[2][1] = (currDoG[adj(ic, +1, +1, +0)] - currDoG[adj(ic, +1, -1, +0)] -
                                         currDoG[adj(ic, -1, +1, +0)] + currDoG[adj(ic, -1, -1, +0)]) * kDScaleC;
                    // clang-format on

                    sol = dD; // since solve_inplace solves a 3x3 linear system H * sol = dD in-place

                    if (!cuda::math::solve_inplace(H, sol))
                    {
                        break; // if there is no solution to the linear system, convergence is not possible
                    }

                    sol = -sol;

                    if (cuda::abs(sol[2]) < 0.5f && cuda::abs(sol[1]) < 0.5f && cuda::abs(sol[0]) < 0.5f)
                    {
                        converged = true;
                        break; // less than half a pixel step means converged
                    }

                    if (cuda::abs(sol[2]) > cuda::TypeTraits<int>::max / 3.f
                        || cuda::abs(sol[1]) > cuda::TypeTraits<int>::max / 3.f
                        || cuda::abs(sol[0]) > cuda::TypeTraits<int>::max / 3.f)
                    {
                        break; // too big a step, convergence is not possible
                    }

                    ic.x += cuda::round<int>(sol[0]);
                    ic.y += cuda::round<int>(sol[1]);
                    ic.w += cuda::round<int>(sol[2]);

                    if (ic.w < 1 || ic.w > numOctaveLayers || ic.x < kBorderSize.x
                        || ic.x >= currShape.x - kBorderSize.x || ic.y < kBorderSize.y
                        || ic.y >= currShape.y - kBorderSize.y)
                    {
                        break; // went outside valid bounds, convergence is not possible
                    }
                }

                if (!converged)
                {
                    continue; // if not converged, skip this extremum point
                }

                // Feature score is proportional to the solution of the linear system
                float featScore = cuda::abs(cv * kImageScale + cuda::math::dot(dD, sol) * .5f);
                if (featScore * numOctaveLayers < contrastThreshold)
                {
                    continue; // skip features with score below contrast threshold
                }

                // Compute trace and determinant of the Hessian matrix to avoid strong responses of the Laplacian
                // operator (approximated by the DoG) to edges, this uses the edge threshold
                float trace       = H[0][0] + H[1][1];
                float determinant = H[0][0] * H[1][1] - H[0][1] * H[1][0];
                if (determinant <= 0
                    || trace * trace * edgeThreshold >= (edgeThreshold + 1) * (edgeThreshold + 1) * determinant)
                {
                    continue;
                }

                // Feature radius is proportional to the normalized layer solution and initial sigma
                float featRadius = initSigma * ::pow(2, (ic.w + sol[2]) / numOctaveLayers);

                // The radius used to compute the histogram for feature angle calculation
                int radius = cuda::round<int>(featRadius * kOrientationRadius);

                float weightScale = -1.f / (2.f * (featRadius * kOrientationSigma) * (featRadius * kOrientationSigma));

                for (int i = 0; i < kHistogramBins; ++i)
                {
                    hist[i] = 0.f;
                }

                // Loop the radius around extremum point to compute angle and vector magnitude
                for (int i = -radius; i <= radius; i++)
                {
                    if (ic.y + i <= 0 || ic.y + i >= currShape.y - 1)
                    {
                        continue;
                    }

                    for (int j = -radius; j <= radius; j++)
                    {
                        if (ic.x + j <= 0 || ic.x + j >= currShape.x - 1)
                        {
                            continue;
                        }

                        // Compute Gaussian delta x and y using 1-halo neighborhood
                        float dx = currGauss[adj(ic, 0, i + 0, j + 1)] - currGauss[adj(ic, 0, i + 0, j - 1)];
                        float dy = currGauss[adj(ic, 0, i - 1, j + 0)] - currGauss[adj(ic, 0, i + 1, j + 0)];

                        float angle     = ::atan2f(dy, dx) * kRadiansToDegrees;   // angle in degrees
                        float weight    = ::exp2f((i * i + j * j) * weightScale); // weighted scale of vector magnitude
                        float magnitude = ::sqrtf(dx * dx + dy * dy);             // neighbor vector magnitude

                        if (angle < 0.f)
                            angle += 360.f;
                        if (angle >= 360.f)
                            angle -= 360.f;

                        int bin = cuda::round<int>(angle * kAngleToBin); // the angle indexes the histogram bin
                        bin = (bin >= kHistogramBins ? bin - kHistogramBins : (bin < 0 ? bin + kHistogramBins : bin));

                        hist[bin] += weight * magnitude; // store the vector magnitude of the angle bin
                    }
                }

                int   bin       = 0;
                float prologue1 = hist[kHistogramBins - 1];
                float prologue2 = hist[kHistogramBins - 2];
                float center    = hist[bin];
                float epilogue1 = hist[bin + 1];
                float epilogue2 = hist[bin + 2];
                float first     = hist[0];
                float second    = hist[1];
                do // smooth histogram in-place using loaded prologues and epilogues (plus center, first and second)
                {
                    hist[bin++]
                        = (prologue2 + epilogue2) * 1.f / 16 + (prologue1 + epilogue1) * 4.f / 16 + center * 6.f / 16;

                    prologue2 = prologue1;
                    prologue1 = center;
                    center    = epilogue1;
                    epilogue1 = epilogue2;
                    epilogue2
                        = bin == kHistogramBins - 2 ? first : (bin >= kHistogramBins - 1 ? second : hist[bin + 2]);
                }
                while (bin < kHistogramBins);

                // Compute maximum (peak) value in histogram
                float histPeak = hist[0];
                for (int i = 1; i < kHistogramBins; ++i)
                {
                    histPeak = cuda::max(histPeak, hist[i]);
                }
                histPeak *= kHistogramPeakRatio; // apply a ratio to the histogram peak

                int validAngles = 0;

                bin       = 0;
                prologue1 = hist[kHistogramBins - 1];
                center    = hist[bin];
                epilogue1 = hist[bin + 1];
                first     = hist[0];
                do // find valid angles (close to histogram peak) replacing histogram by the valid angles
                {
                    if (center > prologue1 && center > epilogue1 && center >= histPeak)
                    {
                        float fbin = bin + .5f * (prologue1 - epilogue1) / (prologue1 - 2 * center + epilogue1);

                        fbin = (fbin < 0 ? kHistogramBins + fbin
                                         : (fbin >= kHistogramBins ? fbin - kHistogramBins : fbin));

                        hist[validAngles++] = 360.f - fbin * kBinToAngle;
                    }

                    bin++;
                    prologue1 = center;
                    center    = epilogue1;
                    epilogue1 = bin + 1 >= kHistogramBins ? first : hist[bin + 1];
                }
                while (bin < kHistogramBins);

                if (validAngles == 0)
                {
                    continue; // if there is no valid angle, skip this extremum point
                }

                int firstFeatIdx = atomicAdd(numFeatures.ptr(ic.z), validAngles);
                if (firstFeatIdx >= maxCapacity)
                {
                    continue; // skip if the feature index is beyond maximum capacity to store features
                }

                for (int vi = 0; vi < validAngles; ++vi)
                {
                    if (firstFeatIdx + vi >= maxCapacity)
                    {
                        break; // skip if the feature index is beyond maximum capacity to store features
                    }

                    float angle = hist[vi]; // load valid angle stored in the histogram
                    if (cuda::abs(angle - 360.f) < 1e-5)
                        angle = 0.f;

                    // Write the feature coordinate (octave scaled) and metadata in corresponding output tensors
                    *featCoords.ptr(ic.z, firstFeatIdx + vi)
                        = float4{(ic.x + sol[0]) * scaleOctave, (ic.y + sol[1]) * scaleOctave, (float)featOctave,
                                 (ic.w + sol[2])};
                    *featMetadata.ptr(ic.z, firstFeatIdx + vi)
                        = float3{angle, featScore, featRadius * 2.f * scaleOctave};
                }
            }
        }
    }
}

} // anonymous namespace

namespace cvcuda::priv {

// SIFT private functions ------------------------------------------------------

// Find SIFT features as extrema keypoints and compute their metadata, DT is the input data type
template<typename DT>
void SIFT::FindExtrema(const nvcv::TensorDataStridedCuda &featCoordsData,
                       const nvcv::TensorDataStridedCuda &featMetadataData,
                       const nvcv::TensorDataStridedCuda &featDescriptorsData, int maxCapacity,
                       const nvcv::TensorDataStridedCuda &numFeaturesData, int3 currShape, int firstOctave,
                       int numOctaves, int numOctaveLayers, float contrastThreshold, float edgeThreshold,
                       float initSigma, cudaStream_t stream) const
{
    constexpr int BW = 32; // data block tile width handled by find extrema kernel
    constexpr int BH = 4;  // data block tile height handled by find extrema kernel

    // Computational block of threads and grid of blocks used by find extrema (1) and compute descriptors (2) kernels
    dim3 compThreads1{128, 1, 1};
    dim3 compThreads2{256, 1, 1};
    dim3 compBlocks1;
    dim3 compBlocks2(maxCapacity, 1, currShape.z);

    cuda::Tensor2DWrap<float4> featCoordsWrap(featCoordsData.basePtr(), (int)featCoordsData.stride(0));
    cuda::Tensor2DWrap<float3> featMetadataWrap(featMetadataData.basePtr(), (int)featMetadataData.stride(0));
    cuda::Tensor1DWrap<int>    numFeaturesWrap(numFeaturesData.basePtr());
    TensorWrapForDescriptor    featDescriptorsWrap(featDescriptorsData.basePtr(), (int)featDescriptorsData.stride(0));

    // Initially set to zero the number of features for each image within source tensor, currShape.z = # of images
    NVCV_CHECK_THROW(cudaMemsetAsync(numFeaturesData.basePtr(), 0, sizeof(int) * currShape.z, stream));

    int intThreshold = std::floor(.5f * contrastThreshold / numOctaveLayers * 255);

    for (int octave = 0; octave < numOctaves; octave++)
    {
        // The Gaussian pyramid stores at octave=0 the base (potentially expanded) input image, that is why the
        // current Gaussian data is shifted by 1 while DoG (Difference of Gaussians) data is not

        auto currGaussData = m_runPyramidGaussian[octave + 1].exportData<nvcv::TensorDataStridedCuda>();
        CVCUDA_SIFT_INTERNAL_DATA_CHECK(currGaussData);

        auto currDoGData = m_runPyramidDoG[octave].exportData<nvcv::TensorDataStridedCuda>();
        CVCUDA_SIFT_INTERNAL_DATA_CHECK(currDoGData);

        TensorWrapLNHW<const float> currGaussWrap(*currGaussData);
        TensorWrapLNHW<const float> currDoGWrap(*currDoGData);

        compBlocks1 = dim3(util::DivUp(currShape.x, BW), util::DivUp(currShape.y, BH), currShape.z);

        int   featOctave    = firstOctave + octave; // feature octave may start at -1 base expanded input image
        float scaleOctave   = ::pow(2, featOctave); // scale feature coordinate or size back to base image
        float unscaleOctave = 1.f / scaleOctave;    // un-scale feature coordinate or size back to current octave

        // First run the DoFindExtrema kernel to find extrema points (the features) and compute their metadata

        DoFindExtrema<BW, BH, DT><<<compBlocks1, compThreads1, 0, stream>>>(
            featCoordsWrap, featMetadataWrap, maxCapacity, numFeaturesWrap, currGaussWrap, currDoGWrap, currShape,
            featOctave, scaleOctave, numOctaveLayers, intThreshold, contrastThreshold, edgeThreshold, initSigma);

        // Second run the DoComputeDescriptors kernel to compute the descriptor of each feature found

        DoComputeDescriptors<<<compBlocks2, compThreads2, 0, stream>>>(featDescriptorsWrap, featCoordsWrap,
                                                                       featMetadataWrap, numFeaturesWrap, currGaussWrap,
                                                                       currShape, featOctave, unscaleOctave);

        currShape.x /= 2;
        currShape.y /= 2;
    }
}

// Compute run-time pyramids, DT is the input data type
template<typename DT>
void SIFT::ComputePyramids(const nvcv::TensorDataStridedCuda &inData, int3 currShape, bool expandInput, int numOctaves,
                           int numOctaveLayers, float initSigma, cudaStream_t stream) const
{
    constexpr int BW = 32; // data block tile width in shared memory (smem) used by compute pyramids kernel
    constexpr int BH = 32; // data block tile height in shared memory (smem) used by compute pyramids kernel

    // Block of threads and grid of blocks used by copy and compute pyramid kernels
    dim3 copyThreads{256, 1, 1};
    dim3 copyBlocks(util::DivUp(currShape.x, copyThreads.x), currShape.y, currShape.z);

    dim3 compThreads{256, 1, 1};
    dim3 compBlocks;

    NVCV_ASSERT((numOctaves + 1) <= (int)m_runPyramidGaussian.size());
    NVCV_ASSERT(numOctaves <= (int)m_runPyramidDoG.size());

    // Copy inData into base Gaussian pyramid (octave=0), casting from U8 to F32 (and potentially upscaling it)

    auto dstBaseData = m_runPyramidGaussian[0].exportData<nvcv::TensorDataStridedCuda>();
    CVCUDA_SIFT_INTERNAL_DATA_CHECK(dstBaseData);

    TensorWrapLNHW<float> dstBaseWrap(*dstBaseData);

    if (expandInput)
    {
        auto srcBaseWrap = cuda::CreateInterpolationWrapNHW<const DT, kBorderInterp, kInterpUp>(inData);

        UpCopy<<<copyBlocks, copyThreads, 0, stream>>>(dstBaseWrap, srcBaseWrap, currShape); // upscale copy
    }
    else
    {
        auto srcBaseWrap = cuda::CreateTensorWrapNHW<const DT>(inData);

        Copy<<<copyBlocks, copyThreads, 0, stream>>>(dstBaseWrap, srcBaseWrap, currShape); // direct copy
    }

    // Set sigma scale, current and base sigma for Gaussian filter kernel computation

    int   sigmaScale = (expandInput ? 4 : 1); // due to source scaled up by 2x in each dimension
    float currSigma  = cuda::sqrt(cuda::max(initSigma * initSigma - kPrevSigma * kPrevSigma * sigmaScale, kMinSigma));
    float baseSigma  = cuda::pow(2.f, 1.f / numOctaveLayers);

    cuda::math::Vector<float, kMaxKernelSize> gaussSepKernel; // separable Gaussian filter kernel

    for (int octave = 0; octave < numOctaves; octave++)
    {
        if (octave > 0)
        {
            currSigma = initSigma; // for every octave after first, the current sigma is the initial sigma
        }

        // Set previous and current octave Gaussian and DoG (Difference of Gaussians) pyramid data

        auto prevGaussData = m_runPyramidGaussian[octave].exportData<nvcv::TensorDataStridedCuda>();
        CVCUDA_SIFT_INTERNAL_DATA_CHECK(prevGaussData);

        auto currGaussData = m_runPyramidGaussian[octave + 1].exportData<nvcv::TensorDataStridedCuda>();
        CVCUDA_SIFT_INTERNAL_DATA_CHECK(currGaussData);

        auto currDoGData = m_runPyramidDoG[octave].exportData<nvcv::TensorDataStridedCuda>();
        CVCUDA_SIFT_INTERNAL_DATA_CHECK(currDoGData);

        // Using BorderWrap (BW) when border handling is needed and TensorWrap (TW) when not

        TensorWrapLNHW<float> prevGaussTW(*prevGaussData);
        TensorWrapLNHW<float> currGaussTW(*currGaussData);
        BorderWrapLNHW<float> prevGaussBW(*prevGaussData);
        BorderWrapLNHW<float> currGaussBW(*currGaussData);
        TensorWrapLNHW<float> currDoGTW(*currDoGData);

        copyBlocks = dim3(util::DivUp(currShape.x, copyThreads.x), currShape.y, currShape.z);
        compBlocks = dim3(util::DivUp(currShape.x, BW), util::DivUp(currShape.y, BH), currShape.z);

        // For each layer up to numOctaveLayers +3 to have +1 for DoG and +2 for top/bottom neihbors
        for (int layer = 0; layer < numOctaveLayers + 3; layer++)
        {
            if (layer == 0 && octave > 0)
            {
                // First layer of every octave after first skips Gaussian blur and does only a downscale copy
                DownCopy<<<copyBlocks, copyThreads, 0, stream>>>(currGaussTW, prevGaussTW, currShape, numOctaveLayers);
            }
            else
            {
                if (layer > 0)
                {
                    // Update current sigma for every layer after the first
                    float prevSigma  = cuda::pow(baseSigma, (float)(layer - 1)) * initSigma;
                    float totalSigma = baseSigma * prevSigma;
                    currSigma        = std::sqrt(totalSigma * totalSigma - prevSigma * prevSigma);
                }

                // Compute the separable Gaussian filter kernel for the Gaussian pyramid
                int   ksize = ComputeGaussianKernelSize(currSigma);
                int   half  = ksize / 2;
                float ss2   = currSigma * currSigma * 2;
                float sp2   = currSigma * cuda::sqrt(M_PI * 2);
                float sum   = 0.f;

                for (int kx = -half; kx <= half; ++kx)
                {
                    // Compute separable Gaussian function value using its half-equation
                    float w = cuda::exp(-((kx * kx) / ss2)) / sp2;

                    sum += w;

                    gaussSepKernel[half + kx] = w;
                }
                for (int kx = -half; kx <= half; ++kx)
                {
                    gaussSepKernel[half + kx] /= sum;
                }

                // Run the DoComputePyramids kernel to compute the Gaussian and DoG pyramids

                if (octave == 0 && layer == 0)
                {
                    // Only for the first octave and first layer the base level previous Gaussian data is used
                    DoComputePyramids<BW, BH><<<compBlocks, compThreads, 0, stream>>>(
                        prevGaussBW, currGaussTW, currDoGTW, currShape, layer, ksize, gaussSepKernel);
                }
                else
                {
                    // For every other octave and layer the current Gaussian data (border-aware) is used
                    DoComputePyramids<BW, BH><<<compBlocks, compThreads, 0, stream>>>(
                        currGaussBW, currGaussTW, currDoGTW, currShape, layer, ksize, gaussSepKernel);
                }
            }
        }

        currShape.x /= 2;
        currShape.y /= 2;
    }
}

// Reshape payload-time maxPyramids to submit (or execution) time runPyramids
void SIFT::ReshapePyramids(const int3 &inShape, int numOctaves, int numOctaveLayers) const
{
    // Create shapes converting from int3=WHN and L=octaveLayers to LNHWC or NHWC tensor shapes
    nvcv::TensorShape::ShapeType shapeGaussian{numOctaveLayers + 3, inShape.z, inShape.y, inShape.x, 1};
    nvcv::TensorShape::ShapeType shapeDoG{numOctaveLayers + 2, inShape.z, inShape.y, inShape.x, 1};
    nvcv::TensorShape::ShapeType shapeBase({1, inShape.z, inShape.y, inShape.x, 1});

    NVCV_ASSERT((numOctaves + 1) <= (int)m_maxPyramidGaussian.size());
    NVCV_ASSERT((numOctaves + 1) <= (int)m_runPyramidGaussian.size());
    NVCV_ASSERT(numOctaves <= (int)m_maxPyramidDoG.size());
    NVCV_ASSERT(numOctaves <= (int)m_runPyramidDoG.size());

    // The idea is to get a view of a tensor from maxPyramid into runPyramid.  In the 1st reshape pyramid, it will
    // drop the null tensor, which does not deallocate memory.  In the 2nd reshape pyramid, it will drop the tensor
    // view, which also does not deallocate the parent tensor since maxPyramid holds it.

    {
        auto gpData = m_maxPyramidGaussian[0].exportData<nvcv::TensorDataStridedCuda>();
        CVCUDA_SIFT_INTERNAL_DATA_CHECK(gpData);

        m_runPyramidGaussian[0] = GetViewFrom(*gpData, shapeBase);
    }

    for (int octave = 0; octave < numOctaves; octave++)
    {
        auto gpData = m_maxPyramidGaussian[octave + 1].exportData<nvcv::TensorDataStridedCuda>();
        CVCUDA_SIFT_INTERNAL_DATA_CHECK(gpData);

        m_runPyramidGaussian[octave + 1] = GetViewFrom(*gpData, shapeGaussian);

        auto dogData = m_maxPyramidDoG[octave].exportData<nvcv::TensorDataStridedCuda>();
        CVCUDA_SIFT_INTERNAL_DATA_CHECK(dogData);

        m_runPyramidDoG[octave] = GetViewFrom(*dogData, shapeDoG);

        shapeGaussian[2] /= 2;
        shapeGaussian[3] /= 2;
        shapeDoG[2] /= 2;
        shapeDoG[3] /= 2;
    }
}

// Allocate (create) the initial SIFT pyramids for maximum shape (run-time is null) (done on operator constructor)
void SIFT::CreatePyramids()
{
    // Tensor LNHWC layout lives within an octave and has L octave levels each of which with NHWC tensors
    constexpr nvcv::TensorLayout TENSOR_LNHWC(NVCVTensorLayout{"LNHWC", 5});

    // Create shapes converting from int3=WHN and L=octaveLayers to LNHWC tensor shapes

    // Gaussian pyramid with maxOctaveLayers +3 to have +1 for DoG and +2 for top/bottom neighbors
    // DoG pyramid with maxOctaveLayers +2 for top/bottom neighbors
    // Base level pyramid with 1 layer to store original image (potentially expanded)

    nvcv::TensorShape::ShapeType shapeGaussian{m_maxOctaveLayers + 3, m_maxShape.z, m_maxShape.y, m_maxShape.x, 1};
    nvcv::TensorShape::ShapeType shapeDoG{m_maxOctaveLayers + 2, m_maxShape.z, m_maxShape.y, m_maxShape.x, 1};
    nvcv::TensorShape::ShapeType shapeBase({1, m_maxShape.z, m_maxShape.y, m_maxShape.x, 1});

    // Gaussian pyramid has +1 in octaves to have original base image (potentially expanded) at octave=0

    m_maxPyramidGaussian.reserve(m_maxOctaves + 1);
    m_runPyramidGaussian.reserve(m_maxOctaves + 1);
    m_maxPyramidDoG.reserve(m_maxOctaves);
    m_runPyramidDoG.reserve(m_maxOctaves);

    // runPyramids store null tensors as they will be replaced by a view of maxPyramids tensors later on

    m_maxPyramidGaussian.emplace_back(nvcv::TensorShape{shapeBase, TENSOR_LNHWC}, nvcv::TYPE_F32);
    m_runPyramidGaussian.emplace_back(nvcv::Tensor{});

    for (int octave = 0; octave < m_maxOctaves; octave++)
    {
        m_maxPyramidGaussian.emplace_back(nvcv::TensorShape{shapeGaussian, TENSOR_LNHWC}, nvcv::TYPE_F32);
        m_runPyramidGaussian.emplace_back(nvcv::Tensor{});

        m_maxPyramidDoG.emplace_back(nvcv::TensorShape{shapeDoG, TENSOR_LNHWC}, nvcv::TYPE_F32);
        m_runPyramidDoG.emplace_back(nvcv::Tensor{});

        shapeGaussian[2] /= 2;
        shapeGaussian[3] /= 2;
        shapeDoG[2] /= 2;
        shapeDoG[3] /= 2;
    }
}

// Constructor -----------------------------------------------------------------

SIFT::SIFT(int3 maxShape, int maxOctaveLayers)
    : m_maxShape{maxShape}
    , m_maxOctaves{ComputeNumberOfOctaves(maxShape.x, maxShape.y)}
    , m_maxOctaveLayers{maxOctaveLayers}
{
    if (maxShape.x < 2 || maxShape.y < 2 || maxShape.z < 1 || maxShape.z > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Operator constructor arg. maxShape (W=%d H=%d N=%d) must have W and H >= 2 and N in "
                              "[1, 65535]",
                              maxShape.x, maxShape.y, maxShape.z);
    }
    if (maxOctaveLayers < 1 || maxOctaveLayers > 16)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Operator constructor arg. maxOctaveLayers=%d must be in [1, 16]", maxOctaveLayers);
    }

    CreatePyramids();
}

// Submit ----------------------------------------------------------------------

void SIFT::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &featCoords,
                      const nvcv::Tensor &featMetadata, const nvcv::Tensor &featDescriptors,
                      const nvcv::Tensor &numFeatures, int numOctaveLayers, float contrastThreshold,
                      float edgeThreshold, float initSigma, NVCVSIFTFlagType flags) const
{
    // Check each tensor layout, strides, shape and data type if it is conforming to what is expected

    if (!(in.layout() == nvcv::TENSOR_HWC || in.layout() == nvcv::TENSOR_NHWC))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor layout must be HWC or NHWC");
    }

    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (!inData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be a valid CUDA strided tensor");
    }
    if (inData->dtype() != nvcv::TYPE_U8) // The only supported data type is U8
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor dtype must be U8");
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be a valid image-based tensor");
    }
    if (inAccess->numChannels() > 1 || inAccess->numPlanes() > 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input tensor must have 1 channel and 1 plane");
    }

    int3 inShape{(int)inAccess->numCols(), (int)inAccess->numRows(), (int)inAccess->numSamples()};

    bool expandInput = (flags == NVCV_SIFT_USE_EXPANDED_INPUT);

    if (expandInput)
    {
        inShape.x *= 2;
        inShape.y *= 2;
    }

    int numOctaves  = ComputeNumberOfOctaves(inShape.x, inShape.y);
    int firstOctave = expandInput ? -1 : 0;

    if (inShape.x < 2 || inShape.y < 2 || inShape.z < 1 || inShape.x > m_maxShape.x || inShape.y > m_maxShape.y
        || inShape.z > m_maxShape.z)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensor shape (W=%d H=%d N=%d) must have W and H >= 2 and N >= 1 and be "
                              "smaller than or equal to maxShape (W=%d H=%d N=%d) defined in operator constructor",
                              inShape.x, inShape.y, inShape.z, m_maxShape.x, m_maxShape.y, m_maxShape.z);
    }

    if (initSigma <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Operator call arg. initSigma=%f must be positive",
                              initSigma);
    }
    if (contrastThreshold <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Argument contrastThreshold=%f must be positive",
                              contrastThreshold);
    }
    if (edgeThreshold <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Argument edgeThreshold=%f must be positive",
                              edgeThreshold);
    }

    if (numOctaveLayers < 1 || numOctaveLayers > m_maxOctaveLayers)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Operator call arg. numOctaveLayers=%d must be in [1, %d]", numOctaveLayers,
                              m_maxOctaveLayers);
    }

    auto featCoordsData = featCoords.exportData<nvcv::TensorDataStridedCuda>();
    if (!featCoordsData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output featCoords must be a valid CUDA strided tensor");
    }
    if (!((featCoordsData->rank() == 3
           && ((featCoordsData->dtype() == nvcv::TYPE_F32 && featCoordsData->shape(2) == 4)
               || (featCoordsData->dtype() == nvcv::TYPE_4F32 && featCoordsData->shape(2) == 1)))
          || (featCoordsData->rank() == 2 && featCoordsData->dtype() == nvcv::TYPE_4F32)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output featCoords must have rank 2 or 3 and 4xF32 or 4F32 data type");
    }
    if (featCoordsData->shape(0) != inShape.z)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output featCoords 1st shape must be the same as input tensor number of samples, "
                              "%d != %d",
                              (int)featCoordsData->shape(0), inShape.z);
    }
    if (!(featCoordsData->stride(1) == sizeof(float4)
              && (featCoordsData->rank() == 3
                  && ((featCoordsData->dtype() == nvcv::TYPE_F32 && featCoordsData->stride(2) == sizeof(float))
                      || (featCoordsData->dtype() == nvcv::TYPE_4F32 && featCoordsData->stride(2) == sizeof(float4))))
          || (featCoordsData->rank() == 2)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output featCoords last strides must be packed for 4xF32 data type, stride1=%d",
                              (int)featCoordsData->stride(1));
    }

    int maxCapacity = featCoordsData->shape(1);

    auto featMetadataData = featMetadata.exportData<nvcv::TensorDataStridedCuda>();
    if (!featMetadataData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output featMetadata must be a valid CUDA strided tensor");
    }
    if (!((featMetadataData->rank() == 3
           && ((featMetadataData->dtype() == nvcv::TYPE_F32 && featMetadataData->shape(2) == 3)
               || (featMetadataData->dtype() == nvcv::TYPE_3F32 && featMetadataData->shape(2) == 1)))
          || (featMetadataData->rank() == 2 && featMetadataData->dtype() == nvcv::TYPE_3F32)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output featMetadata must have rank 2 or 3 and 3xF32 or 3F32 data type");
    }
    if (featMetadataData->shape(0) != inShape.z)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output featMetadata 1st shape must be the same as input tensor number of samples, "
                              "%d != %d",
                              (int)featMetadataData->shape(0), inShape.z);
    }
    if (featMetadataData->shape(1) != maxCapacity)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output featMetadata 2nd shape must be the same across output tensors, %d != %d",
                              (int)featMetadataData->shape(1), maxCapacity);
    }
    if (!(featMetadataData->stride(1) == sizeof(float3)
              && (featMetadataData->rank() == 3
                  && ((featMetadataData->dtype() == nvcv::TYPE_F32 && featMetadataData->stride(2) == sizeof(float))
                      || (featMetadataData->dtype() == nvcv::TYPE_3F32
                          && featMetadataData->stride(2) == sizeof(float3))))
          || (featMetadataData->rank() == 2)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output featMetadata last strides must be packed for 3xF32 data type, stride1=%d",
                              (int)featMetadataData->stride(1));
    }

    auto featDescriptorsData = featDescriptors.exportData<nvcv::TensorDataStridedCuda>();
    if (!featDescriptorsData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output featDescriptors must be a valid CUDA strided tensor");
    }
    if (!(featDescriptorsData->rank() == 3 && featDescriptorsData->dtype() == nvcv::TYPE_U8
          && featDescriptorsData->shape(2) == kDescSize))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output featDescriptors must have rank 3, U8 data type and last shape equals %d",
                              kDescSize);
    }
    if (featDescriptorsData->shape(0) != inShape.z)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output featDescriptors 1st shape must be the same as input tensor number of samples, "
                              "%d != %d",
                              (int)featDescriptorsData->shape(0), inShape.z);
    }
    if (featDescriptorsData->shape(1) != maxCapacity)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output featDescriptors 2nd shape must be the same across output tensors, %d != %d",
                              (int)featDescriptorsData->shape(1), maxCapacity);
    }
    if (!(featDescriptorsData->stride(2) == sizeof(uint8_t)
          && featDescriptorsData->stride(1) == kDescSize * sizeof(uint8_t)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output featDescriptors last strides must be packed with 128 U8 data type, "
                              "stride1=%d stride2=%d",
                              (int)featDescriptorsData->stride(1), (int)featDescriptorsData->stride(2));
    }

    auto numFeaturesData = numFeatures.exportData<nvcv::TensorDataStridedCuda>();
    if (!numFeaturesData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output numFeatures must be a valid CUDA strided tensor");
    }
    if (!(numFeaturesData->dtype() == nvcv::TYPE_S32
          && ((numFeaturesData->rank() == 2 && numFeaturesData->shape(1) == 1) || (numFeaturesData->rank() == 1))))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output numFeatures must have rank 1 or 2 and 1xS32 or S32 data type");
    }
    if (numFeaturesData->shape(0) != inShape.z)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output numFeatures 1st shape must be the same as input tensor number of samples, "
                              "%d != %d",
                              (int)numFeaturesData->shape(0), inShape.z);
    }
    if (!(numFeaturesData->stride(0) == sizeof(int)
          && ((numFeaturesData->rank() == 2 && numFeaturesData->stride(1) == sizeof(int))
              || (numFeaturesData->rank() == 1))))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output numFeatures last strides must be packed for 1xS32 data type, stride0=%d",
                              (int)numFeaturesData->stride(0));
    }

    // The only supported data type is U8, hence uint8_t in DT templates below.
    // After all checkings are done, the input and output tensors can be used in executing the operator.
    // First reshape the pyramids, then compute them and, finally, find extrema points, i.e. features.

    ReshapePyramids(inShape, numOctaves, numOctaveLayers);

    ComputePyramids<uint8_t>(*inData, inShape, expandInput, numOctaves, numOctaveLayers, initSigma, stream);

    FindExtrema<uint8_t>(*featCoordsData, *featMetadataData, *featDescriptorsData, maxCapacity, *numFeaturesData,
                         inShape, firstOctave, numOctaves, numOctaveLayers, contrastThreshold, edgeThreshold, initSigma,
                         stream);
}

} // namespace cvcuda::priv
