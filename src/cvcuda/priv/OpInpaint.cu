/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
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

#include "LegacyFormatClassify.hpp"
#include "Nvtx.hpp"
#include "OpInpaint.hpp"
#include "SafeSize.hpp"
#include "TensorAccessValidation.hpp"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

using uchar = unsigned char;

// Telea fast-marching classification of a pixel.
constexpr uchar KNOWN  = 0; // known, outside the narrow band
constexpr uchar BAND   = 1; // narrow band (known)
constexpr uchar INSIDE = 2; // unknown, to be inpainted

constexpr int kBlock           = 32; // mask-copy block is kBlock x kBlock/4
constexpr int kBlockS          = 16; // fast-marching block edge
constexpr int kReduceGridSize  = 64; // reduction blocks per sample when counting the remaining band
constexpr int kReduceBlockSize = 256;
constexpr int kTeleaIterations = 10; // two kernels per sweep, then check whether the front is empty

// ---------------------------------------------------------------------------------------------
// Scratch views
//
// f, t, band and the padded inpaint mask are plain device allocations carved out of the
// operator's workspace.  They are addressed with the same int32 strides the legacy Ptr2dNHWC
// carried, so the generated address arithmetic is unchanged.
// ---------------------------------------------------------------------------------------------

using MaskView = cuda::Tensor3DWrap<uchar, int32_t>;
using DistView = cuda::Tensor3DWrap<float, int32_t>;

template<typename T>
inline cuda::Tensor3DWrap<T, int32_t> ScratchView(void *data, int rows, int cols)
{
    const int32_t rowStride   = static_cast<int32_t>(cols * static_cast<int>(sizeof(T)));
    const int32_t planeStride = static_cast<int32_t>(rows * cols * static_cast<int>(sizeof(T)));

    return cuda::Tensor3DWrap<T, int32_t>(reinterpret_cast<T *>(data), planeStride, rowStride);
}

// ---------------------------------------------------------------------------------------------
// Extents
//
// Every kernel is launched over the batch-wide padded extent.  The Tensor path clips each sample
// to the batch's single image size; the VarShape path clips it to that sample's own image.  Both
// report the *unpadded* image extent; kernels add the one-pixel border where the algorithm does.
// ---------------------------------------------------------------------------------------------

struct UniformExtent
{
    int rows;
    int cols;

    inline __device__ int Rows(int) const
    {
        return rows;
    }

    inline __device__ int Cols(int) const
    {
        return cols;
    }
};

template<typename SizeWrapper>
struct PerImageExtent
{
    SizeWrapper sizes;

    inline __device__ int Rows(int b) const
    {
        return sizes.height(b);
    }

    inline __device__ int Cols(int b) const
    {
        return sizes.width(b);
    }
};

// ---------------------------------------------------------------------------------------------
// Source scan window
//
// The Telea sweep accumulates over the source pixels that lie both inside the disc of radius
// `range` around (i, j) and inside the padded plane's interior.  That set, and the order it is
// visited in, is the same in every mode; the modes differ only in how much of it the loop bounds
// decide instead of a test in the innermost loop.
//
//   Guarded  scans the box [i +/- range] x [j +/- range] and tests the interior and the disc per
//            pixel.  The interior test also guards the `f` load, so it has to come first.
//   Clipped  folds the interior into the loop bounds; the disc is still tested per pixel.
//   Disc     folds both into the loop bounds.  Per row k, `|l - j| <= floor(sqrt(range^2 -
//            (k - i)^2))` is exactly the disc: l is an integer, and the radicand is an integer in
//            [0, range^2] because k is already clipped to [i - range, i + range] and `range` is
//            clamped to [1, 100] on the host, so it is exact in float and `sqrtf` (correctly
//            rounded) cannot round across an integer -- below 100 the closest a non-square root
//            comes to an integer from below is 100 - sqrt(9999) ~= 5e-3, against an ulp of ~8e-6.
// ---------------------------------------------------------------------------------------------

enum class ScanMode
{
    Guarded,
    Clipped,
    Disc,
};

// ---------------------------------------------------------------------------------------------
// Output accessors
//
// The four output shapes the operator serves, behind one At(sample, row, col, channel) accessor
// plus the padded extent the fast-marching sweep bounds itself with.  Each also names the scan
// mode its kernel is built with, so the two cannot be paired up wrongly.
// ---------------------------------------------------------------------------------------------

template<typename T>
struct TensorInterleavedOut
{
    static constexpr ScanMode kScanMode = ScanMode::Disc;

    cuda::Tensor4DWrap<T, int32_t> data;
    int                            paddedRows;
    int                            paddedCols;

    inline __device__ T *At(int b, int y, int x, int c) const
    {
        return data.ptr(b, y, x, c);
    }

    inline __device__ int PaddedRows(int) const
    {
        return paddedRows;
    }

    inline __device__ int PaddedCols(int) const
    {
        return paddedCols;
    }
};

template<typename T>
struct TensorPlanarOut
{
    static constexpr ScanMode kScanMode = ScanMode::Clipped;

    cuda::Tensor4DWrap<T, int32_t> data;
    int                            paddedRows;
    int                            paddedCols;

    inline __device__ T *At(int b, int y, int x, int c) const
    {
        return data.ptr(b, c, y, x);
    }

    inline __device__ int PaddedRows(int) const
    {
        return paddedRows;
    }

    inline __device__ int PaddedCols(int) const
    {
        return paddedCols;
    }
};

template<typename T>
struct VarShapeInterleavedOut
{
    static constexpr ScanMode kScanMode = ScanMode::Guarded;

    cuda::ImageBatchVarShapeWrapNHWC<T> data;

    inline __device__ T *At(int b, int y, int x, int c) const
    {
        return data.ptr(b, y, x, c);
    }

    inline __device__ int PaddedRows(int b) const
    {
        return data.height(b) + 2;
    }

    inline __device__ int PaddedCols(int b) const
    {
        return data.width(b) + 2;
    }
};

template<typename T>
struct VarShapePlanarOut
{
    static constexpr ScanMode kScanMode = ScanMode::Guarded;

    cuda::ImageBatchVarShapeWrap<T> data;

    inline __device__ T *At(int b, int y, int x, int c) const
    {
        return data.ptr(b, c, y, x);
    }

    inline __device__ int PaddedRows(int b) const
    {
        return data.height(b) + 2;
    }

    inline __device__ int PaddedCols(int b) const
    {
        return data.width(b) + 2;
    }
};

// ---------------------------------------------------------------------------------------------
// Telea weighting arithmetic
//
// The Tensor and VarShape kernels shared their structure but not their arithmetic: the Tensor
// kernel evaluates the weight in single precision, the VarShape kernel promotes the distance and
// level terms to double before narrowing back to float.  The two therefore do not produce the
// same bits, so each keeps its own expressions verbatim behind this policy.
// ---------------------------------------------------------------------------------------------

struct TensorTeleaPolicy
{
    // See the note on TeleaInpaintFMM. The tensor form reads its extent from scalars, needs fewer
    // live values than the var-shape form, and measured best at the allocator's own four-block
    // target; asking for three only inflates the allocation without buying anything back.
    static constexpr int kMinBlocksPerSM = 4;

    inline static __device__ float Dst(const float2 &r)
    {
        float len = cuda::dot(r, r);
        return 1.0f / (len * sqrtf(len));
    }

    inline static __device__ float Lev(float tkl, float tij)
    {
        return 1.0f / (1.0f + fabsf(tkl - tij));
    }

    inline static __device__ float Dir(float dir)
    {
        // Preserve the Telea weighting floor: near-orthogonal directions get a tiny
        // nonzero contribution instead of disappearing through floating-point noise.
        return fabsf(dir) <= 0.01f ? 0.000001f : dir;
    }

    inline static __device__ float Weight(float dst, float lev, float dir)
    {
        return fabsf(dst * lev * dir);
    }

    inline static __device__ float Saturation(float Ia, float s, float Jx, float Jy)
    {
        return Ia / s + (Jx + Jy) / (sqrtf(Jx * Jx + Jy * Jy) + 1.0e-20f) + 0.5f;
    }
};

struct VarShapeTeleaPolicy
{
    // The var-shape form keeps the per-image extent live across the neighbour scan, so it is the
    // one that pays when the allocation is squeezed -- but only where 64 registers is actually
    // tight. The legacy kernel needed 72 on Blackwell and only 62 on Hopper, so asking for three
    // blocks helps the former and merely loosens the cap on the latter. CUDA 12.5 takes that
    // slack: on H100 it spends the extra registers, drops from four resident blocks to three, and
    // inpaint_uchar3_basic_planar_nchw measures +6.9% against main across five burn-in legs --
    // while the same source on the same silicon under CUDA 13.3 is within 0.1%. Ask for the
    // legacy occupancy per architecture rather than assuming one number covers both.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    static constexpr int kMinBlocksPerSM = 3;
#else
    static constexpr int kMinBlocksPerSM = 4;
#endif

    inline static __device__ float Dst(const float2 &r)
    {
        return (float)(1. / (cuda::dot(r, r) * sqrt((double)cuda::dot(r, r))));
    }

    inline static __device__ float Lev(float tkl, float tij)
    {
        return (float)(1. / (1 + fabs(tkl - tij)));
    }

    inline static __device__ float Dir(float dir)
    {
        return fabs(dir) <= 0.01 ? 0.000001f : dir;
    }

    inline static __device__ float Weight(float dst, float lev, float dir)
    {
        return (float)fabs(dst * lev * dir);
    }

    inline static __device__ float Saturation(float Ia, float s, float Jx, float Jy)
    {
        return (float)((Ia / s + (Jx + Jy) / (sqrt(Jx * Jx + Jy * Jy) + 1.0e-20f) + 0.5f));
    }
};

// ---------------------------------------------------------------------------------------------
// Fast marching
// ---------------------------------------------------------------------------------------------

inline __device__ float Min4(float a, float b, float c, float d)
{
    a = min(a, b);
    c = min(c, d);
    return min(a, c);
}

inline __device__ float FastMarchingSolve(int i1, int j1, int i2, int j2, MaskView f, DistView t)
{
    const int batch_idx = blockIdx.z;
    double    sol, a11, a22, m12;
    // Unknown distances may be written by another pixel in this sweep. They are not inputs.
    a11 = *f.ptr(batch_idx, i1, j1) != INSIDE ? *t.ptr(batch_idx, i1, j1) : 1.0e6f;
    a22 = *f.ptr(batch_idx, i2, j2) != INSIDE ? *t.ptr(batch_idx, i2, j2) : 1.0e6f;
    m12 = ((a11) > (a22) ? (a22) : (a11));

    if (*f.ptr(batch_idx, i1, j1) != INSIDE)
        if (*f.ptr(batch_idx, i2, j2) != INSIDE)
            if (fabs(a11 - a22) >= 1.0)
                sol = 1 + m12;
            else
                sol = (a11 + a22 + sqrt((double)(2 - (a11 - a22) * (a11 - a22)))) * 0.5;
        else
            sol = 1 + a11;
    else if (*f.ptr(batch_idx, i2, j2) != INSIDE)
        sol = 1 + a22;
    else
        sol = 1 + m12;

    return (float)sol;
}

template<class Policy, class OutWrapper>
inline __device__ void InpaintPixel(MaskView f, DistView t, OutWrapper out, int i, int j, int range, int channels,
                                    int rows, int cols, MaskView nextPixels)
{
    constexpr ScanMode kScanMode = OutWrapper::kScanMode;

    const int batch_idx = blockIdx.z;
    const int range2    = range * range;

    int kBegin = i - range, kEnd = i + range;
    int lBegin = j - range, lEnd = j + range;
    if constexpr (kScanMode != ScanMode::Guarded)
    {
        kBegin = kBegin > 0 ? kBegin : 1;
        kEnd   = kEnd < rows - 1 ? kEnd : rows - 2;
        lBegin = lBegin > 0 ? lBegin : 1;
        lEnd   = lEnd < cols - 1 ? lEnd : cols - 2;
    }

    for (int color = 0; color < channels; color++)
    {
        float2 gradI, gradT, r;
        float  Ia = 0, Jx = 0, Jy = 0, s = 1.0e-20f, w, dst, lev, dir, sat;

        if (*f.ptr(batch_idx, i, j + 1) != INSIDE)
        {
            if (*f.ptr(batch_idx, i, j - 1) != INSIDE)
            {
                gradT.x = (float)(*t.ptr(batch_idx, i, j + 1) - *t.ptr(batch_idx, i, j - 1)) * 0.5f;
            }
            else
            {
                gradT.x = (float)(*t.ptr(batch_idx, i, j + 1) - *t.ptr(batch_idx, i, j));
            }
        }
        else
        {
            if (*f.ptr(batch_idx, i, j - 1) != INSIDE)
            {
                gradT.x = (float)(*t.ptr(batch_idx, i, j) - *t.ptr(batch_idx, i, j - 1));
            }
            else
            {
                gradT.x = 0;
            }
        }
        if (*f.ptr(batch_idx, i + 1, j) != INSIDE)
        {
            if (*f.ptr(batch_idx, i - 1, j) != INSIDE)
            {
                gradT.y = (float)(*t.ptr(batch_idx, i + 1, j) - *t.ptr(batch_idx, i - 1, j)) * 0.5f;
            }
            else
            {
                gradT.y = (float)(*t.ptr(batch_idx, i + 1, j) - *t.ptr(batch_idx, i, j));
            }
        }
        else
        {
            if (*f.ptr(batch_idx, i - 1, j) != INSIDE)
            {
                gradT.y = (float)(*t.ptr(batch_idx, i, j) - *t.ptr(batch_idx, i - 1, j));
            }
            else
            {
                gradT.y = 0;
            }
        }
        for (int k = kBegin; k <= kEnd; k++)
        {
            int rowBegin = lBegin;
            int rowEnd   = lEnd;
            if constexpr (kScanMode == ScanMode::Disc)
            {
                const int kOffset = k - i;
                const int lRadius = (int)sqrtf((float)(range2 - kOffset * kOffset));
                rowBegin          = j - lRadius > lBegin ? j - lRadius : lBegin;
                rowEnd            = j + lRadius < lEnd ? j + lRadius : lEnd;
            }
            int km = k - 1 + (k == 1), kp = k - 1 - (k == rows - 2);
            for (int l = rowBegin; l <= rowEnd; l++)
            {
                int lm = l - 1 + (l == 1), lp = l - 1 - (l == cols - 2);
                if ((kScanMode != ScanMode::Guarded || (k > 0 && l > 0 && k < rows - 1 && l < cols - 1))
                    && (kScanMode == ScanMode::Disc || ((l - j) * (l - j) + (k - i) * (k - i) <= range2))
                    && ((i != k) || (j != l)) && (*f.ptr(batch_idx, k, l) != INSIDE)) // r != 0
                {
                    r.y = (float)(i - k);
                    r.x = (float)(j - l);

                    dst = Policy::Dst(r);
                    lev = Policy::Lev(*t.ptr(batch_idx, k, l), *t.ptr(batch_idx, i, j));

                    dir = Policy::Dir(cuda::dot(r, gradT));
                    w   = Policy::Weight(dst, lev, dir);

                    if (*f.ptr(batch_idx, k, l + 1) != INSIDE)
                    {
                        if (*f.ptr(batch_idx, k, l - 1) != INSIDE)
                        {
                            gradI.x = (float)((*out.At(batch_idx, km, lp + 1, color)
                                               - *out.At(batch_idx, km, lm - 1, color)))
                                    * 2.0f;
                        }
                        else
                        {
                            gradI.x
                                = (float)((*out.At(batch_idx, km, lp + 1, color) - *out.At(batch_idx, km, lm, color)));
                        }
                    }
                    else
                    {
                        if (*f.ptr(batch_idx, k, l - 1) != INSIDE)
                        {
                            gradI.x
                                = (float)((*out.At(batch_idx, km, lp, color) - *out.At(batch_idx, km, lm - 1, color)));
                        }
                        else
                        {
                            gradI.x = 0;
                        }
                    }
                    if (*f.ptr(batch_idx, k + 1, l) != INSIDE)
                    {
                        if (*f.ptr(batch_idx, k - 1, l) != INSIDE)
                        {
                            gradI.y = (float)((*out.At(batch_idx, kp + 1, lm, color)
                                               - *out.At(batch_idx, km - 1, lm, color)))
                                    * 2.0f;
                        }
                        else
                        {
                            gradI.y
                                = (float)((*out.At(batch_idx, kp + 1, lm, color) - *out.At(batch_idx, km, lm, color)));
                        }
                    }
                    else
                    {
                        if (*f.ptr(batch_idx, k - 1, l) != INSIDE)
                        {
                            gradI.y
                                = (float)((*out.At(batch_idx, kp, lm, color) - *out.At(batch_idx, km - 1, lm, color)));
                        }
                        else
                        {
                            gradI.y = 0;
                        }
                    }
                    Ia += (float)w * (float)(*out.At(batch_idx, km, lm, color));
                    Jx -= (float)w * (float)(gradI.x * r.x);
                    Jy -= (float)w * (float)(gradI.y * r.y);
                    s += w;
                }
            }
        }
        sat = Policy::Saturation(Ia, s, Jx, Jy);
        // Stage the saturated value so even border-gradient taps read only the previous front.
        *nextPixels.ptr(batch_idx, i, j * 4 + color) = cuda::SaturateCast<uchar>(sat);
    }
}

// The unified views are smaller than the legacy scratch structs they replaced (16-byte
// `Tensor3DWrap` against 32-byte `Ptr2dNHWC`), which shrinks the kernel argument buffer enough to
// move the register allocator's target: it starts aiming for more resident blocks than the legacy
// kernel got and pays for them in registers. That trade is backwards here. Fast-marching keeps only
// the narrow band alive, so the extra block buys nothing, while the registers it costs come back as
// `IMAD.MOV.U32` copies sitting on the dependency chain of the inner neighbour scan.
//
// The bound names an occupancy rather than a register count, because the register file per SM is not
// the same on every target: asking for four blocks caps the allocation at 64 registers, which is
// above what the legacy kernel needed on Hopper (62) but below what it needed on Blackwell (72), so
// on Blackwell that cap was binding and reintroduced exactly the copies it was added to remove.
//
// The two forms want different numbers, so each policy carries its own -- measured on a GB300 over
// the inpaint benchmark suite, where a single shared bound could only move the regression from one
// half of the suite to the other.
template<class Policy, class OutWrapper>
__launch_bounds__(kBlockS *kBlockS, Policy::kMinBlocksPerSM) __global__
    void TeleaInpaintFMM(MaskView f, DistView t, OutWrapper out, int range, MaskView band, int channels,
                         MaskView nextBand, MaskView nextPixels)
{
    // Map the x direction to columns so adjacent threads access adjacent pixels. The
    // frozen-front inputs make each target independent, so this changes only the traversal
    // order and keeps the per-pixel arithmetic unchanged.
    const int i    = blockIdx.y * blockDim.y + threadIdx.y;
    const int j    = blockIdx.x * blockDim.x + threadIdx.x;
    const int b    = blockIdx.z;
    const int rows = out.PaddedRows(b), cols = out.PaddedCols(b);
    if (i >= rows || j >= cols)
        return;

    *nextBand.ptr(b, i, j) = 0;
    if (i <= 0 || j <= 0 || i >= rows - 1 || j >= cols - 1 || *f.ptr(b, i, j) != INSIDE)
        return;
    if (!(*band.ptr(b, i - 1, j) || *band.ptr(b, i + 1, j) || *band.ptr(b, i, j - 1) || *band.ptr(b, i, j + 1)))
        return;

    // Each target has one writer. Keep f, band and out immutable until every target has read them.
    *t.ptr(b, i, j) = Min4(FastMarchingSolve(i - 1, j, i, j - 1, f, t), FastMarchingSolve(i + 1, j, i, j - 1, f, t),
                           FastMarchingSolve(i - 1, j, i, j + 1, f, t), FastMarchingSolve(i + 1, j, i, j + 1, f, t));
    InpaintPixel<Policy>(f, t, out, i, j, range, channels, rows, cols, nextPixels);
    *nextBand.ptr(b, i, j) = 1;
}

template<class OutWrapper>
__global__ void PublishInpaintFront(MaskView f, MaskView band, OutWrapper out, int channels, MaskView nextBand,
                                    MaskView nextPixels)
{
    // Keep publication coalesced in the same row-major order as the gather kernel.
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.z;
    if (i >= out.PaddedRows(b) || j >= out.PaddedCols(b))
        return;
    const uchar next   = *nextBand.ptr(b, i, j);
    *band.ptr(b, i, j) = next;
    if (next)
    {
        for (int c = 0; c < channels; ++c) *out.At(b, i - 1, j - 1, c) = *nextPixels.ptr(b, i, j * 4 + c);
        *f.ptr(b, i, j) = KNOWN;
    }
}

// ---------------------------------------------------------------------------------------------
// Scratch-plane kernels
// ---------------------------------------------------------------------------------------------

template<class SrcWrapper, class Extent>
__global__ void CopyMaskData(SrcWrapper src, MaskView dst, int rowOffset, int colOffset, uchar value, Extent extent)
{
    int       srcX      = blockIdx.x * blockDim.x + threadIdx.x;
    int       srcY      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (srcX >= extent.Cols(batch_idx) || srcY >= extent.Rows(batch_idx))
        return;

    if (*src.ptr(batch_idx, srcY, srcX, 0))
    {
        *dst.ptr(batch_idx, srcY + rowOffset, srcX + colOffset) = value;
    }
}

// Replicate-clamp, spelled as min/max rather than through cuda::GetIndexWithBorder: the two agree
// on every input, but this form folds into the hardware's fused integer min/max-with-clamp, which
// the helper's ternary chain compiles to a longer setp/select sequence inside the tap loop.
inline __device__ int ReplicateIdx(int v, int last)
{
    return max(min(v, last), 0);
}

// 3x3 cross dilation of the padded inpaint mask, replicate-clamped against the scratch plane's
// batch-wide extent (which is what the legacy BrdReplicate reader clamped to, in both paths).
template<class Extent>
__global__ void DilateKernel(MaskView src, MaskView dst, const uchar *__restrict__ kernel, const int kWidth,
                             const int kHeight, const int anchorX, const int anchorY, Extent extent, int srcRows,
                             int srcCols)
{
    uchar res = 0; // std::numeric_limits<uchar>::min(), the legacy neutral element for max

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    if (x >= extent.Cols(batch_idx) + 2 || y >= extent.Rows(batch_idx) + 2)
        return;

    int kInd = 0;

    for (int i = 0; i < kHeight; ++i)
    {
        for (int j = 0; j < kWidth; ++j)
        {
            if (kernel[kInd++])
            {
                uchar v = *src.ptr(batch_idx, ReplicateIdx(y - anchorY + i, srcRows - 1),
                                   ReplicateIdx(x - anchorX + j, srcCols - 1));
                if (v > res)
                {
                    res = v;
                }
            }
        }
    }

    *dst.ptr(batch_idx, y, x) = res;
}

template<class Extent>
__global__ void SubKernel(MaskView src1, MaskView src2, MaskView dst, Extent extent)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (x >= extent.Cols(batch_idx) + 2 || y >= extent.Rows(batch_idx) + 2)
        return;

    *dst.ptr(batch_idx, y, x) = *src1.ptr(batch_idx, y, x) - *src2.ptr(batch_idx, y, x);
}

template<class Extent>
__global__ void SetBorderHori(MaskView src, Extent extent, uchar value)
{
    const int i         = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.z;
    if (i >= extent.Cols(batch_idx) + 2)
        return;

    *src.ptr(batch_idx, 0, i)                          = value;
    *src.ptr(batch_idx, extent.Rows(batch_idx) + 1, i) = value;
}

template<class Extent>
__global__ void SetBorderVert(MaskView src, Extent extent, uchar value)
{
    const int i         = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.z;
    if (i >= extent.Rows(batch_idx) + 2)
        return;

    *src.ptr(batch_idx, i, 0)                          = value;
    *src.ptr(batch_idx, i, extent.Cols(batch_idx) + 1) = value;
}

template<typename T, class Extent>
__global__ void SetValueKernel(cuda::Tensor3DWrap<T, int32_t> src, MaskView mask, T value, Extent extent)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (x >= extent.Cols(batch_idx) + 2 || y >= extent.Rows(batch_idx) + 2)
        return;

    if (*mask.ptr(batch_idx, y, x))
    {
        *src.ptr(batch_idx, y, x) = value;
    }
}

template<typename T>
__global__ void FillValueKernel(cuda::Tensor3DWrap<T, int32_t> dst, T value, int rows, int cols)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (x >= cols || y >= rows)
        return;

    *dst.ptr(batch_idx, y, x) = value;
}

// ---------------------------------------------------------------------------------------------
// Band-population reduction: the loop termination test
// ---------------------------------------------------------------------------------------------

constexpr unsigned int kFinalMask = 0xffffffff;

template<typename T>
inline __device__ T WarpReduceSum(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) val += __shfl_xor_sync(kFinalMask, val, mask, 32);
    return val;
}

template<typename T>
inline __device__ T BlockReduceSum(T val)
{
    __shared__ T shared[32];
    int          lane = threadIdx.x & 0x1f;
    int          wid  = threadIdx.x >> 5;

    val = WarpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    // The bound is a float division so that a block size that is not a multiple of 32 still admits
    // its partial warp's slot.
    val = (threadIdx.x < (blockDim.x / 32.)) ? shared[lane] : (T)(0.0f);
    val = WarpReduceSum<T>(val);

    return val;
}

template<class Extent>
__global__ void DeviceReduceImage(MaskView src, int *out, Extent extent, int width)
{
    int batch_idx        = blockIdx.z;
    int block_idx        = blockIdx.x + blockIdx.y * gridDim.x;
    int grid_size        = gridDim.x * gridDim.y;
    int height           = extent.Rows(batch_idx) + 2;
    int tid              = threadIdx.x;
    int out_batch_offset = batch_idx * grid_size + block_idx;
    int sum              = 0;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < height * width; i += blockDim.x * gridDim.x)
    {
        int x = i % width;
        int y = i / width;
        sum += *src.ptr(batch_idx, y, x);
    }

    sum = BlockReduceSum(sum);
    if (tid == 0)
    {
        out[out_batch_offset] = sum;
    }
}

__global__ void DeviceReducePoints(const int *g_in, int *g_out, const int N)
{
    int tid = threadIdx.x;
    int sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum += g_in[i];
    }
    sum = BlockReduceSum(sum);
    if (tid == 0)
    {
        g_out[tid] = sum;
    }
}

// ---------------------------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------------------------

// The only kernel this operator dilates with is the 3x3 cross InitDilateKernel uploads, so its
// shape and anchor are fixed here rather than threaded through as parameters that never vary.
template<class Extent>
inline void DilateCrossCaller(MaskView src, MaskView dst, const uchar *kernel, Extent extent, int rows, int cols,
                              int batches, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(util::DivUp(cols, static_cast<int>(block.x)), util::DivUp(rows, static_cast<int>(block.y)), batches);

    DilateKernel<<<grid, block, 0, stream>>>(src, dst, kernel, 3, 3, 1, 1, extent, rows, cols);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<class Extent>
inline void SubHelper(MaskView src1, MaskView src2, MaskView dst, Extent extent, int rows, int cols, int batches,
                      cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(util::DivUp(cols, static_cast<int>(block.x)), util::DivUp(rows, static_cast<int>(block.y)), batches);

    SubKernel<<<grid, block, 0, stream>>>(src1, src2, dst, extent);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<class Extent>
inline void SetBorderHelper(MaskView src, Extent extent, uchar value, int rows, int cols, int batches,
                            cudaStream_t stream)
{
    dim3 block(16, 1);
    dim3 grid(util::DivUp(std::max(cols, rows), static_cast<int>(block.x)), 1, batches);

    SetBorderHori<<<grid, block, 0, stream>>>(src, extent, value);
    NVCV_CHECK_THROW(cudaGetLastError());
    SetBorderVert<<<grid, block, 0, stream>>>(src, extent, value);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, class Extent>
inline void SetValue(cuda::Tensor3DWrap<T, int32_t> src, MaskView mask, T value, Extent extent, int rows, int cols,
                     int batches, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(util::DivUp(cols, static_cast<int>(block.x)), util::DivUp(rows, static_cast<int>(block.y)), batches);

    SetValueKernel<<<grid, block, 0, stream>>>(src, mask, value, extent);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
inline void FillValue(cuda::Tensor3DWrap<T, int32_t> dst, T value, int rows, int cols, int batches, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(util::DivUp(cols, static_cast<int>(block.x)), util::DivUp(rows, static_cast<int>(block.y)), batches);

    FillValueKernel<<<grid, block, 0, stream>>>(dst, value, rows, cols);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<class Extent>
inline int FinishFlagReduce(MaskView src, int *dOut, int *dOut2, Extent extent, int cols, int batches,
                            cudaStream_t stream)
{
    DeviceReduceImage<<<dim3(kReduceGridSize, 1, batches), kReduceBlockSize, 0, stream>>>(src, dOut, extent, cols);
    NVCV_CHECK_THROW(cudaGetLastError());

    // One block folds every sample's kReduceGridSize partials into the single count read back below.
    DeviceReducePoints<<<1, kReduceGridSize, 0, stream>>>(dOut, dOut2, kReduceGridSize * batches);
    NVCV_CHECK_THROW(cudaGetLastError());

    int rst = 0;
    NVCV_CHECK_THROW(cudaMemcpyAsync(&rst, dOut2, sizeof(int), cudaMemcpyDeviceToHost, stream));
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream));
    return rst;
}

inline void InitDilateKernel(uchar *crossKernel, cudaStream_t stream)
{
    uchar K[9] = {0, 1, 0, 1, 1, 1, 0, 1, 0};
    NVCV_CHECK_THROW(cudaMemcpyAsync(crossKernel, K, 9 * sizeof(uchar), cudaMemcpyHostToDevice, stream));
}

// ---------------------------------------------------------------------------------------------
// Workspace sizing
// ---------------------------------------------------------------------------------------------

constexpr const char *kSizeOverflow = "Inpaint workspace size overflow";

// The one primitive SafeSize.hpp does not carry.
inline size_t CheckedAdd(size_t a, size_t b, const char *message)
{
    if (a > std::numeric_limits<size_t>::max() - b)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s", message);
    }
    return a + b;
}

inline size_t ComputeInpaintKernelSize(int maxBatchSize)
{
    return cvcuda::priv::CheckedMulMany({sizeof(uchar), static_cast<size_t>(maxBatchSize), 3, 3}, kSizeOverflow);
}

inline size_t ComputeInpaintWorkspaceSize(int maxBatchSize, nvcv::Size2D maxShape)
{
    size_t batch = static_cast<size_t>(maxBatchSize);
    size_t erows = CheckedAdd(static_cast<size_t>(maxShape.h), 2, kSizeOverflow);
    size_t ecols = CheckedAdd(static_cast<size_t>(maxShape.w), 2, kSizeOverflow);

    size_t reduceSize = cvcuda::priv::CheckedMul(
        sizeof(int), CheckedAdd(cvcuda::priv::CheckedMul(kReduceGridSize, batch, kSizeOverflow), 1, kSizeOverflow),
        kSizeOverflow);
    size_t imageSize
        = cvcuda::priv::CheckedMulMany({batch, erows, ecols, sizeof(float) + sizeof(uchar) * 7}, kSizeOverflow);

    return CheckedAdd(reduceSize, imageSize, kSizeOverflow);
}

// Scratch planes and staged channel values share one workspace allocation.
struct ScratchPlanes
{
    int     *reduceBuffer1;
    int     *reduceBuffer2;
    MaskView f;
    DistView t;
    MaskView band;
    MaskView inpaintMask;
    MaskView nextPixels;
    uchar   *fPtr;
    uchar   *bandPtr;
    uchar   *inpaintMaskPtr;
    size_t   planeBytes; // one uchar plane; the memsets and the carve offsets must agree on this
};

inline ScratchPlanes CarveWorkspace(void *workspace, int batch, int erows, int ecols, int maxBatchSize)
{
    ScratchPlanes p;

    p.planeBytes = sizeof(uchar) * batch * erows * ecols * 1;

    p.reduceBuffer1 = (int *)workspace;
    p.reduceBuffer2 = (int *)((char *)p.reduceBuffer1 + sizeof(int) * kReduceGridSize * maxBatchSize);

    float *tPtr      = (float *)((char *)p.reduceBuffer2 + sizeof(int) * 1);
    p.fPtr           = (uchar *)((char *)tPtr + sizeof(float) * batch * erows * ecols * 1);
    p.bandPtr        = (uchar *)((char *)p.fPtr + p.planeBytes);
    p.inpaintMaskPtr = (uchar *)((char *)p.bandPtr + p.planeBytes);

    p.f           = ScratchView<uchar>(p.fPtr, erows, ecols);
    p.t           = ScratchView<float>(tPtr, erows, ecols);
    p.band        = ScratchView<uchar>(p.bandPtr, erows, ecols);
    p.inpaintMask = ScratchView<uchar>(p.inpaintMaskPtr, erows, ecols);
    // All output types use U8 saturation; reserve four staged channel values per pixel.
    p.nextPixels = ScratchView<uchar>(p.inpaintMaskPtr + p.planeBytes, erows, ecols * 4);

    return p;
}

// ---------------------------------------------------------------------------------------------
// Validation
//
// The legacy path classified dtypes through GetLegacyDataType, which keys on the data kind plus
// the bits-per-channel -- so a packed 3-channel 8-bit type carries the same code as scalar U8 --
// and rejects any (kind, width) pair its enum had no name for.  Reproducing the classifier keeps
// the admitted set identical instead of merely similar.
// ---------------------------------------------------------------------------------------------

enum class DTypeCode
{
    U8,
    U16,
    S8,
    S16,
    S32,
    F16,
    F32,
    F64
};

inline DTypeCode ClassifyBitDepth(int32_t bpc, nvcv::DataKind kind)
{
    switch (kind)
    {
    case nvcv::DataKind::FLOAT:
        if (bpc == 64)
            return DTypeCode::F64;
        if (bpc == 32)
            return DTypeCode::F32;
        if (bpc == 16)
            return DTypeCode::F16;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for float cuda op type ", bpc);

    case nvcv::DataKind::SIGNED:
        if (bpc == 8)
            return DTypeCode::S8;
        if (bpc == 16)
            return DTypeCode::S16;
        if (bpc == 32)
            return DTypeCode::S32;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for signed cuda op type ", bpc);

    case nvcv::DataKind::UNSIGNED:
        if (bpc == 8)
            return DTypeCode::U8;
        if (bpc == 16)
            return DTypeCode::U16;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for unsigned cuda op type ", bpc);

    case nvcv::DataKind::COMPLEX:
    case nvcv::DataKind::UNSPECIFIED:
        break;
    }

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Only floating-point, signed integer and unsigned integer data kinds are supported ");
}

inline DTypeCode ClassifyDataType(nvcv::DataType dtype)
{
    auto bpc = dtype.bitsPerChannel();

    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    return ClassifyBitDepth(bpc[0], dtype.dataKind());
}

using cvcuda::priv::legacy_format::RequireUniformPlaneDataType;
using cvcuda::priv::legacy_format::RequireUniformVarShapeFormat;

inline DTypeCode ClassifyImageFormat(nvcv::ImageFormat fmt)
{
    RequireUniformPlaneDataType(fmt);

    return ClassifyDataType(fmt.planeDataType(0));
}

inline bool IsSupportedInpaintDataType(DTypeCode code)
{
    return code == DTypeCode::U8 || code == DTypeCode::S32 || code == DTypeCode::F32;
}

// Mirrors GetLegacyDataFormat(TensorLayout): only the four image layouts are named, everything
// else is rejected before the operator's own format gate can run.
inline bool IsPlanarTensorLayout(const nvcv::TensorLayout &layout)
{
    if (layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW)
    {
        return true;
    }
    if (layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_HWC)
    {
        return false;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
}

// Mirrors GetLegacyDataFormat(ImageBatchVarShapeDataStridedCuda).
inline bool IsPlanarVarShapeBatch(const nvcv::ImageBatchVarShapeDataStridedCuda &batch)
{
    nvcv::ImageFormat fmt = RequireUniformVarShapeFormat(batch);

    if (fmt.numPlanes() >= 2)
    {
        return true;
    }

    return false;
}

inline int InpaintRange(double inpaintRadius)
{
    return std::clamp((int)std::round(inpaintRadius), 1, 100);
}

// ---------------------------------------------------------------------------------------------
// Shared setup and sweep
// ---------------------------------------------------------------------------------------------

// Steps 1-4 of the Telea setup: seed the padded inpaint mask from the caller's mask, clear its
// border, initialise f and t, then dilate-minus-mask to seed the narrow band.  The inpaint-mask
// plane is cleared by the caller rather than here, because the Tensor path's mask-size rejection
// has to keep firing between that memset and this call.
template<class MaskWrapper, class Extent>
inline void InitFastMarchingPlanes(const ScratchPlanes &scratch, MaskWrapper orgMask, Extent extent, dim3 copyGrid,
                                   dim3 copyBlock, uchar *kernelPtr, bool &initFlag, bool clearBand, int erows,
                                   int ecols, int batch, cudaStream_t stream)
{
    CopyMaskData<<<copyGrid, copyBlock, 0, stream>>>(orgMask, scratch.inpaintMask, 1, 1, INSIDE, extent);
    NVCV_CHECK_THROW(cudaGetLastError());

    SetBorderHelper(scratch.inpaintMask, extent, KNOWN, erows, ecols, batch, stream);

    NVCV_CHECK_THROW(cudaMemsetAsync(scratch.fPtr, KNOWN, scratch.planeBytes, stream));
    FillValue<float>(scratch.t, 1.0e6f, erows, ecols, batch, stream);
    if (clearBand)
    {
        NVCV_CHECK_THROW(cudaMemsetAsync(scratch.bandPtr, KNOWN, scratch.planeBytes, stream));
    }

    if (!initFlag)
    {
        InitDilateKernel(kernelPtr, stream);
        initFlag = true;
    }

    DilateCrossCaller(scratch.inpaintMask, scratch.band, kernelPtr, extent, erows, ecols, batch, stream);
    SubHelper(scratch.band, scratch.inpaintMask, scratch.band, extent, erows, ecols, batch, stream);
    SetBorderHelper(scratch.band, extent, 0, erows, ecols, batch, stream);

    // The band now carries the front the sweep advances, so f and t are seeded from it.
    SetValue<uchar>(scratch.f, scratch.band, BAND, extent, erows, ecols, batch, stream);
    SetValue<uchar>(scratch.f, scratch.inpaintMask, INSIDE, extent, erows, ecols, batch, stream);
    SetValue<float>(scratch.t, scratch.band, 0, extent, erows, ecols, batch, stream);
}

// ---------------------------------------------------------------------------------------------
// Tensor path
// ---------------------------------------------------------------------------------------------

inline void CopyTensorInputToOutput(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                    const nvcv::TensorDataAccessStridedImagePlanar &outAccess, bool isPlanar,
                                    cudaStream_t stream)
{
    for (int32_t i = 0; i < inAccess.numSamples(); ++i)
    {
        if (isPlanar)
        {
            const int rowBytes = inAccess.numCols() * inAccess.colStride();
            for (int c = 0; c < inAccess.numChannels(); ++c)
            {
                void *inSampData  = inAccess.sampleData(i) + c * inAccess.chStride();
                void *outSampData = outAccess.sampleData(i) + c * outAccess.chStride();

                NVCV_CHECK_THROW(cudaMemcpy2DAsync(outSampData, outAccess.rowStride(), inSampData, inAccess.rowStride(),
                                                   rowBytes, inAccess.numRows(), cudaMemcpyDeviceToDevice, stream));
            }
        }
        else
        {
            void *inSampData  = inAccess.sampleData(i);
            void *outSampData = outAccess.sampleData(i);

            NVCV_CHECK_THROW(cudaMemcpy2DAsync(outSampData, outAccess.rowStride(), inSampData, inAccess.rowStride(),
                                               inAccess.numCols() * inAccess.colStride(), inAccess.numRows(),
                                               cudaMemcpyDeviceToDevice, stream));
        }
    }
}

// Drives the flood fill to convergence: kTeleaIterations sweeps, then a device-side count of the
// pixels still in the band.  The count is what terminates the loop, so its readback synchronises.
template<class Policy, class OutWrapper, class Extent>
inline void RunTeleaUntilConverged(const ScratchPlanes &scratch, OutWrapper dst, Extent extent, int range, int channel,
                                   int erows, int ecols, int batch, cudaStream_t stream)
{
    dim3 block(kBlockS, kBlockS);
    dim3 grid(util::DivUp(ecols, static_cast<int>(block.x)), util::DivUp(erows, static_cast<int>(block.y)), batch);

    int flag = 1;
    while (flag)
    {
        for (int i = 0; i < kTeleaIterations; i++)
        {
            TeleaInpaintFMM<Policy><<<grid, block, 0, stream>>>(scratch.inpaintMask, scratch.t, dst, range,
                                                                scratch.band, channel, scratch.f, scratch.nextPixels);
            PublishInpaintFront<<<grid, block, 0, stream>>>(scratch.inpaintMask, scratch.band, dst, channel, scratch.f,
                                                            scratch.nextPixels);
        }
        flag = FinishFlagReduce(scratch.band, scratch.reduceBuffer1, scratch.reduceBuffer2, extent, ecols, batch,
                                stream);
    }
}

template<typename T>
inline void RunTensorInpaint(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &mask,
                             const nvcv::TensorDataStridedCuda &outData, void *workspace, uchar *kernelPtr, int range,
                             bool &initFlag, int maxBatchSize, bool isPlanar, cudaStream_t stream)
{
    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);

    const int batch   = inAccess->numSamples();
    const int height  = inAccess->numRows();
    const int width   = inAccess->numCols();
    const int channel = inAccess->numChannels();
    const int erows   = height + 2;
    const int ecols   = width + 2;

    dim3 blockSize(kBlock, kBlock / 4, 1);
    dim3 gridSize(util::DivUp(ecols, static_cast<int>(blockSize.x)), util::DivUp(erows, static_cast<int>(blockSize.y)),
                  batch);

    ScratchPlanes scratch = CarveWorkspace(workspace, batch, erows, ecols, maxBatchSize);

    UniformExtent extent{height, width};

    CopyTensorInputToOutput(*inAccess, *outAccess, isPlanar, stream);

    NVCV_CHECK_THROW(cudaMemsetAsync(scratch.inpaintMaskPtr, KNOWN, scratch.planeBytes, stream));

    auto maskAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(mask);
    if (maskAccess->sampleStride() * batch > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mask size exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<int32_t>::max);
    }

    InitFastMarchingPlanes(scratch, cuda::CreateTensorWrapNHWC<uchar, int32_t>(mask), extent, gridSize, blockSize,
                           kernelPtr, initFlag, /*clearBand=*/false, erows, ecols, batch, stream);

    if (outAccess->sampleStride() * batch > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output size exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<int32_t>::max);
    }

    if (isPlanar)
    {
        TensorPlanarOut<T> dst{cuda::CreateTensorWrapNCHW<T, int32_t>(outData), erows, ecols};
        RunTeleaUntilConverged<TensorTeleaPolicy>(scratch, dst, extent, range, channel, erows, ecols, batch, stream);
    }
    else
    {
        TensorInterleavedOut<T> dst{cuda::CreateTensorWrapNHWC<T, int32_t>(outData), erows, ecols};
        RunTeleaUntilConverged<TensorTeleaPolicy>(scratch, dst, extent, range, channel, erows, ecols, batch, stream);
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------
// VarShape path
// ---------------------------------------------------------------------------------------------

inline void CopyImageToOutput(const nvcv::ImageDataStridedCuda &inimgdata, const nvcv::ImageDataStridedCuda &outimgdata,
                              cudaStream_t stream)
{
    for (int p = 0; p < inimgdata.numPlanes(); ++p)
    {
        const nvcv::ImagePlaneStrided &inplane  = inimgdata.plane(p);
        const nvcv::ImagePlaneStrided &outplane = outimgdata.plane(p);
        const int                      rowBytes = inplane.width * inimgdata.format().planePixelStrideBytes(p);
        NVCV_CHECK_THROW(cudaMemcpy2DAsync(outplane.basePtr, outplane.rowStride, inplane.basePtr, inplane.rowStride,
                                           rowBytes, inplane.height, cudaMemcpyDeviceToDevice, stream));
    }
}

template<typename T>
inline void RunVarShapeInpaint(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                               const nvcv::ImageBatchVarShapeDataStridedCuda &mask,
                               const nvcv::ImageBatchVarShapeDataStridedCuda &outData, void *workspace,
                               uchar *kernelPtr, int range, bool &initFlag, int channel, int maxBatchSize,
                               bool isPlanar, cudaStream_t stream)
{
    const nvcv::Size2D maxsize = inData.maxSize();
    const int          batch   = inData.numImages();
    const int          erows   = maxsize.h + 2;
    const int          ecols   = maxsize.w + 2;

    dim3 blockSize(kBlock, kBlock / 4, 1);
    dim3 gridSize(util::DivUp(ecols, static_cast<int>(blockSize.x)), util::DivUp(erows, static_cast<int>(blockSize.y)),
                  batch);

    // data type for mask is 8UC1
    cuda::ImageBatchVarShapeWrapNHWC<uchar> orgMask(mask, 1);

    PerImageExtent<cuda::ImageBatchVarShapeWrapNHWC<uchar>> extent{orgMask};

    ScratchPlanes scratch = CarveWorkspace(workspace, batch, erows, ecols, maxBatchSize);

    // The input-to-output copy is done by the caller: it iterates the ImageBatchVarShape handles,
    // which this function never receives.
    NVCV_CHECK_THROW(cudaMemsetAsync(scratch.inpaintMaskPtr, KNOWN, scratch.planeBytes, stream));

    InitFastMarchingPlanes(scratch, orgMask, extent, gridSize, blockSize, kernelPtr, initFlag, /*clearBand=*/true,
                           erows, ecols, batch, stream);

    if (isPlanar)
    {
        VarShapePlanarOut<T> dst{cuda::ImageBatchVarShapeWrap<T>(outData)};
        RunTeleaUntilConverged<VarShapeTeleaPolicy>(scratch, dst, extent, range, channel, erows, ecols, batch, stream);
    }
    else
    {
        VarShapeInterleavedOut<T> dst{cuda::ImageBatchVarShapeWrapNHWC<T>(outData, channel)};
        RunTeleaUntilConverged<VarShapeTeleaPolicy>(scratch, dst, extent, range, channel, erows, ecols, batch, stream);
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

} // namespace

namespace cvcuda::priv {

Inpaint::DeviceBuffers::DeviceBuffers(int maxBatchSize, nvcv::Size2D maxShape)
{
    NVCV_CHECK_THROW(cudaMalloc(&kernel, ComputeInpaintKernelSize(maxBatchSize)));

    try
    {
        NVCV_CHECK_THROW(cudaMalloc(&workspace, ComputeInpaintWorkspaceSize(maxBatchSize, maxShape)));
    }
    catch (...)
    {
        NVCV_CHECK_LOG(cudaFree(kernel));
        throw;
    }
}

Inpaint::DeviceBuffers::~DeviceBuffers()
{
    NVCV_CHECK_LOG(cudaFree(kernel));
    NVCV_CHECK_LOG(cudaFree(workspace));
}

Inpaint::Inpaint(int maxBatchSize, nvcv::Size2D maxShape)
    // The scratch buffers are allocated for one CUDA device.  PerDeviceResource creates
    // one set per device for transparent multi-GPU support.
    : m_maxBatchSize(maxBatchSize)
    , m_maxShape(maxShape)
    , m_buffers([maxBatchSize, maxShape](int) { return std::make_unique<DeviceBuffers>(maxBatchSize, maxShape); })
    , m_buffersVarShape([maxBatchSize, maxShape](int)
                        { return std::make_unique<DeviceBuffers>(maxBatchSize, maxShape); })
{
}

void Inpaint::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &masks,
                         const nvcv::Tensor &out, double inpaintRadius) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Inpaint::operator()[Tensor]");
    auto inData    = RequireCudaTensor(in, "Input");
    auto masksData = RequireCudaTensor(masks, "Masks");
    auto outData   = RequireCudaTensor(out, "Output");

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be image-compatible tensor");
    }

    if (inAccess->numSamples() > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input batch exceeds maxBatchSize");
    }
    if (inAccess->numCols() > m_maxShape.w || inAccess->numRows() > m_maxShape.h)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input shape exceeds maxShape");
    }

    const bool      isPlanar = IsPlanarTensorLayout(inData->layout());
    const DTypeCode inCode   = ClassifyDataType(inData->dtype());

    if (!IsSupportedInpaintDataType(inCode))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input DataType");
    }

    const int inChannels = inAccess->numChannels();

    if (inChannels > 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", inChannels);
    }
    if (isPlanar && inChannels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-channel planar Inpaint is unsupported");
    }

    const bool      outIsPlanar = IsPlanarTensorLayout(outData->layout());
    const DTypeCode outCode     = ClassifyDataType(outData->dtype());

    if (outIsPlanar != isPlanar)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must both be interleaved or both planar");
    }

    if (inCode != outCode)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "DataType of input and output must be equal");
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    if (!outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be image-compatible tensor");
    }
    const int outChannels = outAccess->numChannels();

    if (outChannels != inChannels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "input and output channel number must be equal, but got %d and %d", inChannels,
                              outChannels);
    }

    if (ClassifyDataType(masksData->dtype()) != DTypeCode::U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid mask DataType");
    }

    auto maskAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*masksData);
    if (!maskAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Masks must be image-compatible tensor");
    }
    const int maskChannels = maskAccess->numChannels();
    if (maskChannels != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid mask channel number %d", maskChannels);
    }

    // RunTensorInpaint copies the input into the output over the input's extent and reads the mask
    // over the same extent, so a smaller output or mask would be written and read past its end.
    if (inAccess->numSamples() != outAccess->numSamples() || inAccess->numRows() != outAccess->numRows()
        || inAccess->numCols() != outAccess->numCols())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same shape");
    }
    if (inAccess->numSamples() != maskAccess->numSamples() || inAccess->numRows() != maskAccess->numRows()
        || inAccess->numCols() != maskAccess->numCols())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Masks must have the same sample count and size as the input");
    }

    DeviceBuffers &buffers = m_buffers.get();

    const int range = InpaintRange(inpaintRadius);

    switch (inCode)
    {
    case DTypeCode::U8:
        RunTensorInpaint<uchar>(*inData, *masksData, *outData, buffers.workspace, buffers.kernel, range,
                                buffers.initDilate, m_maxBatchSize, isPlanar, stream);
        break;
    case DTypeCode::S32:
        RunTensorInpaint<int>(*inData, *masksData, *outData, buffers.workspace, buffers.kernel, range,
                              buffers.initDilate, m_maxBatchSize, isPlanar, stream);
        break;
    default: // DTypeCode::F32 -- the dtype gate above admits exactly U8, S32 and F32
        RunTensorInpaint<float>(*inData, *masksData, *outData, buffers.workspace, buffers.kernel, range,
                                buffers.initDilate, m_maxBatchSize, isPlanar, stream);
        break;
    }
}

void Inpaint::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &masks,
                         const nvcv::ImageBatchVarShape &out, double inpaintRadius) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Inpaint::operator()[ImageBatchVarShape]");
    if (in.numImages() > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input batch exceeds maxBatchSize");
    }
    if (in.numImages() != masks.numImages() || in.numImages() != out.numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input, masks, and output must have the same number of images");
    }
    // CopyImageToOutput and the Telea setup both use the input extent for every image, so a smaller
    // output or mask image would be accessed past its own buffer.
    for (int i = 0; i < in.numImages(); ++i)
    {
        const nvcv::Size2D inSize   = in[i].size();
        const nvcv::Size2D outSize  = out[i].size();
        const nvcv::Size2D maskSize = masks[i].size();
        if (inSize != outSize || inSize != maskSize)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input, mask and output image %d must have the same size", i);
        }
    }

    if (nvcv::Size2D inMaxSize = in.maxSize(); inMaxSize.w > m_maxShape.w || inMaxSize.h > m_maxShape.h)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input shape exceeds maxShape");
    }

    auto masksData = masks.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (masksData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Masks must be varshape image batch");
    }

    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }
    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    const bool      isPlanar = IsPlanarVarShapeBatch(*inData);
    const DTypeCode inCode   = ClassifyImageFormat(inData->uniqueFormat());

    if (!IsSupportedInpaintDataType(inCode))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input DataType");
    }

    const int inChannels = inData->uniqueFormat().numChannels();

    if (inChannels > 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", inChannels);
    }
    if (isPlanar && inChannels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-channel planar Inpaint is unsupported");
    }

    const bool      outIsPlanar = IsPlanarVarShapeBatch(*outData);
    const DTypeCode outCode     = ClassifyImageFormat(outData->uniqueFormat());

    if (outIsPlanar != isPlanar)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataFormat between input and output");
    }

    if (outCode != inCode)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType between input and output");
    }

    const int outChannels = outData->uniqueFormat().numChannels();

    if (outChannels != inChannels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "input and output channel number must be equal, but got %d and %d", inChannels,
                              outChannels);
    }

    if (ClassifyImageFormat(masksData->uniqueFormat()) != DTypeCode::U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid mask DataType");
    }

    const int maskChannels = masksData->uniqueFormat().numChannels();
    if (maskChannels != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid mask channel number %d", maskChannels);
    }

    //copy input to output
    for (auto init = in.begin(), outit = out.begin(); init != in.end() && outit != out.end(); ++init, ++outit)
    {
        const nvcv::Image &inimg      = *init;
        const nvcv::Image &outimg     = *outit;
        auto               inimgdata  = inimg.exportData<nvcv::ImageDataStridedCuda>();
        auto               outimgdata = outimg.exportData<nvcv::ImageDataStridedCuda>();
        CopyImageToOutput(*inimgdata, *outimgdata, stream);
    }

    DeviceBuffers &buffers = m_buffersVarShape.get();

    const int range = InpaintRange(inpaintRadius);

    switch (inCode)
    {
    case DTypeCode::U8:
        RunVarShapeInpaint<uchar>(*inData, *masksData, *outData, buffers.workspace, buffers.kernel, range,
                                  buffers.initDilate, inChannels, m_maxBatchSize, isPlanar, stream);
        break;
    case DTypeCode::S32:
        RunVarShapeInpaint<int>(*inData, *masksData, *outData, buffers.workspace, buffers.kernel, range,
                                buffers.initDilate, inChannels, m_maxBatchSize, isPlanar, stream);
        break;
    default: // DTypeCode::F32 -- the dtype gate above admits exactly U8, S32 and F32
        RunVarShapeInpaint<float>(*inData, *masksData, *outData, buffers.workspace, buffers.kernel, range,
                                  buffers.initDilate, inChannels, m_maxBatchSize, isPlanar, stream);
        break;
    }
}

} // namespace cvcuda::priv
