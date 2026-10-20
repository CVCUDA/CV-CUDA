/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

#include "Nvtx.hpp"
#include "OpWarpAffine.hpp"
#include "TensorAccessValidation.hpp"

#include "WarpCommon.cuh"

#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/math/LinAlg.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>

#include <cmath>
#include <cstdint>
#include <memory>

namespace cuda = nvcv::cuda;
namespace warp = cvcuda::priv::warp;

namespace {

inline bool IsExactlyRepresentableInteger(float value)
{
    constexpr float kMaxExactInteger = 16777216.0f;
    return std::isfinite(value) && -kMaxExactInteger <= value && value <= kMaxExactInteger
        && std::trunc(value) == value;
}

// Precondition for sampling an all-integer inverse affine map with the nearest specialization
// instead of the cubic one. With NVCV_WARP_INVERSE_MAP the caller's six coefficients are the
// destination-to-source map itself, so when all six are exact integers every integer destination
// pixel maps to an integer source coordinate; the cubic weights GetCubicCoeffs produces for a zero
// fractional part are exactly (0, 1, 0, 0), so the 4x4 gather returns the center tap -- the same
// pixel the nearest sampler reads, resolved through the same border rule. The window is kept as
// narrow: interleaved or planar 8-bit 3-channel input and output with REFLECT borders.
// It does not carry over to WarpPerspective,
// whose homogeneous divide leaves the source coordinate non-integral even for an integer matrix.
//
// `params` comes from warp::ValidateTensor, which has already classified the input dtype and
// channel count and established that both tensors share a layout kind, so nothing here throws.
inline bool IsIntegerGridU8CubicReflect(const nvcv::TensorDataStridedCuda &inData,
                                        const nvcv::TensorDataStridedCuda &outData, const NVCVAffineTransform xform,
                                        const int32_t flags, const NVCVBorderType borderMode,
                                        const warp::WarpParams &params)
{
    if (flags != (NVCV_INTERP_CUBIC | NVCV_WARP_INVERSE_MAP) || borderMode != NVCV_BORDER_REFLECT)
    {
        return false;
    }

    const bool isPlanar = warp::IsPlanar(params.layout);
    if ((params.layout != warp::LayoutKind::kNHWC && !isPlanar) || params.elem != warp::ElemKind::kU8
        || params.channels != 3 || inData.dtype() != outData.dtype())
    {
        return false;
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess || outAccess->numChannels() != 3)
    {
        return false;
    }

    for (int i = 0; i < 6; ++i)
    {
        if (!IsExactlyRepresentableInteger(xform[i]))
        {
            return false;
        }
    }
    if (isPlanar)
    {
        constexpr double kMaxHalfIntegerCoordinate = 8388608.0;
        const double     lastX                     = outAccess->numCols() - 1;
        const double     lastY                     = outAccess->numRows() - 1;
        for (int coefficient = 0; coefficient < 6; coefficient += 3)
        {
            const double extent = std::abs(static_cast<double>(xform[coefficient])) * lastX
                                + std::abs(static_cast<double>(xform[coefficient + 1])) * lastY
                                + std::abs(static_cast<double>(xform[coefficient + 2]));
            if (extent >= kMaxHalfIntegerCoordinate)
            {
                return false;
            }
        }
    }
    return true;
}

// One thread per image inverts that image's 2x3 affine matrix in place into the operator's
// coefficient buffer, which is strided by 9 floats per image because the var-shape warp kernels
// are shared with WarpPerspective.
__global__ void inverseMatWarpAffine(const int numImages, const cuda::Tensor2DWrap<float, int32_t> in,
                                     cuda::Tensor2DWrap<float, int32_t> out)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numImages)
    {
        return;
    }

    cuda::math::Matrix<float, 2, 3> M;
    M[0][0] = (float)(*in.ptr(index, 0));
    M[0][1] = (float)(*in.ptr(index, 1));
    M[0][2] = (float)(*in.ptr(index, 2));
    M[1][0] = (float)(*in.ptr(index, 3));
    M[1][1] = (float)(*in.ptr(index, 4));
    M[1][2] = (float)(*in.ptr(index, 5));

    // M is stored in row-major format M[0,0], M[0,1], M[0,2], M[1,0], M[1,1], M[1,2]
    float den          = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    den                = std::abs(den) > 1e-5 ? 1. / den : .0;
    *out.ptr(index, 0) = (float)M[1][1] * den;
    *out.ptr(index, 1) = (float)-M[0][1] * den;
    *out.ptr(index, 2) = (float)(M[0][1] * M[1][2] - M[1][1] * M[0][2]) * den;
    *out.ptr(index, 3) = (float)-M[1][0] * den;
    *out.ptr(index, 4) = (float)M[0][0] * den;
    *out.ptr(index, 5) = (float)(M[1][0] * M[0][2] - M[0][0] * M[1][2]) * den;
}

inline void invertMat(const float *M, float *h_aCoeffs)
{
    // M is stored in row-major format M[0,0], M[0,1], M[0,2], M[1,0], M[1,1], M[1,2]
    float den    = M[0] * M[4] - M[1] * M[3];
    den          = std::abs(den) > 1e-5 ? 1. / den : .0;
    h_aCoeffs[0] = (float)M[4] * den;
    h_aCoeffs[1] = (float)-M[1] * den;
    h_aCoeffs[2] = (float)(M[1] * M[5] - M[4] * M[2]) * den;
    h_aCoeffs[3] = (float)-M[3] * den;
    h_aCoeffs[4] = (float)M[0] * den;
    h_aCoeffs[5] = (float)(M[3] * M[2] - M[0] * M[5]) * den;
}

} // namespace

namespace cvcuda::priv {

WarpAffine::WarpAffine(const int32_t maxVarShapeBatchSize)
    : m_maxVarShapeBatchSize(maxVarShapeBatchSize)
    , m_transformMatrix([maxVarShapeBatchSize](int)
                        { return std::make_unique<warp::TransformMatrix>(maxVarShapeBatchSize); })
{
}

void WarpAffine::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                            const NVCVAffineTransform xform, const int32_t flags, const NVCVBorderType borderMode,
                            const float4 borderValue) const
{
    CVCUDA_NVTX_RANGE("cvcuda::WarpAffine::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    warp::WarpParams params = warp::ValidateTensor(*inData, *outData, flags, borderMode);

    // Integer inverse coordinates make cubic weights collapse to the center tap; use the
    // cheaper nearest specialization for the RGB8 reflect path where that is bit-exact.
    if (IsIntegerGridU8CubicReflect(*inData, *outData, xform, flags, borderMode, params))
    {
        params.interpolation = NVCV_INTERP_NEAREST;
    }

    // The kernels stage all nine coefficients (the storage is shared with WarpPerspective) but an
    // affine map only reads the first six, so the trailing three are zeroed rather than left
    // indeterminate.
    warp::WarpAffineTransform transform = {};

    if (flags & NVCV_WARP_INVERSE_MAP)
    {
        for (int i = 0; i < warp::kNumCoeffs; i++)
        {
            transform.xform[i] = i < 6 ? (float)(xform[i]) : 0.0f;
        }
    }
    else
    {
        invertMat(xform, transform.xform);
    }

    warp::RunWarpTensor<warp::WarpAffineTransform>(*inData, *outData, transform, params, borderMode, borderValue,
                                                   stream);
}

void WarpAffine::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                            const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &transMatrix, const int32_t flags,
                            const NVCVBorderType borderMode, const float4 borderValue) const
{
    CVCUDA_NVTX_RANGE("cvcuda::WarpAffine::operator()[ImageBatchVarShape]");
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

    auto transMatrixData = RequireCudaTensor(transMatrix, "transformation matrix");

    if (m_maxVarShapeBatchSize <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Operator warp affine var shape is not initialized properly, maxVarShapeBatchSize: %d",
                              m_maxVarShapeBatchSize);
    }

    RequireUniqueFormat(*inData, "input");

    if (m_maxVarShapeBatchSize < inData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid number of images, it should not exceed %d",
                              m_maxVarShapeBatchSize);
    }

    const warp::WarpParams params = warp::ValidateVarShape(*inData, *outData, flags, borderMode);

    // Both branches below read the six affine coefficients for every image out of transMatrix, so
    // reject a matrix that cannot supply them before wrapping it. Note this is 6, not kNumCoeffs:
    // the nine-float stride belongs to the destination buffer, which is shared with WarpPerspective.
    constexpr int kNumAffineCoeffs = 6;

    if (transMatrixData->dtype() != nvcv::TYPE_F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Transformation matrix must be F32");
    }
    if (transMatrixData->rank() != 2 || transMatrixData->shape(0) < inData->numImages()
        || transMatrixData->shape(1) < kNumAffineCoeffs)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Transformation matrix must be a 2D tensor of at least %d x %d, one row per image",
                              inData->numImages(), kNumAffineCoeffs);
    }

    float *d_matrix = m_transformMatrix.get().matrix;

    // Wrap the matrix in 2D wrappers with proper pitch
    cuda::Tensor2DWrap<float, int32_t> transMatrixInput(*transMatrixData);
    cuda::Tensor2DWrap<float, int32_t> transMatrixOutput(d_matrix, static_cast<int>(sizeof(float) * warp::kNumCoeffs));

    // Check if inverse op is needed
    const bool performInverse = !(flags & NVCV_WARP_INVERSE_MAP);

    if (performInverse)
    {
        // Cap at 256 threads per block so the launch remains valid when
        // numImages exceeds the per-block thread limit (typically 1024).
        constexpr int kThreadsPerBlock = 256;
        const int     numBlocks        = (inData->numImages() + kThreadsPerBlock - 1) / kThreadsPerBlock;
        inverseMatWarpAffine<<<numBlocks, kThreadsPerBlock, 0, stream>>>(inData->numImages(), transMatrixInput,
                                                                         transMatrixOutput);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        NVCV_CHECK_THROW(cudaMemcpy2DAsync(d_matrix, sizeof(float) * warp::kNumCoeffs, transMatrixInput.ptr(0, 0),
                                           transMatrixInput.strides()[0], sizeof(float) * 6, inData->numImages(),
                                           cudaMemcpyDeviceToDevice, stream));
    }

    warp::RunWarpVarShape<warp::WarpAffineTransform>(*inData, *outData, transMatrixOutput, params, borderMode,
                                                     borderValue, stream);
}

} // namespace cvcuda::priv
