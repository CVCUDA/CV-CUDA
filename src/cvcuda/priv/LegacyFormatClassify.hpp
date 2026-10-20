/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCUDA_PRIV_LEGACY_FORMAT_CLASSIFY_HPP
#define CVCUDA_PRIV_LEGACY_FORMAT_CLASSIFY_HPP

#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/TensorLayout.hpp>

#include <cstdint>

// Shared dtype/format classifiers; operators depend on their rejection order and messages.

namespace cvcuda::priv::legacy_format {

// Codes classify a dtype by data kind plus bits-per-channel, not by exact type, so packed
// multi-channel types (TYPE_3U8, TYPE_4U8, ...) share their scalar counterpart's code.
// The numeric values are printed in rejection messages, so do not renumber.
enum class DataTypeCode : int
{
    kU8  = 0,
    kS8  = 1,
    kU16 = 2,
    kS16 = 3,
    kS32 = 4,
    kF32 = 5,
    kF64 = 6,
    kF16 = 7,
};

// Values appear in rejection messages, so do not renumber.
enum class DataFormatCode : int
{
    kNCHW = 0,
    kNHWC = 1,
    kCHW  = 2,
    kHWC  = 3,
};

inline bool IsPlanar(DataFormatCode format)
{
    return format == DataFormatCode::kNCHW || format == DataFormatCode::kCHW;
}

// Rejects every (kind, bpc) pair without a DataTypeCode (32/64-bit unsigned, 64-bit signed,
// 8-bit float, complex, unspecified); callers classify before checking their own dtype list,
// so these rejections take precedence.
inline DataTypeCode ClassifyDataType(int32_t bpc, nvcv::DataKind kind)
{
    switch (kind)
    {
    case nvcv::DataKind::FLOAT:
        if (bpc == 64)
            return DataTypeCode::kF64;
        if (bpc == 32)
            return DataTypeCode::kF32;
        if (bpc == 16)
            return DataTypeCode::kF16;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for float cuda op type ", bpc);

    case nvcv::DataKind::SIGNED:
        if (bpc == 8)
            return DataTypeCode::kS8;
        if (bpc == 16)
            return DataTypeCode::kS16;
        if (bpc == 32)
            return DataTypeCode::kS32;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for signed cuda op type ", bpc);

    case nvcv::DataKind::UNSIGNED:
        if (bpc == 8)
            return DataTypeCode::kU8;
        if (bpc == 16)
            return DataTypeCode::kU16;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for unsigned cuda op type ", bpc);

    case nvcv::DataKind::COMPLEX:
    case nvcv::DataKind::UNSPECIFIED:
        break;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Only floating-point, signed integer and unsigned integer data kinds are supported ");
}

inline DataTypeCode ClassifyDataType(const nvcv::DataType &dtype)
{
    // Mixed-width packings (e.g. X32_Y24b8) exist, so reject them before classifying.
    auto      bpc         = dtype.bitsPerChannel();
    const int numChannels = dtype.numChannels();
    for (int i = 1; i < numChannels; ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    return ClassifyDataType(bpc[0], dtype.dataKind());
}

inline void RequireUniformPlaneDataType(const nvcv::ImageFormat &fmt)
{
    // numPlanes()/planeDataType() cross the nvcv shared-library boundary, so they cannot be hoisted
    // by the compiler.
    const int            numPlanes = fmt.numPlanes();
    const nvcv::DataType plane0    = fmt.planeDataType(0);
    for (int i = 1; i < numPlanes; ++i)
    {
        if (fmt.planeDataType(i) != plane0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }
}

inline DataTypeCode ClassifyDataType(const nvcv::ImageFormat &fmt)
{
    RequireUniformPlaneDataType(fmt);
    return ClassifyDataType(fmt.planeDataType(0));
}

// Any layout outside these four throws here, before the operator's own format allowlist runs.
inline DataFormatCode ClassifyDataFormat(const nvcv::TensorLayout &layout)
{
    if (layout == nvcv::TENSOR_NCHW)
    {
        return DataFormatCode::kNCHW;
    }
    if (layout == nvcv::TENSOR_CHW)
    {
        return DataFormatCode::kCHW;
    }
    if (layout == nvcv::TENSOR_NHWC)
    {
        return DataFormatCode::kNHWC;
    }
    if (layout == nvcv::TENSOR_HWC)
    {
        return DataFormatCode::kHWC;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
}

// Checks, in throw order: one uniqueFormat, the same plane data type, one channel per plane for planar.
template<class Batch>
inline nvcv::ImageFormat RequireUniformVarShapeFormat(const Batch &batch)
{
    nvcv::ImageFormat fmt = batch.uniqueFormat();
    if (!fmt)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All images must have the same format");
    }

    RequireUniformPlaneDataType(fmt);

    if (fmt.numPlanes() >= 2 && fmt.numPlanes() != fmt.numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Planar images must have one channel per plane");
    }
    return fmt;
}

// One image classifies as CHW/HWC and 2+ as NCHW/NHWC, so 1-vs-2+ image counts mismatch.
inline DataFormatCode ClassifyDataFormat(const nvcv::ImageBatchVarShapeDataStridedCuda &imgBatch)
{
    if (RequireUniformVarShapeFormat(imgBatch).numPlanes() >= 2)
    {
        return imgBatch.numImages() >= 2 ? DataFormatCode::kNCHW : DataFormatCode::kCHW;
    }

    return imgBatch.numImages() >= 2 ? DataFormatCode::kNHWC : DataFormatCode::kHWC;
}

} // namespace cvcuda::priv::legacy_format

#endif // CVCUDA_PRIV_LEGACY_FORMAT_CLASSIFY_HPP
