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

#include "DLPackUtils.hpp"

#include "DataType.hpp"

#include <common/PyUtil.hpp>
#include <nvcv/TensorData.hpp>

namespace nvcvpy::priv {

DLPackTensor::DLPackTensor() noexcept
    : m_tensor{}
{
}

DLPackTensor::DLPackTensor(DLManagedTensor &&managedTensor)
    : m_tensor{std::move(managedTensor)}
{
    managedTensor = {};
}

DLPackTensor::DLPackTensor(const DLTensor &tensor)
    : DLPackTensor(DLManagedTensor{tensor})
{
}

DLPackTensor::DLPackTensor(const py::buffer_info &info, const DLDevice &dev)
    : m_tensor{}
{
    DLTensor &dlTensor = m_tensor.dl_tensor;
    dlTensor.data      = info.ptr;

    if (std::optional<nvcv::DataType> dt = ToNVCVDataType(util::ToDType(info)))
    {
        dlTensor.dtype = ToDLDataType(*dt);
    }
    else
    {
        throw std::runtime_error("Cannot wrap buffer with given format");
    }
    dlTensor.ndim        = info.ndim;
    dlTensor.device      = dev;
    dlTensor.byte_offset = 0;

    m_tensor.deleter = [](DLManagedTensor *self)
    {
        delete[] self->dl_tensor.shape;
        self->dl_tensor.shape = nullptr;

        delete[] self->dl_tensor.strides;
        self->dl_tensor.strides = nullptr;
    };

    try
    {
        dlTensor.shape = new int64_t[info.ndim];
        std::copy_n(info.shape.begin(), info.shape.size(), dlTensor.shape);

        dlTensor.strides = new int64_t[info.ndim];
        for (int i = 0; i < info.ndim; ++i)
        {
            if (info.strides[i] % info.itemsize != 0)
            {
                throw std::runtime_error("Stride must be a multiple of the element size in bytes");
            }

            dlTensor.strides[i] = info.strides[i] / info.itemsize;
        }
    }
    catch (...)
    {
        m_tensor.deleter(&m_tensor);
        throw;
    }
}

DLPackTensor::DLPackTensor(const nvcv::TensorDataStrided &tensorData)
{
    m_tensor         = {};
    m_tensor.deleter = [](DLManagedTensor *self)
    {
        delete[] self->dl_tensor.shape;
        delete[] self->dl_tensor.strides;
    };

    try
    {
        DLTensor &tensor = m_tensor.dl_tensor;

        // Set up device
        if (tensorData.IsCompatible<nvcv::TensorDataStridedCuda>())
        {
            // TODO: detect correct device_type from memory buffer
            tensor.device.device_type = kDLCUDA;
            // TODO: detect correct device_id from memory buffer (if possible)
            tensor.device.device_id = 0;
        }
        else
        {
            throw std::runtime_error("Tensor buffer type not supported, must be either CUDA or Host (CPU)");
        }

        // Set up ndim
        tensor.ndim = tensorData.rank();

        // Set up data
        tensor.data        = tensorData.basePtr();
        tensor.byte_offset = 0;

        // Set up shape
        tensor.shape = new int64_t[tensor.ndim];
        std::copy_n(tensorData.shape().shape().begin(), tensor.ndim, tensor.shape);

        // Set up dtype
        tensor.dtype = ToDLDataType(tensorData.dtype());

        // Set up strides
        tensor.strides = new int64_t[tensor.ndim];
        for (int i = 0; i < tensor.ndim; ++i)
        {
            int64_t stride = tensorData.cdata().buffer.strided.strides[i];

            if (stride % tensorData.dtype().strideBytes() != 0)
            {
                throw std::runtime_error("Stride must be multiple of the element size in bytes");
            }

            tensor.strides[i] = tensorData.cdata().buffer.strided.strides[i] / tensorData.dtype().strideBytes();
        }
    }
    catch (...)
    {
        m_tensor.deleter(&m_tensor);
        throw;
    }
}

DLPackTensor::DLPackTensor(DLPackTensor &&that) noexcept
    : m_tensor{std::move(that.m_tensor)}
{
    that.m_tensor = {};
}

DLPackTensor::~DLPackTensor()
{
    if (m_tensor.deleter)
    {
        m_tensor.deleter(&m_tensor);
    }
}

DLPackTensor &DLPackTensor::operator=(DLPackTensor &&that) noexcept
{
    if (this != &that)
    {
        if (m_tensor.deleter)
        {
            m_tensor.deleter(&m_tensor);
        }
        m_tensor = std::move(that.m_tensor);

        that.m_tensor = {};
    }
    return *this;
}

const DLTensor *DLPackTensor::operator->() const
{
    return &m_tensor.dl_tensor;
}

DLTensor *DLPackTensor::operator->()
{
    return &m_tensor.dl_tensor;
}

const DLTensor &DLPackTensor::operator*() const
{
    return m_tensor.dl_tensor;
}

DLTensor &DLPackTensor::operator*()
{
    return m_tensor.dl_tensor;
}

bool IsCudaAccessible(DLDeviceType devType)
{
    switch (devType)
    {
    case kDLCUDAHost:
    case kDLCUDA:
    case kDLCUDAManaged:
        return true;
    default:
        return false;
    }
}

nvcv::DataType ToNVCVDataType(const DLDataType &dtype)
{
    nvcv::PackingParams pp;
    pp.byteOrder = nvcv::ByteOrder::LSB;

    int lanes = dtype.lanes;
    int bits  = dtype.bits;

    switch (lanes)
    {
    case 1:
        pp.swizzle = nvcv::Swizzle::S_X000;
        break;
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
        throw std::runtime_error("DLPack buffer's data type must have at most 4 lanes");
    }

    for (int i = 0; i < lanes; ++i)
    {
        pp.bits[i] = bits;
    }
    for (int i = lanes; i < 4; ++i)
    {
        pp.bits[i] = 0;
    }

    nvcv::Packing packing = nvcv::MakePacking(pp);

    nvcv::DataKind kind;

    switch (dtype.code)
    {
    case kDLBool:
    case kDLInt:
        kind = nvcv::DataKind::SIGNED;
        break;
    case kDLUInt:
        kind = nvcv::DataKind::UNSIGNED;
        break;
    case kDLComplex:
        kind = nvcv::DataKind::COMPLEX;
        break;
    case kDLFloat:
        kind = nvcv::DataKind::FLOAT;
        break;
    default:
        throw std::runtime_error("Data type code not supported, must be Int, UInt, Float, Complex or Bool");
    }

    return nvcv::DataType(kind, packing);
}

DLDataType ToDLDataType(const nvcv::DataType &dataType)
{
    DLDataType dt = {};
    dt.lanes      = dataType.numChannels();

    switch (dataType.dataKind())
    {
    case nvcv::DataKind::UNSIGNED:
        dt.code = kDLUInt;
        break;
    case nvcv::DataKind::SIGNED:
        dt.code = kDLInt;
        break;
    case nvcv::DataKind::FLOAT:
        dt.code = kDLFloat;
        break;
    case nvcv::DataKind::COMPLEX:
        dt.code = kDLComplex;
        break;
    default:
        throw std::runtime_error("Data kind not supported, must be UNSIGNED, SIGNED, FLOAT or COMPLEX");
    }

    std::array<int32_t, 4> bpc = dataType.bitsPerChannel();

    for (int i = 1; i < dataType.numChannels(); ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw std::runtime_error("All lanes must have the same bit depth");
        }
    }

    dt.bits = bpc[0];

    return dt;
}

} // namespace nvcvpy::priv
