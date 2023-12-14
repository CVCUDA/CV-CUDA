/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ExternalBuffer.hpp"

#include "DataType.hpp"

#include <common/Assert.hpp>
#include <common/PyUtil.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <functional> // for std::multiplies

namespace nvcvpy::priv {

using namespace py::literals;

static void CheckValidCUDABuffer(const void *ptr)
{
    if (ptr == nullptr)
    {
        throw std::runtime_error("NULL CUDA buffer not accepted");
    }

    cudaPointerAttributes attrs = {};
    cudaError_t           err   = cudaPointerGetAttributes(&attrs, ptr);
    cudaGetLastError(); // reset the cuda error (if any)
    if (err != cudaSuccess || attrs.type == cudaMemoryTypeUnregistered)
    {
        throw std::runtime_error("Buffer is not CUDA-accessible");
    }
}

static std::string ToFormatString(const DLDataType &dtype)
{
    py::dtype dt = ToDType(ToNVCVDataType(dtype));
    return dt.attr("str").cast<std::string>();
}

py::object ExternalBuffer::Create(DLPackTensor &&dlPackTensor, py::object wrappedObj)
{
    std::shared_ptr<ExternalBuffer> buf(new ExternalBuffer(std::move(dlPackTensor)));

    // We must make the returned object keep wrappedObj alive.
    // Using py::return_value_policy::reference_internal in py::cast doesn't work
    // because buf is a shared_ptr.
    py::object o = py::cast(buf);
    py::detail::keep_alive_impl(o, wrappedObj);

    // If we expose cuda array interface,
    if (auto iface = buf->cudaArrayInterface())
    {
        o.attr("__cuda_array_interface__") = *iface;
    }

    return o;
}

ExternalBuffer::ExternalBuffer(DLPackTensor &&dlTensor)
{
    if (!IsCudaAccessible(dlTensor->device.device_type))
    {
        throw std::runtime_error("Only CUDA memory buffers can be wrapped");
    }

    if (dlTensor->data != nullptr)
    {
        CheckValidCUDABuffer(dlTensor->data);
    }

    m_dlTensor = std::move(dlTensor);
}

Shape ExternalBuffer::shape() const
{
    Shape shape(m_dlTensor->ndim);
    for (size_t i = 0; i < shape.size(); ++i)
    {
        shape[i] = m_dlTensor->shape[i];
    }

    return shape;
}

py::tuple ExternalBuffer::strides() const
{
    py::tuple strides(m_dlTensor->ndim);

    for (size_t i = 0; i < strides.size(); ++i)
    {
        strides[i] = m_dlTensor->strides[i];
    }

    return strides;
}

py::object ExternalBuffer::dtype() const
{
    return ToDType(ToNVCVDataType(m_dlTensor->dtype));
}

void *ExternalBuffer::data() const
{
    return m_dlTensor->data;
}

bool ExternalBuffer::load(PyObject *o)
{
    if (!o)
    {
        return false;
    }

    py::object tmp = py::reinterpret_borrow<py::object>(o);

    if (hasattr(tmp, "__cuda_array_interface__"))
    {
        py::dict iface = tmp.attr("__cuda_array_interface__").cast<py::dict>();

        if (!iface.contains("shape") || !iface.contains("typestr") || !iface.contains("data")
            || !iface.contains("version"))
        {
            return false;
        }

        int version = iface["version"].cast<int>();
        if (version < 2)
        {
            return false;
        }

        DLPackTensor dlTensor;
        {
            DLManagedTensor dlManagedTensor = {};
            dlManagedTensor.deleter         = [](DLManagedTensor *self)
            {
                delete[] self->dl_tensor.shape;
                delete[] self->dl_tensor.strides;
            };
            dlTensor = DLPackTensor{std::move(dlManagedTensor)};
        }

        dlTensor->byte_offset = 0;

        // TODO: infer the device type from the memory buffer
        dlTensor->device.device_type = kDLCUDA;
        // TODO: infer the device from the memory buffer
        dlTensor->device.device_id = 0;

        // Convert data
        py::tuple tdata = iface["data"].cast<py::tuple>();
        void     *ptr   = reinterpret_cast<void *>(tdata[0].cast<long>());
        CheckValidCUDABuffer(ptr);
        dlTensor->data = ptr;

        // Convert DataType
        py::dtype dt = util::ToDType(iface["typestr"].cast<std::string>());
        if (std::optional<nvcv::DataType> dtype = ToNVCVDataType(dt))
        {
            dlTensor->dtype = ToDLDataType(*dtype);
        }

        // Convert ndim
        py::tuple shape = iface["shape"].cast<py::tuple>();
        dlTensor->ndim  = shape.size();

        // Convert shape
        dlTensor->shape = new int64_t[dlTensor->ndim];
        for (int i = 0; i < dlTensor->ndim; ++i)
        {
            dlTensor->shape[i] = shape[i].cast<long>();
        }

        // Convert strides
        dlTensor->strides = new int64_t[dlTensor->ndim];
        if (iface.contains("strides") && !iface["strides"].is_none())
        {
            py::tuple strides = iface["strides"].cast<py::tuple>();
            for (int i = 0; i < dlTensor->ndim; ++i)
            {
                dlTensor->strides[i] = strides[i].cast<long>();
                if (dlTensor->strides[i] % dt.itemsize() != 0)
                {
                    throw std::runtime_error("Stride must be a multiple of the element size in bytes");
                }
                dlTensor->strides[i] /= dt.itemsize();
            }
        }
        else
        {
            // If strides isn't defined, according to cuda array interface, we must
            // set them up for packed, row-major strides.
            dlTensor->strides[dlTensor->ndim - 1] = 1;
            for (int i = dlTensor->ndim - 1; i > 0; --i)
            {
                dlTensor->strides[i - 1] = dlTensor->strides[i] * dlTensor->shape[i];
            }
        }

        if (dlTensor->ndim >= 1)
        {
            m_wrappedObj              = tmp;
            m_cacheCudaArrayInterface = std::move(iface);
            m_dlTensor                = std::move(dlTensor);
            return true;
        }
    }
    else if (hasattr(tmp, "__dlpack__"))
    {
        // Quickly check if we support the device
        if (hasattr(tmp, "__dlpack_device__"))
        {
            py::tuple dlpackDevice = tmp.attr("__dlpack_device__")().cast<py::tuple>();
            auto      devType      = static_cast<DLDeviceType>(dlpackDevice[0].cast<int>());
            if (!IsCudaAccessible(devType))
            {
                throw std::runtime_error("Only CUDA-accessible memory buffers can be wrapped");
            }
        }

        py::capsule cap = tmp.attr("__dlpack__")(1).cast<py::capsule>();

        if (auto *tensor = static_cast<DLManagedTensor *>(cap.get_pointer()))
        {
            m_dlTensor = DLPackTensor{std::move(*tensor)};
            // signal that producer don't have to call tensor's deleter, we
            // (consumer will do it instead.
            cap.set_name("used_dltensor");
        }
        else
        {
            m_dlTensor = {};
        }
        return true;
    }

    return false;
}

std::optional<py::dict> ExternalBuffer::cudaArrayInterface() const
{
    if (!m_cacheCudaArrayInterface)
    {
        if (!IsCudaAccessible(m_dlTensor->device.device_type))
        {
            return std::nullopt;
        }

        nvcv::DataType dataType = ToNVCVDataType(m_dlTensor->dtype);

        NVCV_ASSERT(dataType.strideBytes() * 8 == m_dlTensor->dtype.bits);
        NVCV_ASSERT(m_dlTensor->dtype.bits % 8 == 0);
        int elemStrideBytes = m_dlTensor->dtype.bits / 8;

        py::object strides;

        if (m_dlTensor->strides == nullptr)
        {
            strides = py::none();
        }
        else
        {
            py::tuple vStrides(m_dlTensor->ndim);
            for (size_t i = 0; i < vStrides.size(); ++i)
            {
                vStrides[i] = m_dlTensor->strides[i] * elemStrideBytes;
            }
            strides = vStrides;
        }

        std::string format = ToFormatString(m_dlTensor->dtype);

        // clang-format off
        m_cacheCudaArrayInterface = py::dict
        {
            "shape"_a = this->shape(),
            "strides"_a = strides,
            "typestr"_a = format,
            "data"_a = py::make_tuple(reinterpret_cast<long>(m_dlTensor->data), false /* read/write */),
            "version"_a = 2
        };
    }

    return *m_cacheCudaArrayInterface;
}

py::capsule ExternalBuffer::dlpack(py::object stream) const
{
    struct ManagerCtx
    {
        DLManagedTensor tensor;
        std::shared_ptr<const ExternalBuffer> extBuffer;
    };

    auto ctx = std::make_unique<ManagerCtx>();

    // Set up tensor deleter to delete the ManagerCtx
    ctx->tensor.manager_ctx = ctx.get();
    ctx->tensor.deleter = [](DLManagedTensor *tensor)
    {
        auto *ctx = static_cast<ManagerCtx *>(tensor->manager_ctx);
        delete ctx;
    };

    // Copy tensor data
    ctx->tensor.dl_tensor = *m_dlTensor;

    // Manager context holds a reference to this External Buffer so that
    // GC doesn't delete this buffer while the dlpack tensor still refers to it.
    ctx->extBuffer = this->shared_from_this();

    // Creates the python capsule with the DLManagedTensor instance we're returning.
    py::capsule cap(&ctx->tensor, "dltensor", [](PyObject *ptr)
                    {
                        if(PyCapsule_IsValid(ptr, "dltensor"))
                        {
                            // If consumer didn't delete the tensor,
                            if(auto *dlTensor = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(ptr, "dltensor")))
                            {
                                // Delete the tensor.
                                if(dlTensor->deleter != nullptr)
                                {
                                    dlTensor->deleter(dlTensor);
                                }
                            }
                        }
                    });

    // Now that the capsule is created and the manager ctx was transfered to it,
    // we can release the unique_ptr.
    ctx.release();

    return cap;
}

py::tuple ExternalBuffer::dlpackDevice() const
{
    return py::make_tuple(py::int_(static_cast<int>(m_dlTensor->device.device_type)),
                          py::int_(static_cast<int>(m_dlTensor->device.device_id)));
}

const DLTensor &ExternalBuffer::dlTensor() const
{
    return *m_dlTensor;
}

void ExternalBuffer::Export(py::module &m)
{
    py::class_<ExternalBuffer, std::shared_ptr<ExternalBuffer>>(m, "ExternalBuffer", py::dynamic_attr())
        .def_property_readonly("shape", &ExternalBuffer::shape, "Get the shape of the buffer as an array")
        .def_property_readonly("strides", &ExternalBuffer::strides, "Get the strides of the buffer")
        .def_property_readonly("dtype", &ExternalBuffer::dtype, "Get the data type of the buffer")
        .def("__dlpack__", &ExternalBuffer::dlpack, "stream"_a=1, "Export the buffer as a DLPack tensor")
        .def("__dlpack_device__", &ExternalBuffer::dlpackDevice, "Get the device associated with the buffer");
}

} // namespace nv::vpi::python

namespace pybind11::detail {

namespace priv = nvcvpy::priv;

// Python -> C++
bool type_caster<priv::ExternalBuffer>::load(handle src, bool implicit_conv)
{
    PyTypeObject *srctype = Py_TYPE(src.ptr());
    const type_info *cuda_buffer_type = get_type_info(typeid(priv::ExternalBuffer));

    // src's type is ExternalBuffer?
    if(srctype == cuda_buffer_type->type)
    {
        // We know it's managed by a shared pointer (holder), let's use it
        value_and_holder vh = reinterpret_cast<instance *>(src.ptr())->get_value_and_holder();
        value = vh.template holder<std::shared_ptr<priv::ExternalBuffer>>();
        NVCV_ASSERT(value != nullptr);
        return true;
    }
    // If not, it could be an object that implements that __cuda_array_interface, let's try to
    // create a ExternalBuffer out of it.
    else
    {
        value.reset(new priv::ExternalBuffer);
        return value->load(src.ptr());
    }
}

} // namespace pybind11::detail
