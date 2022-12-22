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

#include "CudaBuffer.hpp"

#include <common/Assert.hpp>
#include <common/PyUtil.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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

static py::buffer_info CopyBuffer(const py::buffer_info &src, bool writable)
{
    void       *newData;
    int         len = src.shape[0] * src.strides[0];
    cudaError_t err = cudaMalloc(&newData, len);
    if (err != cudaSuccess)
    {
        std::ostringstream ss;
        ss << "Error allocating " << len << " bytes of cuda memory: " << cudaGetErrorName(err) << " - "
           << cudaGetErrorString(err);
        throw std::runtime_error(ss.str());
    }

    try
    {
        err = cudaMemcpy2D(newData, src.strides[0], src.ptr, src.strides[0],
                           (src.shape.size() >= 2 ? src.shape[1] : 1) * src.itemsize, src.shape[0],
                           cudaMemcpyDeviceToDevice);

        if (err != cudaSuccess)
        {
            std::ostringstream ss;
            ss << "Error copying cuda buffer: " << cudaGetErrorName(err) << " - " << cudaGetErrorString(err);
            throw std::runtime_error(ss.str());
        }

        return py::buffer_info(newData, src.itemsize, src.format, src.shape.size(), src.shape, src.strides, !writable);
    }
    catch (...)
    {
        cudaFree(newData);
        throw;
    }
}

CudaBuffer::CudaBuffer()
    : CudaBuffer(py::buffer_info{})
{
}

CudaBuffer::CudaBuffer(const py::buffer_info &info, bool copy, py::object wrappedObj)
    : m_wrappedObj(wrappedObj)
{
    if (info.ptr != nullptr)
    {
        CheckValidCUDABuffer(info.ptr);
    }

    py::buffer_info        tmp;
    const py::buffer_info *pinfo;
    if (copy && info.ptr != nullptr)
    {
        tmp    = CopyBuffer(info, true);
        pinfo  = &tmp;
        m_owns = true;
    }
    else
    {
        pinfo  = &info;
        m_owns = false;
    }

    try
    {
        // clang-format off
        m_cudaArrayInterface = py::dict
        {
            "shape"_a = pinfo->shape,
            "strides"_a = pinfo->strides,
            "typestr"_a = pinfo->format,
            "data"_a = py::make_tuple(reinterpret_cast<long>(pinfo->ptr), pinfo->readonly),
            "version"_a = 2
        };
    }
    catch(...)
    {
        if(copy)
        {
            cudaFree(tmp.ptr);
        }
        throw;
    }
}

CudaBuffer::~CudaBuffer()
{
    if(m_owns)
    {
        void *ptr = this->data();
        cudaFree(ptr);
    }
}

py::object CudaBuffer::shape() const
{
    return m_cudaArrayInterface["shape"];
}

py::object CudaBuffer::dtype() const
{
    return util::ToDType(this->request());
}

void *CudaBuffer::data() const
{
    if (m_cudaArrayInterface)
    {
        py::tuple tdata = m_cudaArrayInterface["data"].cast<py::tuple>();
        return reinterpret_cast<void *>(tdata[0].cast<long>());
    }
    else
    {
        return nullptr;
    }
}

bool CudaBuffer::load(PyObject *o)
{
    if (!o)
    {
        return false;
    }

    py::object tmp = py::reinterpret_borrow<py::object>(o);

    if (hasattr(tmp, "__cuda_array_interface__"))
    {
        py::dict iface = tmp.attr("__cuda_array_interface__").cast<py::dict>();

        if (!iface.contains("shape") || !iface.contains("typestr") || !iface.contains("data") || !iface.contains("version"))
        {
            return false;
        }

        int version = iface["version"].cast<int>();
        if (version < 2)
        {
            return false;
        }

        py::tuple tdata = iface["data"].cast<py::tuple>();
        void *ptr = reinterpret_cast<void *>(tdata[0].cast<long>());

        CheckValidCUDABuffer(ptr);

        std::vector<long> vshape;
        py::tuple shape = iface["shape"].cast<py::tuple>();
        for (auto &o : shape)
        {
            vshape.push_back(o.cast<long>());
        }

        if(vshape.size() >= 1)
        {
            m_wrappedObj = tmp; // hold the reference to the wrapped object
            m_cudaArrayInterface = std::move(iface);
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        return false;
    }
}

py::dict CudaBuffer::cuda_interface() const
{
    return m_cudaArrayInterface;
}

py::buffer_info CudaBuffer::request(bool writable) const
{
    void *ptr = this->data();

    std::string typestr = m_cudaArrayInterface["typestr"].cast<std::string>();
    int itemsize        = py::dtype(typestr).itemsize();

    std::vector<long> vshape;
    py::tuple shape = m_cudaArrayInterface["shape"].cast<py::tuple>();
    for (auto &o : shape)
    {
        vshape.push_back(o.cast<long>());
    }

    std::vector<long> vstrides;
    if (m_cudaArrayInterface.contains("strides"))
    {
        py::object strides = m_cudaArrayInterface["strides"];
        if (!strides.is(py::none()))
        {
            strides = strides.cast<py::tuple>();
            for (auto &o : strides)
            {
                vstrides.push_back(o.cast<long>());
            }
        }
    }

    if (vstrides.empty())
    {
        vstrides = py::detail::c_strides(vshape, itemsize);
    }

    return py::buffer_info(ptr, itemsize, typestr, vshape.size(), vshape, vstrides, !writable);
}

void CudaBuffer::Export(py::module &m)
{
    py::class_<CudaBuffer, std::shared_ptr<CudaBuffer>>(m, "Buffer")
        .def_property_readonly("__cuda_array_interface__", &CudaBuffer::cuda_interface)
        .def_property_readonly("shape", &CudaBuffer::shape)
        .def_property_readonly("dtype", &CudaBuffer::dtype);
}

} // namespace nv::vpi::python

namespace pybind11::detail {

namespace priv = nvcvpy::priv;

// Python -> C++
bool type_caster<priv::CudaBuffer>::load(handle src, bool implicit_conv)
{
    PyTypeObject *srctype = Py_TYPE(src.ptr());
    const type_info *cuda_buffer_type = get_type_info(typeid(priv::CudaBuffer));

    // src's type is CudaBuffer?
    if(srctype == cuda_buffer_type->type)
    {
        // We know it's managed by a shared pointer (holder), let's use it
        value_and_holder vh = reinterpret_cast<instance *>(src.ptr())->get_value_and_holder();
        value = vh.template holder<std::shared_ptr<priv::CudaBuffer>>();
        NVCV_ASSERT(value != nullptr);
        src.inc_ref();
        return true;
    }
    // If not, it could be an object that implements that __cuda_array_interface, let's try to
    // create a CudaBuffer out of it.
    else
    {
        value.reset(new priv::CudaBuffer);
        return value->load(src.ptr());
    }
}

} // namespace pybind11::detail
