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

#include "CAPI.hpp"

#include "Cache.hpp"
#include "DataType.hpp"
#include "Image.hpp"
#include "ImageBatch.hpp"
#include "Stream.hpp"
#include "Tensor.hpp"

#include <common/Assert.hpp>
#include <nvcv/python/CAPI.hpp>
#include <nvcv/python/Cache.hpp>
#include <nvcv/python/Container.hpp>
#include <pybind11/stl.h>

namespace nvcvpy::priv {

namespace {

template<class T>
std::shared_ptr<T> ToSharedObj(PyObject *obj)
{
    return py::reinterpret_borrow<py::object>(obj).cast<std::shared_ptr<T>>();
}

template<class T>
T ToObj(PyObject *obj)
{
    return py::reinterpret_borrow<py::object>(obj).cast<T>();
}

extern "C" PyObject *ImplDataType_ToPython(NVCVDataType p)
{
    py::object obj = py::cast(nvcv::DataType(p));
    return obj.ptr();
}

extern "C" NVCVDataType ImplDataType_FromPython(PyObject *obj)
{
    return ToObj<nvcv::DataType>(obj);
}

extern "C" PyObject *ImplImageFormat_ToPython(NVCVImageFormat p)
{
    py::object obj = py::cast(nvcv::ImageFormat(p));
    return obj.ptr();
}

extern "C" NVCVImageFormat ImplImageFormat_FromPython(PyObject *obj)
{
    return ToObj<nvcv::ImageFormat>(obj);
}

extern "C" NVCVTensorHandle ImplTensor_GetHandle(PyObject *obj)
{
    return ToSharedObj<Tensor>(obj)->impl().handle();
}

LockMode ToLockMode(PyObject *_mode)
{
    std::string s = ToObj<std::string>(_mode);
    if (s.empty())
    {
        return LockMode::LOCK_NONE;
    }
    else if (s == "r")
    {
        return LockMode::LOCK_READ;
    }
    else if (s == "w")
    {
        return LockMode::LOCK_WRITE;
    }
    else if (s == "rw")
    {
        return LockMode::LOCK_READWRITE;
    }
    else
    {
        throw std::runtime_error("Lock mode not understood: '" + s + "'");
    }
}

extern "C" void ImplResource_SubmitSync(PyObject *res, PyObject *stream, PyObject *lockMode)
{
    ToSharedObj<Resource>(res)->submitSync(*ToSharedObj<Stream>(stream), ToLockMode(lockMode));
}

extern "C" void ImplResource_SubmitSignal(PyObject *res, PyObject *stream, PyObject *lockMode)
{
    ToSharedObj<Resource>(res)->submitSignal(*ToSharedObj<Stream>(stream), ToLockMode(lockMode));
}

extern "C" void ImplStream_HoldResources(PyObject *stream, PyObject *resourceList)
{
    py::list resList = ToObj<py::list>(resourceList);

    LockResources resVector;

    for (py::handle h : resList)
    {
        py::tuple t = h.cast<py::tuple>();
        if (t.size() != 2)
        {
            throw std::runtime_error("ResourcePerMode tuple must have two elements");
        }

        auto lockMode = ToLockMode(t[0].ptr());
        auto res      = ToSharedObj<const Resource>(t[1].ptr());

        resVector.emplace(lockMode, res);
    }

    ToSharedObj<Stream>(stream)->holdResources(std::move(resVector));
}

extern "C" PyObject *ImplStream_GetCurrent()
{
    return py::cast(Stream::Current().shared_from_this()).ptr();
}

extern "C" cudaStream_t ImplStream_GetCudaHandle(PyObject *stream)
{
    return ToSharedObj<Stream>(stream)->handle();
}

extern "C" PyObject *ImplTensor_Create(int32_t ndim, const int64_t *shape, NVCVDataType dtype, NVCVTensorLayout layout)
{
    std::optional<nvcv::TensorLayout> cxxLayout;
    if (layout != NVCV_TENSOR_NONE)
    {
        cxxLayout = nvcv::TensorLayout(layout);
    }

    std::shared_ptr<Tensor> tensor
        = Tensor::Create(Shape(shape, shape + ndim), nvcv::DataType{dtype}, std::move(layout));

    return py::cast(std::move(tensor)).release().ptr();
}

extern "C" PyObject *ImplImageBatchVarShape_Create(int32_t capacity)
{
    std::shared_ptr<ImageBatchVarShape> varshape = ImageBatchVarShape::Create(capacity);
    return py::cast(std::move(varshape)).release().ptr();
}

extern "C" NVCVImageBatchHandle ImplImageBatchVarShape_GetHandle(PyObject *varshape)
{
    return ToSharedObj<ImageBatchVarShape>(varshape)->impl().handle();
}

extern "C" PyObject *ImplTensor_CreateForImageBatch(int32_t numImages, int32_t width, int32_t height,
                                                    NVCVImageFormat fmt)
{
    std::shared_ptr<Tensor> tensor = Tensor::CreateForImageBatch(numImages, {width, height}, nvcv::ImageFormat(fmt));
    return py::cast(std::move(tensor)).release().ptr();
}

extern "C" void ImplImageBatchVarShape_PushBack(PyObject *varshape, PyObject *image)
{
    auto pimage = ToSharedObj<Image>(image);
    return ToSharedObj<ImageBatchVarShape>(varshape)->pushBack(*pimage);
}

extern "C" void ImplImageBatchVarShape_PopBack(PyObject *varshape, int32_t cnt)
{
    return ToSharedObj<ImageBatchVarShape>(varshape)->popBack(cnt);
}

extern "C" void ImplImageBatchVarShape_Clear(PyObject *varshape)
{
    return ToSharedObj<ImageBatchVarShape>(varshape)->clear();
}

extern "C" void ImplCache_Add(ICacheItem *extItem)
{
    auto item = std::make_shared<ExternalCacheItem>(extItem->shared_from_this());
    Cache::Instance().add(*item);
}

extern "C" ICacheItem **ImplCache_Fetch(const IKey *pkey)
{
    NVCV_ASSERT(pkey != nullptr);

    std::vector<std::shared_ptr<priv::CacheItem>> vcont = Cache::Instance().fetch(*pkey);

    std::unique_ptr<nvcvpy::ICacheItem *[]> out(new ICacheItem *[vcont.size() + 1]);
    for (size_t i = 0; i < vcont.size(); ++i)
    {
        ExternalCacheItem *extItem = dynamic_cast<ExternalCacheItem *>(vcont[i].get());
        NVCV_ASSERT(extItem != nullptr);

        out[i] = extItem->obj.get();
    }
    out[vcont.size()] = nullptr; // end of list

    return out.release();
}

extern "C" PyObject *ImplImage_Create(int32_t width, int32_t height, NVCVImageFormat fmt)
{
    std::shared_ptr<Image> img = Image::Create({width, height}, nvcv::ImageFormat{fmt});
    return py::cast(std::move(img)).release().ptr();
}

extern "C" NVCVImageHandle ImplImage_GetHandle(PyObject *img)
{
    return ToSharedObj<Image>(img)->impl().handle();
}

extern "C" PyObject *ImplContainer_Create(nvcvpy::Container *pcont)
{
    NVCV_ASSERT(pcont != nullptr);
    auto cont = std::make_shared<ExternalContainer>(*pcont);

    py::object ocont = py::cast(cont);
    return ocont.release().ptr();
}

} // namespace

void ExportCAPI(py::module &m)
{
    static CAPI capi = {
        .DataType_ToPython            = &ImplDataType_ToPython,
        .DataType_FromPython          = &ImplDataType_FromPython,
        .ImageFormat_ToPython         = &ImplImageFormat_ToPython,
        .ImageFormat_FromPython       = &ImplImageFormat_FromPython,
        .Resource_SubmitSync          = &ImplResource_SubmitSync,
        .Resource_SubmitSignal        = &ImplResource_SubmitSignal,
        .Stream_HoldResources         = &ImplStream_HoldResources,
        .Stream_GetCurrent            = &ImplStream_GetCurrent,
        .Stream_GetCudaHandle         = &ImplStream_GetCudaHandle,
        .Tensor_GetHandle             = &ImplTensor_GetHandle,
        .Tensor_Create                = &ImplTensor_Create,
        .Tensor_CreateForImageBatch   = &ImplTensor_CreateForImageBatch,
        .ImageBatchVarShape_Create    = &ImplImageBatchVarShape_Create,
        .ImageBatchVarShape_GetHandle = &ImplImageBatchVarShape_GetHandle,
        .ImageBatchVarShape_PushBack  = &ImplImageBatchVarShape_PushBack,
        .ImageBatchVarShape_PopBack   = &ImplImageBatchVarShape_PopBack,
        .ImageBatchVarShape_Clear     = &ImplImageBatchVarShape_Clear,
        .Cache_Add                    = &ImplCache_Add,
        .Cache_Fetch                  = &ImplCache_Fetch,
        .Image_Create                 = &ImplImage_Create,
        .Image_GetHandle              = &ImplImage_GetHandle,
        .Container_Create             = &ImplContainer_Create,
    };

    m.add_object("_C_API", py::capsule(&capi, "nvcv._C_API"));
}

} // namespace nvcvpy::priv
