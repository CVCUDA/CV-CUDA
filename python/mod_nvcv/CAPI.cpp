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

#include "CAPI.hpp"

#include "Array.hpp"
#include "Cache.hpp"
#include "DataType.hpp"
#include "Image.hpp"
#include "ImageBatch.hpp"
#include "Stream.hpp"
#include "Tensor.hpp"
#include "TensorBatch.hpp"

#include <common/Assert.hpp>
#include <nvcv/python/CAPI.hpp>
#include <nvcv/python/Cache.hpp>
#include <nvcv/python/Container.hpp>
#include <pybind11/stl.h>

namespace nvcvpy::priv {

namespace {

// We need to catch any exceptions and set the appropriate PyError prior to crossing any C API boundry
#define CATCH_RETURN_DEFAULT(return_value, error_message)                                          \
    catch (const std::exception &e)                                                                \
    {                                                                                              \
        PyErr_SetString(PyExc_ValueError, (std::string(error_message) + ": " + e.what()).c_str()); \
        return return_value;                                                                       \
    }                                                                                              \
    catch (...)                                                                                    \
    {                                                                                              \
        PyErr_SetString(PyExc_ValueError, error_message);                                          \
        return return_value;                                                                       \
    }

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
    try
    {
        py::object obj = py::cast(nvcv::DataType(p));
        return obj.ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Casting PyObject from NVCVDataType failed")
}

extern "C" NVCVDataType ImplDataType_FromPython(PyObject *obj)
{
    try
    {
        return ToObj<nvcv::DataType>(obj);
    }
    CATCH_RETURN_DEFAULT(0, "Casting nvcv::DataType from PyObject failed")
}

extern "C" PyObject *ImplImageFormat_ToPython(NVCVImageFormat p)
{
    try
    {
        py::object obj = py::cast(nvcv::ImageFormat(p));
        return obj.ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Casting PyObject from NVCVImageFormat failed")
}

extern "C" NVCVImageFormat ImplImageFormat_FromPython(PyObject *obj)
{
    try
    {
        return ToObj<nvcv::ImageFormat>(obj);
    }
    CATCH_RETURN_DEFAULT(0, "Casting nvcv::ImageFormat from PyObject failed")
}

extern "C" NVCVTensorHandle ImplTensor_GetHandle(PyObject *obj)
{
    try
    {
        return ToSharedObj<Tensor>(obj)->impl().handle();
    }
    CATCH_RETURN_DEFAULT(0, "Getting Tensor handle from PyObject failed")
}

extern "C" NVCVArrayHandle ImplArray_GetHandle(PyObject *obj)
{
    try
    {
        return ToSharedObj<Array>(obj)->impl().handle();
    }
    CATCH_RETURN_DEFAULT(0, "Getting Array handle from PyObject failed")
}

LockMode ToLockMode(PyObject *_mode)
{
    std::string s = ToObj<std::string>(_mode);
    if (s.empty())
    {
        return LockMode::LOCK_MODE_NONE;
    }
    else if (s == "r")
    {
        return LockMode::LOCK_MODE_READ;
    }
    else if (s == "w")
    {
        return LockMode::LOCK_MODE_WRITE;
    }
    else if (s == "rw")
    {
        return LockMode::LOCK_MODE_READWRITE;
    }
    else
    {
        throw std::runtime_error("Lock mode not understood: '" + s + "'");
    }
}

extern "C" void ImplResource_SubmitSync(PyObject *res, PyObject *stream)
{
    try
    {
        ToSharedObj<Resource>(res)->submitSync(*ToSharedObj<Stream>(stream));
    }
    CATCH_RETURN_DEFAULT(, "Submit sync failed")
}

extern "C" void ImplStream_HoldResources(PyObject *stream, PyObject *resourceList)
{
    try
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
    CATCH_RETURN_DEFAULT(, "Hold resources failed")
}

extern "C" PyObject *ImplStream_GetCurrent()
{
    try
    {
        return py::cast(Stream::Current().shared_from_this()).ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Get current stream failed")
}

extern "C" cudaStream_t ImplStream_GetCudaHandle(PyObject *stream)
{
    try
    {
        return ToSharedObj<Stream>(stream)->handle();
    }
    CATCH_RETURN_DEFAULT(0, "Get cuda handle failed")
}

extern "C" PyObject *ImplTensor_Create(int32_t ndim, const int64_t *shape, NVCVDataType dtype, NVCVTensorLayout layout,
                                       int32_t rowalign)
{
    try
    {
        std::optional<nvcv::TensorLayout> cxxLayout;
        if (layout != NVCV_TENSOR_NONE)
        {
            cxxLayout = nvcv::TensorLayout(layout);
        }

        std::shared_ptr<Tensor> tensor = Tensor::Create(CreateShape(nvcv::TensorShape(shape, ndim, layout)),
                                                        nvcv::DataType{dtype}, std::move(layout), rowalign);
        return py::cast(std::move(tensor)).release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Tensor create failed")
}

extern "C" PyObject *ImplArray_Create(int64_t length, NVCVDataType dtype)
{
    try
    {
        std::shared_ptr<Array> array = Array::Create(length, nvcv::DataType{dtype});

        return py::cast(std::move(array)).release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Array create failed")
}

extern "C" PyObject *ImplImageBatchVarShape_Create(int32_t capacity)
{
    try
    {
        std::shared_ptr<ImageBatchVarShape> varshape = ImageBatchVarShape::Create(capacity);
        return py::cast(std::move(varshape)).release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "ImageBatchVarShape create failed")
}

extern "C" NVCVImageBatchHandle ImplImageBatchVarShape_GetHandle(PyObject *varshape)
{
    try
    {
        return ToSharedObj<ImageBatchVarShape>(varshape)->impl().handle();
    }
    CATCH_RETURN_DEFAULT(0, "ImageBatchVarShape get handle failed")
}

extern "C" PyObject *ImplTensor_CreateForImageBatch(int32_t numImages, int32_t width, int32_t height,
                                                    NVCVImageFormat fmt, int32_t rowalign)
{
    try
    {
        std::shared_ptr<Tensor> tensor
            = Tensor::CreateForImageBatch(numImages, {width, height}, nvcv::ImageFormat(fmt), rowalign);
        return py::cast(std::move(tensor)).release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Tensor for ImageBatch create failed")
}

extern "C" void ImplImageBatchVarShape_PushBack(PyObject *varshape, PyObject *image)
{
    try
    {
        auto pimage = ToSharedObj<Image>(image);
        return ToSharedObj<ImageBatchVarShape>(varshape)->pushBack(*pimage);
    }
    CATCH_RETURN_DEFAULT(, "ImageBatchVarShape push back failed")
}

extern "C" void ImplImageBatchVarShape_PopBack(PyObject *varshape, int32_t cnt)
{
    try
    {
        return ToSharedObj<ImageBatchVarShape>(varshape)->popBack(cnt);
    }
    CATCH_RETURN_DEFAULT(, "ImageBatchVarShape pop back failed")
}

extern "C" void ImplImageBatchVarShape_Clear(PyObject *varshape)
{
    try
    {
        return ToSharedObj<ImageBatchVarShape>(varshape)->clear();
    }
    CATCH_RETURN_DEFAULT(, "ImageBatchVarShape clear failed")
}

extern "C" PyObject *ImplTensorBatch_Create(int32_t capacity)
{
    try
    {
        std::shared_ptr<TensorBatch> tensorBatch = TensorBatch::Create(capacity);
        return py::cast(std::move(tensorBatch)).release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "TensorBatch create failed")
}

extern "C" NVCVTensorBatchHandle ImplTensorBatch_GetHandle(PyObject *tensorBatch)
{
    try
    {
        return ToSharedObj<TensorBatch>(tensorBatch)->impl().handle();
    }
    CATCH_RETURN_DEFAULT(0, "TensorBatch get handle failed")
}

extern "C" void ImplTensorBatch_PushBack(PyObject *tensorBatch, PyObject *tensor)
{
    try
    {
        auto ptensor = ToSharedObj<Tensor>(tensor);
        ToSharedObj<TensorBatch>(tensorBatch)->pushBack(*ptensor);
    }
    CATCH_RETURN_DEFAULT(, "TensorBatch push back failed")
}

extern "C" void ImplTensorBatch_PopBack(PyObject *tensorBatch, uint32_t cnt)
{
    try
    {
        ToSharedObj<TensorBatch>(tensorBatch)->popBack(cnt);
    }
    CATCH_RETURN_DEFAULT(, "TensorBatch pop back failed")
}

extern "C" void ImplTensorBatch_Clear(PyObject *tensorBatch)
{
    try
    {
        ToSharedObj<TensorBatch>(tensorBatch)->clear();
    }
    CATCH_RETURN_DEFAULT(, "TensorBatch clear failed")
}

extern "C" void ImplCache_Add(ICacheItem *extItem)
{
    try
    {
        auto item = std::make_shared<ExternalCacheItem>(extItem->shared_from_this());
        Cache::Instance().add(*item);
    }
    CATCH_RETURN_DEFAULT(, "Cache add item failed")
}

extern "C" ICacheItem **ImplCache_Fetch(const IKey *pkey)
{
    try
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
    CATCH_RETURN_DEFAULT(nullptr, "Cache add fetch failed")
}

extern "C" PyObject *ImplImage_Create(int32_t width, int32_t height, NVCVImageFormat fmt, int32_t rowAlign)
{
    try
    {
        std::shared_ptr<Image> img = Image::Create({width, height}, nvcv::ImageFormat{fmt}, rowAlign);
        return py::cast(std::move(img)).release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Image create failed")
}

extern "C" NVCVImageHandle ImplImage_GetHandle(PyObject *img)
{
    try
    {
        return ToSharedObj<Image>(img)->impl().handle();
    }
    CATCH_RETURN_DEFAULT(0, "Image get handle failed")
}

extern "C" PyObject *ImplContainer_Create(nvcvpy::Container *pcont)
{
    try
    {
        NVCV_ASSERT(pcont != nullptr);
        auto cont = std::make_shared<ExternalContainer>(*pcont);

        py::object ocont = py::cast(cont);
        return ocont.release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Container create failed")
}

extern "C" void ImplCache_RemoveAllNotInUseMatching(const IKey *pkey)
{
    try
    {
        NVCV_ASSERT(pkey != nullptr);

        Cache::Instance().removeAllNotInUseMatching(*pkey);
    }
    CATCH_RETURN_DEFAULT(, "Cache cleanup failed when removing all not in use matching")
}

} // namespace

// Note these functions will set a PyError if an exception is thrown, this must be then checked by calling
// CheckCAPIError() before returning to Python.
void ExportCAPI(py::module &m)
{
    static CAPI capi = {
        .DataType_ToPython               = &ImplDataType_ToPython,
        .DataType_FromPython             = &ImplDataType_FromPython,
        .ImageFormat_ToPython            = &ImplImageFormat_ToPython,
        .ImageFormat_FromPython          = &ImplImageFormat_FromPython,
        .Resource_SubmitSync             = &ImplResource_SubmitSync,
        .Stream_HoldResources            = &ImplStream_HoldResources,
        .Stream_GetCurrent               = &ImplStream_GetCurrent,
        .Stream_GetCudaHandle            = &ImplStream_GetCudaHandle,
        .Tensor_GetHandle                = &ImplTensor_GetHandle,
        .Tensor_Create                   = &ImplTensor_Create,
        .Tensor_CreateForImageBatch      = &ImplTensor_CreateForImageBatch,
        .Array_GetHandle                 = &ImplArray_GetHandle,
        .Array_Create                    = &ImplArray_Create,
        .ImageBatchVarShape_Create       = &ImplImageBatchVarShape_Create,
        .ImageBatchVarShape_GetHandle    = &ImplImageBatchVarShape_GetHandle,
        .ImageBatchVarShape_PushBack     = &ImplImageBatchVarShape_PushBack,
        .ImageBatchVarShape_PopBack      = &ImplImageBatchVarShape_PopBack,
        .ImageBatchVarShape_Clear        = &ImplImageBatchVarShape_Clear,
        .Cache_Add                       = &ImplCache_Add,
        .Cache_Fetch                     = &ImplCache_Fetch,
        .Image_Create                    = &ImplImage_Create,
        .Image_GetHandle                 = &ImplImage_GetHandle,
        .Container_Create                = &ImplContainer_Create,
        .Cache_RemoveAllNotInUseMatching = &ImplCache_RemoveAllNotInUseMatching,
        .TensorBatch_Create              = &ImplTensorBatch_Create,
        .TensorBatch_GetHandle           = &ImplTensorBatch_GetHandle,
        .TensorBatch_PushBack            = &ImplTensorBatch_PushBack,
        .TensorBatch_PopBack             = &ImplTensorBatch_PopBack,
        .TensorBatch_Clear               = &ImplTensorBatch_Clear,
    };

    m.add_object("_C_API", py::capsule(&capi, "nvcv._C_API"));
}

} // namespace nvcvpy::priv
