/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../operators/NativeResourceGuard.hpp"
#include "Array.hpp"
#include "Cache.hpp"
#include "DataType.hpp"
#include "Image.hpp"
#include "ImageBatch.hpp"
#include "PythonObjectAdapters.hpp"
#include "Stream.hpp"
#include "Tensor.hpp"
#include "TensorBatch.hpp"

#include <common/Assert.hpp>
#include <common/Binding.hpp>
#include <nvcv/python/CAPI.hpp>
#include <nvcv/python/Cache.hpp>
#include <nvcv/python/Container.hpp>

#include <memory>
#include <stdexcept>

namespace nvcvpy::priv {

namespace {

class CAPIResourceError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class ExternalCacheProbeKey final : public IKey
{
private:
    size_t doGetHash() const override
    {
        return 0x435643554441ULL;
    }

    bool doIsCompatible(const IKey &) const override
    {
        return true;
    }
};

class ExternalCacheProbeItem final : public ICacheItem
{
public:
    const IKey &key() const override
    {
        return m_key;
    }

    py::object container() const override
    {
        return py::none();
    }

private:
    ExternalCacheProbeKey m_key;
};

class ExternalResourceProbeKey final : public nvcvpy::IKey
{
public:
    ExternalResourceProbeKey() = default;

private:
    size_t doGetHash() const override
    {
        return 0;
    }

    bool doIsCompatible(const nvcvpy::IKey &) const override
    {
        return true;
    }
};

class ExternalResourceProbe final // NOSONAR: depth is imposed by the nanobind facade hierarchy it must extend.
    : public nvcvpy::Container
{
public:
    const nvcvpy::IKey &key() const override
    {
        return m_key;
    }

    py::object container() const override
    {
        return py::none();
    }

private:
    ExternalResourceProbeKey m_key;
};

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

// --- Test-only failure injection ---------------------------------------------
//
// Deterministic failure toggles for the ResourceGuard lifetime regressions in
// tests/cvcuda/python/test_resourceguard.py, armed from Python through the
// private cvcuda._test submodule (see ExportCAPITestHooks). An armed toggle
// makes the corresponding C API callback fail through its production
// CATCH_RETURN_DEFAULT path, so the error the guard observes is shaped exactly
// like a real failure. Plain bools: both the toggles and every injected
// callback run under the GIL.

template<class T>
std::shared_ptr<T> ToSharedObj(PyObject *obj)
{
    return py::cast<std::shared_ptr<T>>(py::reinterpret_borrow<py::object>(obj));
}

template<class T>
T ToObj(PyObject *obj)
{
    return py::cast<T>(py::reinterpret_borrow<py::object>(obj));
}

PyObject *ImplDataType_ToPython(NVCVDataType p)
{
    try
    {
        py::object obj = py::cast(nvcv::DataType(p));
        return obj.release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Casting PyObject from NVCVDataType failed")
}

NVCVDataType ImplDataType_FromPython(PyObject *obj)
{
    try
    {
        return static_cast<NVCVDataType>(ToObj<nvcv::DataType>(obj));
    }
    CATCH_RETURN_DEFAULT(0, "Casting nvcv::DataType from PyObject failed")
}

PyTypeObject *ImplFacade_GetType(nvcvpy::FacadeType type) noexcept
{
    switch (type)
    {
    case nvcvpy::FacadeType::TENSOR:
        return reinterpret_cast<PyTypeObject *>(py::type<Tensor>().ptr());
    case nvcvpy::FacadeType::IMAGE:
        return reinterpret_cast<PyTypeObject *>(py::type<Image>().ptr());
    case nvcvpy::FacadeType::ARRAY:
        return reinterpret_cast<PyTypeObject *>(py::type<Array>().ptr());
    case nvcvpy::FacadeType::TENSOR_BATCH:
        return reinterpret_cast<PyTypeObject *>(py::type<TensorBatch>().ptr());
    case nvcvpy::FacadeType::IMAGE_BATCH_VAR_SHAPE:
        return reinterpret_cast<PyTypeObject *>(py::type<ImageBatchVarShape>().ptr());
    case nvcvpy::FacadeType::STREAM:
        return reinterpret_cast<PyTypeObject *>(py::type<Stream>().ptr());
    }
    return nullptr;
}

PyObject *ImplImageFormat_ToPython(NVCVImageFormat p)
{
    try
    {
        py::object obj = py::cast(nvcv::ImageFormat(p));
        return obj.release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Casting PyObject from NVCVImageFormat failed")
}

NVCVImageFormat ImplImageFormat_FromPython(PyObject *obj)
{
    try
    {
        return static_cast<NVCVImageFormat>(ToObj<nvcv::ImageFormat>(obj));
    }
    CATCH_RETURN_DEFAULT(0, "Casting nvcv::ImageFormat from PyObject failed")
}

NVCVTensorHandle ImplTensor_GetHandle(PyObject *obj)
{
    try
    {
        return ToSharedObj<Tensor>(obj)->impl().handle();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Getting Tensor handle from PyObject failed")
}

NVCVArrayHandle ImplArray_GetHandle(PyObject *obj)
{
    try
    {
        return ToSharedObj<Array>(obj)->impl().handle();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Getting Array handle from PyObject failed")
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
        throw CAPIResourceError("Lock mode not understood: '" + s + "'");
    }
}

void ImplResource_SubmitSync(PyObject *res, PyObject *stream)
{
    try
    {
        ToSharedObj<Resource>(res)->submitSync(*ToSharedObj<Stream>(stream));
    }
    CATCH_RETURN_DEFAULT(, "Submit sync failed")
}

void DoStream_HoldResources(PyObject *stream, PyObject *resourceList, int submissionDevice)
{
    ThrowIfResourceGuardFailureInjected(GetResourceGuardFailureInjection().streamHoldResources,
                                        "injected Stream_HoldResources failure");

    auto pyStream = ToSharedObj<Stream>(stream);

    py::list resList = ToObj<py::list>(resourceList);

    ResourceBag    resVector;
    PythonOwnerBag pythonOwners;
    resVector.reserve(resList.size());
    pythonOwners.reserve(resList.size());

    const PyObject *lastModePtr = nullptr;
    LockMode        lastMode    = LockMode::LOCK_MODE_NONE;

    for (py::handle h : resList)
    {
        py::tuple t = py::cast<py::tuple>(h);
        if (t.size() != 2)
        {
            throw CAPIResourceError("ResourcePerMode tuple must have two elements");
        }

        PyObject *modePtr = t[0].ptr();
        LockMode  lockMode;
        if (modePtr == lastModePtr)
        {
            lockMode = lastMode;
        }
        else
        {
            lockMode    = ToLockMode(modePtr);
            lastModePtr = modePtr;
            lastMode    = lockMode;
        }

        (void)lockMode;
        PyObject *resource = t[1].ptr();
        resVector.emplace_back(ToSharedObj<Resource>(resource));
        pythonOwners.emplace_back(py::reinterpret_borrow<py::object>(resource));
    }

    pyStream->holdResources(std::move(resVector), std::move(pythonOwners), submissionDevice);
}

void ImplStream_HoldResources(PyObject *stream, PyObject *resourceList)
{
    try
    {
        auto pyStream = ToSharedObj<Stream>(stream);
        DoStream_HoldResources(stream, resourceList, pyStream->submissionDeviceId());
    }
    CATCH_RETURN_DEFAULT(, "Hold resources failed")
}

void ImplResources_SubmitSyncOnlyRaw(PyObject *stream, PyObject *const *resources, size_t count)
{
    try
    {
        ThrowIfResourceGuardFailureInjected(GetResourceGuardFailureInjection().resourcesSubmitSyncOnly,
                                            "injected Resources_SubmitSyncOnly failure");
        auto    pyStream         = ToSharedObj<Stream>(stream);
        int     submissionDevice = pyStream->submissionDeviceId();
        Stream &streamRef        = *pyStream;
        pyStream->ensureNotCapturing(submissionDevice);
        for (size_t i = 0; i < count; ++i)
        {
            ToSharedObj<Resource>(resources[i])->submitSync(streamRef, submissionDevice);
        }
    }
    CATCH_RETURN_DEFAULT(, "Submit sync only failed")
}

void ImplStream_HoldResourcesRaw(PyObject *stream, PyObject *const *resources, size_t count)
{
    try
    {
        ThrowIfResourceGuardFailureInjected(GetResourceGuardFailureInjection().streamHoldResources,
                                            "injected Stream_HoldResources failure");
        auto           pyStream = ToSharedObj<Stream>(stream);
        ResourceBag    owners;
        PythonOwnerBag pythonOwners;
        owners.reserve(count);
        pythonOwners.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            owners.emplace_back(ToSharedObj<Resource>(resources[i]));
            pythonOwners.emplace_back(py::reinterpret_borrow<py::object>(resources[i]));
        }
        pyStream->holdResources(std::move(owners), std::move(pythonOwners), pyStream->submissionDeviceId());
    }
    CATCH_RETURN_DEFAULT(, "Hold resources failed")
}

// Batched submit-sync only (no holdResources). Used by ResourceGuard::run()
// to insert all wait_events on the consumer stream BEFORE the op's kernel is
// queued. The hold half runs at guard destruction via Stream_HoldResources.
//
// Splitting the sync from the hold matters because cudaStreamWaitEvent must
// be enqueued before the kernel it's meant to gate — the previous combined
// Resources_SyncAndHold (called at guard destruction) ran AFTER the kernel
// and was therefore a no-op for the kernel it was supposed to protect.
void DoResources_SubmitSyncOnly(PyObject *stream, PyObject *resourceList, int submissionDevice)
{
    ThrowIfResourceGuardFailureInjected(GetResourceGuardFailureInjection().resourcesSubmitSyncOnly,
                                        "injected Resources_SubmitSyncOnly failure");

    py::list resList   = ToObj<py::list>(resourceList);
    auto     pyStream  = ToSharedObj<Stream>(stream);
    Stream  &streamRef = *pyStream;

    pyStream->ensureNotCapturing(submissionDevice);

    for (py::handle h : resList)
    {
        py::tuple t = py::cast<py::tuple>(h);
        if (t.size() != 2)
        {
            throw CAPIResourceError("ResourcePerMode tuple must have two elements");
        }

        ToSharedObj<Resource>(t[1].ptr())->submitSync(streamRef, submissionDevice);
    }
}

void ImplResources_SubmitSyncOnly(PyObject *stream, PyObject *resourceList)
{
    try
    {
        auto pyStream = ToSharedObj<Stream>(stream);
        DoResources_SubmitSyncOnly(stream, resourceList, pyStream->submissionDeviceId());
    }
    CATCH_RETURN_DEFAULT(, "Submit sync only failed")
}

// Batched equivalent of "for r in list: Resource_SubmitSync(r, stream)" followed
// by Stream_HoldResources, executed entirely in C++. Saves N nanobind/C-ABI
// boundary crossings on ops that track many resources.
//
// NOTE: this combined call is retained for backward compatibility with
// out-of-tree consumers that haven't migrated to the run()-based pattern.
// New code should use Resources_SubmitSyncOnly + Stream_HoldResources via
// ResourceGuard::run().
void DoResources_SyncAndHold(PyObject *stream, PyObject *resourceList, int submissionDevice)
{
    ThrowIfResourceGuardFailureInjected(GetResourceGuardFailureInjection().resourcesSyncAndHold,
                                        "injected Resources_SyncAndHold failure");

    py::list resList   = ToObj<py::list>(resourceList);
    auto     pyStream  = ToSharedObj<Stream>(stream);
    Stream  &streamRef = *pyStream;

    pyStream->ensureNotCapturing(submissionDevice);

    ResourceBag    resVector;
    PythonOwnerBag pythonOwners;
    resVector.reserve(resList.size());
    pythonOwners.reserve(resList.size());

    // Within a single commit() call, contiguous items typically share the
    // same py::str instance for their lock mode (one .add(mode, {...})
    // call -> N items with the same mode). A one-slot last-seen cache
    // skips the string parse on those.
    const PyObject *lastModePtr = nullptr;
    LockMode        lastMode    = LockMode::LOCK_MODE_NONE;

    for (py::handle h : resList)
    {
        py::tuple t = py::cast<py::tuple>(h);
        if (t.size() != 2)
        {
            throw CAPIResourceError("ResourcePerMode tuple must have two elements");
        }

        PyObject *modePtr = t[0].ptr();
        LockMode  lockMode;
        if (modePtr == lastModePtr)
        {
            lockMode = lastMode;
        }
        else
        {
            lockMode    = ToLockMode(modePtr);
            lastModePtr = modePtr;
            lastMode    = lockMode;
        }

        PyObject *resource = t[1].ptr();
        auto      res      = ToSharedObj<Resource>(resource);
        res->submitSync(streamRef, submissionDevice);
        (void)lockMode;
        resVector.emplace_back(std::move(res));
        pythonOwners.emplace_back(py::reinterpret_borrow<py::object>(resource));
    }

    pyStream->holdResources(std::move(resVector), std::move(pythonOwners), submissionDevice);
}

void ImplResources_SyncAndHold(PyObject *stream, PyObject *resourceList)
{
    try
    {
        auto pyStream = ToSharedObj<Stream>(stream);
        DoResources_SyncAndHold(stream, resourceList, pyStream->submissionDeviceId());
    }
    CATCH_RETURN_DEFAULT(, "Sync and hold resources failed")
}

PyObject *ImplStream_GetCurrent()
{
    try
    {
        return py::cast(Stream::Current().sharedStream()).ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Get current stream failed")
}

cudaStream_t ImplStream_GetCudaHandle(PyObject *stream)
{
    try
    {
        return ToSharedObj<Stream>(stream)->handle();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Get cuda handle failed")
}

PyObject *ImplTensor_Create(int32_t ndim, const int64_t *shape, NVCVDataType dtype, NVCVTensorLayout layout,
                            int32_t rowalign)
{
    try
    {
        std::optional<nvcv::TensorLayout> cxxLayout;
        if (layout != NVCV_TENSOR_NONE)
        {
            cxxLayout = nvcv::TensorLayout(layout);
        }
        const nvcv::TensorLayout tensorLayout = cxxLayout.value_or(nvcv::TENSOR_NONE);

        std::shared_ptr<Tensor> tensor = Tensor::Create(CreateShape(nvcv::TensorShape(shape, ndim, tensorLayout)),
                                                        nvcv::DataType{dtype}, std::move(cxxLayout), rowalign);
        return py::cast(std::move(tensor)).release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Tensor create failed")
}

PyObject *ImplArray_Create(int64_t length, NVCVDataType dtype)
{
    try
    {
        std::shared_ptr<Array> array = Array::Create(length, nvcv::DataType{dtype});

        return py::cast(std::move(array)).release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Array create failed")
}

PyObject *ImplImageBatchVarShape_Create(int32_t capacity)
{
    try
    {
        std::shared_ptr<ImageBatchVarShape> varshape = ImageBatchVarShape::Create(capacity);
        return py::cast(std::move(varshape)).release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "ImageBatchVarShape create failed")
}

NVCVImageBatchHandle ImplImageBatchVarShape_GetHandle(PyObject *varshape)
{
    try
    {
        return ToSharedObj<ImageBatchVarShape>(varshape)->impl().handle();
    }
    CATCH_RETURN_DEFAULT(nullptr, "ImageBatchVarShape get handle failed")
}

PyObject *ImplTensor_CreateForImageBatch(int32_t numImages, int32_t width, int32_t height, NVCVImageFormat fmt,
                                         int32_t rowalign)
{
    try
    {
        std::shared_ptr<Tensor> tensor
            = Tensor::CreateForImageBatch(numImages, {width, height}, nvcv::ImageFormat(fmt), rowalign);
        return py::cast(std::move(tensor)).release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Tensor for ImageBatch create failed")
}

void ImplImageBatchVarShape_PushBack(PyObject *varshape, PyObject *image)
{
    try
    {
        auto pimage = ToSharedObj<Image>(image);
        return ToSharedObj<ImageBatchVarShape>(varshape)->pushBack(*pimage);
    }
    CATCH_RETURN_DEFAULT(, "ImageBatchVarShape push back failed")
}

void ImplImageBatchVarShape_PopBack(PyObject *varshape, int32_t cnt)
{
    try
    {
        return ToSharedObj<ImageBatchVarShape>(varshape)->popBack(cnt);
    }
    CATCH_RETURN_DEFAULT(, "ImageBatchVarShape pop back failed")
}

void ImplImageBatchVarShape_Clear(PyObject *varshape)
{
    try
    {
        return ToSharedObj<ImageBatchVarShape>(varshape)->clear();
    }
    CATCH_RETURN_DEFAULT(, "ImageBatchVarShape clear failed")
}

PyObject *ImplTensorBatch_Create(int32_t capacity)
{
    try
    {
        std::shared_ptr<TensorBatch> tensorBatch = TensorBatch::Create(capacity);
        return py::cast(std::move(tensorBatch)).release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "TensorBatch create failed")
}

NVCVTensorBatchHandle ImplTensorBatch_GetHandle(PyObject *tensorBatch)
{
    try
    {
        return ToSharedObj<TensorBatch>(tensorBatch)->impl().handle();
    }
    CATCH_RETURN_DEFAULT(nullptr, "TensorBatch get handle failed")
}

void ImplTensorBatch_PushBack(PyObject *tensorBatch, PyObject *tensor)
{
    try
    {
        auto ptensor = ToSharedObj<Tensor>(tensor);
        ToSharedObj<TensorBatch>(tensorBatch)->pushBack(*ptensor);
    }
    CATCH_RETURN_DEFAULT(, "TensorBatch push back failed")
}

void ImplTensorBatch_PopBack(PyObject *tensorBatch, uint32_t cnt)
{
    try
    {
        ToSharedObj<TensorBatch>(tensorBatch)->popBack(cnt);
    }
    CATCH_RETURN_DEFAULT(, "TensorBatch pop back failed")
}

void ImplTensorBatch_Clear(PyObject *tensorBatch)
{
    try
    {
        ToSharedObj<TensorBatch>(tensorBatch)->clear();
    }
    CATCH_RETURN_DEFAULT(, "TensorBatch clear failed")
}

void ImplCache_Add(ICacheItem *extItem)
{
    try
    {
        auto item = std::make_shared<ExternalCacheItem>(extItem->shared_from_this());
        Cache::Instance().add(*item);
    }
    CATCH_RETURN_DEFAULT(, "Cache add item failed")
}

ICacheItem **ImplCache_Fetch(const IKey *pkey)
{
    try
    {
        NVCV_ASSERT(pkey != nullptr);

        std::vector<std::shared_ptr<priv::CacheItem>> vcont = Cache::Instance().fetch(*pkey);

        auto out = std::make_unique<nvcvpy::ICacheItem *[]>(vcont.size() + 1);
        for (size_t i = 0; i < vcont.size(); ++i)
        {
            const auto *extItem = dynamic_cast<const ExternalCacheItem *>(vcont[i].get());
            NVCV_ASSERT(extItem != nullptr);

            out[i] = extItem->obj();
        }
        out[vcont.size()] = nullptr; // end of list

        return out.release();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Cache add fetch failed")
}

PyObject *ImplImage_Create(int32_t width, int32_t height, NVCVImageFormat fmt, int32_t rowAlign)
{
    try
    {
        std::shared_ptr<Image> img = Image::Create({width, height}, nvcv::ImageFormat{fmt}, rowAlign);
        return py::cast(std::move(img)).release().ptr();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Image create failed")
}

NVCVImageHandle ImplImage_GetHandle(PyObject *img)
{
    try
    {
        return ToSharedObj<Image>(img)->impl().handle();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Image get handle failed")
}

PyObject *ImplContainer_Create(nvcvpy::Container *pcont)
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

void ImplCache_RemoveAllNotInUseMatching(const IKey *pkey)
{
    try
    {
        NVCV_ASSERT(pkey != nullptr);

        Cache::Instance().removeAllNotInUseMatching(*pkey);
    }
    CATCH_RETURN_DEFAULT(, "Cache cleanup failed when removing all not in use matching")
}

} // namespace

ResourceGuardFailureInjection &GetResourceGuardFailureInjection()
{
    static ResourceGuardFailureInjection injection;
    return injection;
}

void ThrowIfResourceGuardFailureInjected(bool armed, const char *what)
{
    if (armed)
    {
        throw std::invalid_argument(what);
    }
}

#ifdef NB_FREE_THREADED
namespace {
struct CapiSubmission
{
    std::vector<BatchMutex *>                      mutexes;
    std::unique_ptr<BatchLockScope>                batches;
    std::unique_ptr<cvcudapy::NativeResourceGuard> guard;
};

void *BeginSubmission(PyObject *stream, PyObject *const *resources, const uint8_t *modes, size_t count)
{
    try
    {
        auto                    submission = std::make_unique<CapiSubmission>();
        // The caller retains these Python arguments until Begin returns.
        std::vector<Resource *> borrowedResources;
        for (size_t i = 0; i < count; ++i)
        {
            auto &resource = ToObj<Resource &>(resources[i]);
            if (auto *mutex = resource.submissionBatchMutex())
                submission->mutexes.push_back(mutex);
            borrowedResources.push_back(&resource);
        }
        if (!submission->mutexes.empty())
            submission->batches
                = std::make_unique<BatchLockScope>(submission->mutexes.data(), submission->mutexes.size());
        submission->guard = std::make_unique<cvcudapy::NativeResourceGuard>(ToObj<Stream &>(stream));
        for (size_t i = 0; i < count; ++i)
            submission->guard->add(static_cast<LockMode>(modes[i]), {*borrowedResources[i]});
        submission->guard->begin();
        return submission.release();
    }
    CATCH_RETURN_DEFAULT(nullptr, "Begin submission failed")
}

void EndSubmission(void *token)
{
    try
    {
        std::unique_ptr<CapiSubmission> submission(static_cast<CapiSubmission *>(token));
        submission->guard->finish();
    }
    CATCH_RETURN_DEFAULT(, "End submission failed")
}
} // namespace
#endif

// Note these functions will set a PyError if an exception is thrown, this must be then checked by calling
// CheckCAPIError() before returning to Python.
void ExportCAPI(py::module_ &m)
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
        .Resources_SyncAndHold           = &ImplResources_SyncAndHold,
        .Resources_SubmitSyncOnly        = &ImplResources_SubmitSyncOnly,
        .Resources_SubmitSyncOnlyRaw     = &ImplResources_SubmitSyncOnlyRaw,
        .Stream_HoldResourcesRaw         = &ImplStream_HoldResourcesRaw,
        .Facade_GetType                  = &ImplFacade_GetType,
    };

    m.attr("_C_API") = py::capsule(&capi, "cvcuda._C_API");
#ifdef NB_FREE_THREADED
    static const SubmissionCAPI submissionApi{&BeginSubmission, &EndSubmission};
    m.attr("_C_API_submission") = py::capsule(&submissionApi, "cvcuda._cvcuda._C_API_submission");
#endif
}

void ExportCAPITestHooks(py::module_ &m)
{
    m.def("external_cache_capi_round_trip",
          []
          {
              auto item = std::make_shared<ExternalCacheProbeItem>();
              nvcvpy::Cache::removeAllNotInUseMatching(item->key());
              size_t cacheItems = Cache::Instance().size();
              nvcvpy::Cache::add(*item);

              std::vector<std::shared_ptr<ICacheItem>> fetched = nvcvpy::Cache::fetch(item->key());
              bool exactMatch = fetched.size() == 1 && fetched.front().get() == item.get();
              fetched.clear();
              nvcvpy::Cache::removeAllNotInUseMatching(item->key());

              if (!exactMatch || Cache::Instance().size() != cacheItems)
              {
                  throw CAPIResourceError("external cache C API round trip failed");
              }
          });
    m.def("external_resource_capi_round_trip",
          []
          {
              ExternalResourceProbe facade;
              auto                  resource = ToSharedObj<Resource>(facade.ptr());
              if (!resource)
              {
                  throw CAPIResourceError("external Resource C API round trip failed");
              }
              return resource->id();
          });
    m.def("capi_legacy_hold_resources",
          [](py::object stream, py::object resources)
          {
              ImplStream_HoldResources(stream.ptr(), resources.ptr());
              CheckCAPIError();
          });
    m.def("capi_legacy_submit_sync_only",
          [](py::object stream, py::object resources)
          {
              ImplResources_SubmitSyncOnly(stream.ptr(), resources.ptr());
              CheckCAPIError();
          });
    m.def("capi_legacy_sync_and_hold",
          [](py::object stream, py::object resources)
          {
              ImplResources_SyncAndHold(stream.ptr(), resources.ptr());
              CheckCAPIError();
          });
    m.def("fail_hold_resources", [](bool armed) { GetResourceGuardFailureInjection().streamHoldResources = armed; });
    m.def("fail_submit_sync_only",
          [](bool armed) { GetResourceGuardFailureInjection().resourcesSubmitSyncOnly = armed; });
    m.def("fail_sync_and_hold", [](bool armed) { GetResourceGuardFailureInjection().resourcesSyncAndHold = armed; });
    m.def("hold_resources_on_device",
          [](const std::shared_ptr<Stream> &stream, const py::list &resources, int submissionDevice)
          {
              ResourceBag owners;
              owners.reserve(resources.size());
              for (py::handle resource : resources)
              {
                  owners.push_back(py::cast<std::shared_ptr<Resource>>(resource));
              }
              stream->holdResources(std::move(owners), submissionDevice);
          });
    m.def("ensure_not_capturing_on_device", [](const std::shared_ptr<Stream> &stream, int submissionDevice)
          { stream->ensureNotCapturing(submissionDevice); });
}

} // namespace nvcvpy::priv
