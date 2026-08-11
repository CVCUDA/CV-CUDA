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

#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace nvcvpy::priv {

namespace {

class CAPIResourceError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
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

struct TestFailureInjection
{
    bool streamHoldResources     = false;
    bool resourcesSubmitSyncOnly = false;
    bool resourcesSyncAndHold    = false;
};

TestFailureInjection &testFailureInjection()
{
    static TestFailureInjection injection;
    return injection;
}

void ThrowIfInjected(bool armed, const char *what)
{
    if (armed)
    {
        throw CAPIResourceError(what);
    }
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

PyObject *ImplDataType_ToPython(NVCVDataType p)
{
    try
    {
        py::object obj = py::cast(nvcv::DataType(p));
        return obj.ptr();
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

PyObject *ImplImageFormat_ToPython(NVCVImageFormat p)
{
    try
    {
        py::object obj = py::cast(nvcv::ImageFormat(p));
        return obj.ptr();
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

// ----- Resource-pointer cache (weakref-invalidated) -------------------------
//
// Inside the per-resource hot loop of Resources_SyncAndHold, the dominant
// per-item cost is the pybind11 reverse-cast `ToSharedObj<Resource>(PyObject*)`
// — a type-registry lookup plus holder extraction, on the order of ~1 us per
// resource. Bench-style workloads call the same op repeatedly with the same
// Tensor / ImageBatch wrappers, so caching by raw PyObject* skips the cast on
// repeats.
//
// The hazard with raw PyObject* keys is identity reuse: when a wrapper is
// destroyed, Python's allocator can recycle its memory address for an
// unrelated new object. A naive cache would then return a stale Resource for
// the new wrapper. We block this by registering a weakref-with-callback on
// each cached PyObject; the callback fires inside the wrapper's tp_dealloc,
// *before* the address can be reused, and erases the cache entry.
//
// All access happens under the GIL (this code is reachable only from C-API
// entrypoints invoked by Python), so no extra synchronization is required.

// Self-decref'ing PyObject reference holder (for weakref strong refs).
struct PyRef
{
    PyObject *p = nullptr;

    PyRef() = default;

    explicit PyRef(PyObject *x)
        : p(x)
    {
    }

    ~PyRef()
    {
        if (p)
        {
            Py_DECREF(p);
        }
    }

    PyRef(const PyRef &)            = delete;
    PyRef &operator=(const PyRef &) = delete;

    PyRef(PyRef &&o) noexcept
        : p(o.p)
    {
        o.p = nullptr;
    }

    PyRef &operator=(PyRef &&o) noexcept
    {
        if (this != &o)
        {
            if (p)
            {
                Py_DECREF(p);
            }
            p   = o.p;
            o.p = nullptr;
        }
        return *this;
    }
};

struct ResCacheEntry
{
    std::shared_ptr<Resource> res;
    PyRef                     weakref; // owns a strong ref; ~PyRef -> Py_DECREF on erase
};

std::unordered_map<PyObject *, ResCacheEntry> &ResCache()
{
    static std::unordered_map<PyObject *, ResCacheEntry> cache;
    return cache;
}

std::unordered_map<PyObject *, PyObject *> &WeakrefToObj()
{
    // weakref* -> tracked obj* (borrowed)
    static std::unordered_map<PyObject *, PyObject *> cache;
    return cache;
}

PyObject *ImplResCache_Invalidate(PyObject * /*self*/, PyObject *weakref)
{
    auto &weakrefToObj = WeakrefToObj();
    if (auto wrIt = weakrefToObj.find(weakref); wrIt != weakrefToObj.end())
    {
        // Erasing the cache entry destroys the ResCacheEntry, whose ~PyRef
        // releases our strong ref on `weakref`. CPython's weakref machinery
        // bumps the weakref refcount before invoking us, so the implied
        // decref-to-zero here cannot dangle the object we're inside of.
        ResCache().erase(wrIt->second);
        weakrefToObj.erase(wrIt);
    }
    Py_RETURN_NONE;
}

PyMethodDef &InvalidateMethodDef()
{
    static PyMethodDef methodDef = {"_resCache_invalidate", ImplResCache_Invalidate, METH_O, nullptr};
    return methodDef;
}

PyObject *GetResCacheInvalidateCallback()
{
    static PyObject *cb = []()
    {
        // Module-lifetime, never released. PyCFunction_NewEx requires a
        // module-or-None reference; nullptr is fine for our purposes.
        return PyCFunction_NewEx(&InvalidateMethodDef(), nullptr, nullptr);
    }();
    return cb;
}

// Cache-aware Resource extractor. On miss, performs the pybind11 reverse cast
// and registers a weakref so the entry self-invalidates when the wrapper is
// destroyed. Falls back to plain extraction if weakrefs aren't supported.
std::shared_ptr<Resource> ExtractResourceCached(PyObject *obj)
{
    auto &resCache = ResCache();
    if (auto it = resCache.find(obj); it != resCache.end())
    {
        return it->second.res;
    }
    auto      res    = ToSharedObj<Resource>(obj);
    PyObject *cb     = GetResCacheInvalidateCallback();
    PyObject *wr_raw = (cb != nullptr) ? PyWeakref_NewRef(obj, cb) : nullptr;
    if (wr_raw == nullptr)
    {
        // Wrapper type doesn't support weakrefs (or callback init failed).
        // Skip caching to preserve correctness; pay the cast every call.
        PyErr_Clear();
        return res;
    }

    // Take RAII ownership of the new weakref reference immediately so any
    // throw between here and the successful inserts cleans it up.
    PyRef wr_holder{wr_raw};
    auto &weakrefToObj = WeakrefToObj();
    try
    {
        // Reverse map first so the order matches the natural cleanup path:
        // if the second emplace below throws, we erase this entry in the
        // catch block.
        weakrefToObj.try_emplace(wr_raw, obj);
        // Move ownership of the weakref ref into the cache entry.  If the
        // emplace fails, the temporary ResCacheEntry's destructor (~PyRef)
        // decrefs wr_raw — wr_holder is then moved-from and is a no-op at
        // its own destruction.
        resCache.try_emplace(obj, ResCacheEntry{res, std::move(wr_holder)});
    }
    catch (...)
    {
        // erase() by key is safe even when the entry was never inserted
        // (no-op) and even when wr_raw is now a dangling pointer (we don't
        // dereference it — the unordered_map only hashes/compares the
        // pointer value).
        weakrefToObj.erase(wr_raw);
        throw;
    }
    return res;
}

// ----- end of Resource-pointer cache ----------------------------------------

void ImplResource_SubmitSync(PyObject *res, PyObject *stream)
{
    try
    {
        ToSharedObj<Resource>(res)->submitSync(*ToSharedObj<Stream>(stream));
    }
    CATCH_RETURN_DEFAULT(, "Submit sync failed")
}

void ImplStream_HoldResources(PyObject *stream, PyObject *resourceList)
{
    try
    {
        ThrowIfInjected(testFailureInjection().streamHoldResources, "injected Stream_HoldResources failure");

        py::list resList = ToObj<py::list>(resourceList);

        LockResources resVector;

        const PyObject *lastModePtr = nullptr;
        LockMode        lastMode    = LockMode::LOCK_MODE_NONE;

        for (py::handle h : resList)
        {
            py::tuple t = h.cast<py::tuple>();
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

            auto res = ExtractResourceCached(t[1].ptr());

            resVector.emplace(lockMode, res);
        }

        ToSharedObj<Stream>(stream)->holdResources(std::move(resVector));
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
void ImplResources_SubmitSyncOnly(PyObject *stream, PyObject *resourceList)
{
    try
    {
        ThrowIfInjected(testFailureInjection().resourcesSubmitSyncOnly, "injected Resources_SubmitSyncOnly failure");

        py::list resList   = ToObj<py::list>(resourceList);
        auto     pyStream  = ToSharedObj<Stream>(stream);
        Stream  &streamRef = *pyStream;

        for (py::handle h : resList)
        {
            py::tuple t = h.cast<py::tuple>();
            if (t.size() != 2)
            {
                throw CAPIResourceError("ResourcePerMode tuple must have two elements");
            }

            // Per-resource cast goes through the weakref-invalidated cache so
            // repeat ops on the same wrapper skip the pybind11 holder lookup.
            auto res = ExtractResourceCached(t[1].ptr());
            res->submitSync(streamRef);
        }
    }
    CATCH_RETURN_DEFAULT(, "Submit sync only failed")
}

// Batched equivalent of "for r in list: Resource_SubmitSync(r, stream)" followed
// by Stream_HoldResources, executed entirely in C++. Saves N pybind11/C-ABI
// boundary crossings on ops that track many resources. Per-resource cast goes
// through ExtractResourceCached which dedups repeat lookups for the same
// PyObject; see the cache section above for the identity-reuse safety story.
//
// NOTE: this combined call is retained for backward compatibility with
// out-of-tree consumers that haven't migrated to the run()-based pattern.
// New code should use Resources_SubmitSyncOnly + Stream_HoldResources via
// ResourceGuard::run().
void ImplResources_SyncAndHold(PyObject *stream, PyObject *resourceList)
{
    try
    {
        ThrowIfInjected(testFailureInjection().resourcesSyncAndHold, "injected Resources_SyncAndHold failure");

        py::list resList   = ToObj<py::list>(resourceList);
        auto     pyStream  = ToSharedObj<Stream>(stream);
        Stream  &streamRef = *pyStream;

        LockResources resVector;

        // Within a single commit() call, contiguous items typically share the
        // same py::str instance for their lock mode (one .add(mode, {...})
        // call -> N items with the same mode). A one-slot last-seen cache
        // skips the string parse on those.
        const PyObject *lastModePtr = nullptr;
        LockMode        lastMode    = LockMode::LOCK_MODE_NONE;

        for (py::handle h : resList)
        {
            py::tuple t = h.cast<py::tuple>();
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

            auto res = ExtractResourceCached(t[1].ptr());

            // Per-resource stream-ordering registration.
            res->submitSync(streamRef);

            resVector.emplace(lockMode, res);
        }

        pyStream->holdResources(std::move(resVector));
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
        .Resources_SyncAndHold           = &ImplResources_SyncAndHold,
        .Resources_SubmitSyncOnly        = &ImplResources_SubmitSyncOnly,
    };

    m.add_object("_C_API", py::capsule(&capi, "cvcuda._C_API"));
}

void ExportCAPITestHooks(py::module &m)
{
    m.def("fail_hold_resources", [](bool armed) { testFailureInjection().streamHoldResources = armed; });
    m.def("fail_submit_sync_only", [](bool armed) { testFailureInjection().resourcesSubmitSyncOnly = armed; });
    m.def("fail_sync_and_hold", [](bool armed) { testFailureInjection().resourcesSyncAndHold = armed; });
}

} // namespace nvcvpy::priv
