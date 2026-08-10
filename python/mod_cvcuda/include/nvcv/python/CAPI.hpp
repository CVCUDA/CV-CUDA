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

#ifndef NVCV_PYTHON_CAPI_HPP
#define NVCV_PYTHON_CAPI_HPP

#include <cuda_runtime.h>
#include <nvcv/Array.h>
#include <nvcv/DataType.hpp>
#include <nvcv/ImageBatch.h>
#include <nvcv/Tensor.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

namespace pybind11::detail {
// to force inclusion of "DataType.hpp" if needed
struct type_caster<nvcv::DataType>;
} // namespace pybind11::detail

namespace nvcvpy {

class ICacheItem;
class IKey;
class Container;

class CAPIError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

struct CAPI // NOSONAR: Python extension ABI table must keep one slot per exported callback.
{
    PyObject *(*DataType_ToPython)(NVCVDataType p);
    NVCVDataType (*DataType_FromPython)(PyObject *obj);

    PyObject *(*ImageFormat_ToPython)(NVCVImageFormat p);
    NVCVImageFormat (*ImageFormat_FromPython)(PyObject *obj);

    void (*Resource_SubmitSync)(PyObject *res, PyObject *stream);

    void (*Stream_HoldResources)(PyObject *stream, PyObject *resources);
    PyObject *(*Stream_GetCurrent)();
    cudaStream_t (*Stream_GetCudaHandle)(PyObject *stream);

    NVCVTensorHandle (*Tensor_GetHandle)(PyObject *tensor);
    PyObject *(*Tensor_Create)(int32_t ndim, const int64_t *shape, NVCVDataType dtype, NVCVTensorLayout layout,
                               int32_t rowAlign);
    PyObject *(*Tensor_CreateForImageBatch)(int32_t numImages, int32_t width, int32_t height, NVCVImageFormat fmt,
                                            int32_t rowAlign);

    NVCVArrayHandle (*Array_GetHandle)(PyObject *array);
    PyObject *(*Array_Create)(int64_t length, NVCVDataType dtype);

    PyObject *(*ImageBatchVarShape_Create)(int32_t capacity);
    NVCVImageBatchHandle (*ImageBatchVarShape_GetHandle)(PyObject *varshape);
    void (*ImageBatchVarShape_PushBack)(PyObject *varshape, PyObject *image);
    void (*ImageBatchVarShape_PopBack)(PyObject *varshape, int32_t cnt);
    void (*ImageBatchVarShape_Clear)(PyObject *varshape);

    void (*Cache_Add)(ICacheItem *item);
    ICacheItem **(*Cache_Fetch)(const IKey *key);

    PyObject *(*Image_Create)(int32_t width, int32_t height, NVCVImageFormat fmt, int32_t rowAlign);
    NVCVImageHandle (*Image_GetHandle)(PyObject *img);

    PyObject *(*Container_Create)(Container *cont);

    void (*Cache_RemoveAllNotInUseMatching)(const IKey *key);

    PyObject *(*TensorBatch_Create)(int32_t capacity);

    NVCVTensorBatchHandle (*TensorBatch_GetHandle)(PyObject *tensorBatch);

    void (*TensorBatch_PushBack)(PyObject *tensorBatch, PyObject *tensor);

    void (*TensorBatch_PopBack)(PyObject *tensorBatch, uint32_t cnt);

    void (*TensorBatch_Clear)(PyObject *tensorBatch);

    // Batched sync-and-hold: takes the full resource list (same shape as
    // Stream_HoldResources expects — list of (lockmode_str, resource) tuples)
    // and runs the per-resource submitSync inside C++, then holdResources,
    // in one C-ABI round trip. Avoids N pybind11 boundary crossings for ops
    // with many tracked resources (erase, threshold, normalize, ...). Use
    // this from ResourceGuard::commit() in lieu of N×Resource_SubmitSync +
    // 1×Stream_HoldResources.
    void (*Resources_SyncAndHold)(PyObject *stream, PyObject *resourceList);

    // Batched submit-sync only (no hold): inserts producer→consumer wait
    // events for every resource in the list against `stream`.  This is the
    // correct point in time to insert sync barriers — it must run BEFORE
    // `op->submit()` queues the consumer's kernel on `stream`, otherwise
    // `cudaStreamWaitEvent` is enqueued behind the kernel and provides no
    // protection.  Pair with `Stream_HoldResources` at scope end (run via
    // `ResourceGuard::run()` which handles both halves correctly).
    void (*Resources_SubmitSyncOnly)(PyObject *stream, PyObject *resourceList);

    // always add new functions at the end, and never change the function prototypes above.
};

inline const CAPI &capi()
{
    static const auto *capi = reinterpret_cast<const CAPI *>(PyCapsule_Import("cvcuda._C_API", 0));
    if (capi == nullptr)
    {
        throw CAPIError("Can't load cvcuda C API");
    }
    return *capi;
}

/* Check for an error inside the CAPI, since exceptions cannot cross the C api
 * boundary, this must be called to make sure en exception was not converted to
 * a PyErr
 */
inline void CheckCAPIError()
{
    if (PyErr_Occurred())
    {
        // Propagate the exception to Python
        throw pybind11::error_already_set();
    }
};

template<class T>
decltype(auto) CheckCAPIError(T &&arg)
{
    CheckCAPIError();
    return std::forward<T>(arg);
}

} // namespace nvcvpy

#endif // NVCV_PYTHON_CAPI_HPP
