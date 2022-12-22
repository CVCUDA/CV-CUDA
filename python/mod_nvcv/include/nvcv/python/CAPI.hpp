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

#ifndef NVCV_PYTHON_CAPI_HPP
#define NVCV_PYTHON_CAPI_HPP

#include <cuda_runtime.h>
#include <nvcv/DataType.hpp>
#include <nvcv/ImageBatch.h>
#include <nvcv/Tensor.h>
#include <pybind11/pybind11.h>

namespace pybind11::detail {
// to force inclusion of "DataType.hpp" if needed
struct type_caster<nvcv::DataType>;
} // namespace pybind11::detail

namespace nvcvpy {

class ICacheItem;
class IKey;
class Container;

struct CAPI
{
    PyObject *(*DataType_ToPython)(NVCVDataType p);
    NVCVDataType (*DataType_FromPython)(PyObject *obj);

    PyObject *(*ImageFormat_ToPython)(NVCVImageFormat p);
    NVCVImageFormat (*ImageFormat_FromPython)(PyObject *obj);

    void (*Resource_SubmitSync)(PyObject *res, PyObject *stream, PyObject *lockMode);
    void (*Resource_SubmitSignal)(PyObject *res, PyObject *stream, PyObject *lockMode);

    void (*Stream_HoldResources)(PyObject *stream, PyObject *resources);
    PyObject *(*Stream_GetCurrent)();
    cudaStream_t (*Stream_GetCudaHandle)(PyObject *stream);

    NVCVTensorHandle (*Tensor_GetHandle)(PyObject *tensor);
    PyObject *(*Tensor_Create)(int32_t ndim, const int64_t *shape, NVCVDataType dtype, NVCVTensorLayout layout);
    PyObject *(*Tensor_CreateForImageBatch)(int32_t numImages, int32_t width, int32_t height, NVCVImageFormat fmt);

    PyObject *(*ImageBatchVarShape_Create)(int32_t capacity);
    NVCVImageBatchHandle (*ImageBatchVarShape_GetHandle)(PyObject *varshape);
    void (*ImageBatchVarShape_PushBack)(PyObject *varshape, PyObject *image);
    void (*ImageBatchVarShape_PopBack)(PyObject *varshape, int32_t cnt);
    void (*ImageBatchVarShape_Clear)(PyObject *varshape);

    void (*Cache_Add)(ICacheItem *item);
    ICacheItem **(*Cache_Fetch)(const IKey *key);

    PyObject *(*Image_Create)(int32_t width, int32_t height, NVCVImageFormat fmt);
    NVCVImageHandle (*Image_GetHandle)(PyObject *img);

    PyObject *(*Container_Create)(Container *cont);

    // always add new functions at the end, and never change the function prototypes above.
};

inline const CAPI &capi()
{
    static const CAPI *capi = reinterpret_cast<const CAPI *>(PyCapsule_Import("nvcv._C_API", 0));
    if (capi == nullptr)
    {
        throw std::runtime_error("Can't load pynvcv C API");
    }
    return *capi;
}

} // namespace nvcvpy

#endif // NVCV_PYTHON_CAPI_HPP
