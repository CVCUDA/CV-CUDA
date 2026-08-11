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

#ifndef NVCV_PYTHON_PRIV_EXTERNAL_BUFFER_HPP
#define NVCV_PYTHON_PRIV_EXTERNAL_BUFFER_HPP

#include "DLPackUtils.hpp"

#include <cuda_runtime.h>
#include <nvcv/python/Shape.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace nvcvpy::priv {

namespace py = pybind11;

class ExternalBuffer final : public std::enable_shared_from_this<ExternalBuffer>
{
public:
    static void Export(py::module &m);

    ExternalBuffer(ExternalBuffer &&that) = delete;

    /**
     * @brief Create an ExternalBuffer py::object that wraps a DLPack tensor.
     *
     * @param dlTensor       Tensor data to wrap.
     * @param wrappedObj     Owner Python object kept alive with the buffer.
     * @param exportStream   Stream handle to advertise via the exported CAI
     *                       `stream` field.  Use the special value (void*)0x1
     *                       to skip populating (defaults to legacy default
     *                       stream), or any valid CUDA stream handle when the
     *                       producer (typically a cvcuda Tensor) wants
     *                       downstream consumers to wait on a specific stream.
     *                       Pass (cudaStream_t)-1 to advertise "no sync
     *                       needed" (CAI `stream: -1`).
     * @param setExportStream When true, `exportStream` is used to populate
     *                       the CAI `stream` field (including the -1
     *                       opt-out).  When false, the field defaults to
     *                       `1` (legacy default).
     */
    static py::object Create(DLPackTensor &&dlTensor, py::object wrappedObj, cudaStream_t exportStream = nullptr,
                             bool setExportStream = false);

    const DLTensor &dlTensor() const;

    Shape      shape() const;
    py::tuple  strides() const;
    py::object dtype() const;

    bool load(PyObject *o);

    /**
     * @brief Producer stream handle advertised via CAI `stream` on wrap.
     *
     * Returns the CUDA stream on which the producing library (e.g. cupy,
     * torch) has outstanding work for this buffer, as reported by
     * `__cuda_array_interface__["stream"]` at `load()` time.
     *
     * Returns 0 if:
     *   - The producer did not populate a stream field (v2 CAI, or absent),
     *     in which case the legacy default stream is implied.
     *   - The producer set `stream: None` or `stream: -1` (no sync required).
     *   - The buffer was constructed locally (not wrapped from Python).
     *
     * Valid only after `load()` has populated it; returns 0 otherwise.
     */
    cudaStream_t producerStream() const
    {
        return m_producerStream;
    }

    /**
     * @brief Whether the wrap-time CAI indicated that no sync is required.
     *
     * True when the producer advertised `stream: None` or `stream: -1`.
     * In that case, `producerStream()` returns 0 and the consumer must
     * NOT insert any implicit wait.
     */
    bool producerIsSynced() const
    {
        return m_producerIsSynced;
    }

    /**
     * @brief CUDA device on which the producer stream lives.
     *
     * Captured from `cudaPointerGetAttributes` on the buffer's device
     * pointer at wrap time.  Returns -1 if unknown (e.g. local buffer).
     */
    int producerDevice() const
    {
        return m_producerDevice;
    }

    /**
     * @brief Set the stream handle to advertise on CAI export (`cudaArrayInterface`).
     *
     * Typically called by Tensor::cuda() / Image::cuda() with the current
     * cvcuda owning stream of the data so downstream consumers (cupy/torch)
     * know which stream to synchronize with.
     */
    void setExportStream(cudaStream_t handle)
    {
        m_exportStream    = handle;
        m_hasExportStream = true;
        m_cacheCudaArrayInterface.reset();
    }

    explicit ExternalBuffer(DLPackTensor &&dlTensor);
    ExternalBuffer() = default;

private:
    friend py::detail::type_caster<ExternalBuffer>;

    DLPackTensor                    m_dlTensor;
    mutable std::optional<py::dict> m_cacheCudaArrayInterface;
    py::object                      m_wrappedObj;

    // Producer stream advertised by the wrapped Python object's CAI dict
    // (`__cuda_array_interface__["stream"]`).  See accessors above for
    // semantics of the three fields.
    cudaStream_t m_producerStream   = nullptr;
    bool         m_producerIsSynced = false;
    int          m_producerDevice   = -1;

    // Producer stream to advertise back out via CAI export.  When unset, the
    // exporter emits the conservative "stream: 1" (legacy default stream).
    cudaStream_t m_exportStream    = nullptr;
    bool         m_hasExportStream = false;

    // Owns the DLManagedTensorVersioned for v1.0 imports.
    // null for v0 imports and locally-created tensors.
    struct VersionedDeleter
    {
        void operator()(DLManagedTensorVersioned *p) const
        {
            if (p && p->deleter)
                p->deleter(p);
        }
    };

    std::unique_ptr<DLManagedTensorVersioned, VersionedDeleter> m_dlManagedVersioned;

    // Returns the __cuda_array_interface__ if the buffer is cuda-accessible,
    // or std::nullopt if it's not.
    std::optional<py::dict> cudaArrayInterface() const;

    bool loadCudaArrayInterface(const py::object &object);
    bool loadDLPack(const py::object &object);
    void loadDLPackCapsule(py::capsule &cap);

    // __dlpack__ implementation
    py::capsule dlpack(py::object stream, py::object maxVersion) const;

    // __dlpack_device__ implementation
    py::tuple dlpackDevice() const;
};

} // namespace nvcvpy::priv

namespace PYBIND11_NAMESPACE { namespace detail {

namespace priv = nvcvpy::priv;

template<>
struct type_caster<priv::ExternalBuffer> : public type_caster_base<priv::ExternalBuffer>
{
    using type = priv::ExternalBuffer;
    using Base = type_caster_base<type>;

    PYBIND11_TYPE_CASTER(std::shared_ptr<type>, const_name("cvcuda.ExternalBuffer"));

    explicit operator type *()
    {
        return value.get();
    }

    explicit operator type &()
    {
        return *value;
    }

    bool load(handle src, bool);
};

}} // namespace PYBIND11_NAMESPACE::detail

#endif // NVCV_PYTHON_PRIV_EXTERNAL_BUFFER_HPP
