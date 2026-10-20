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
#include "Submission.hpp"

#include <common/Binding.hpp>
#include <cuda_runtime.h>
#include <nvcv/python/Shape.hpp>

#include <memory>
#include <optional>
#include <utility>

namespace nvcvpy::priv {

namespace py = nanobind;

class StreamHandleOwner;

class ExternalBuffer final : public std::enable_shared_from_this<ExternalBuffer>
{
public:
    static void Export(py::module_ &m);

    ExternalBuffer(ExternalBuffer &&that) = delete;

    /**
     * @brief Create an ExternalBuffer py::object that wraps a DLPack tensor.
     *
     * @param dlTensor       Tensor data to wrap.
     * @param wrappedObj     Owner Python object kept alive with the buffer.
     * @param exportStream   Stream handle to advertise via the exported CAI
     *                       `stream` field. A null handle represents CUDA's
     *                       real legacy default stream; any other valid handle
     *                       identifies the producer stream.
     *                       Pass (cudaStream_t)-1 to advertise "no sync
     *                       needed" (CAI `stream: -1`).
     * @param exportStreamOwner Keeps the advertised handle valid for the
     *                       lifetime of the exported view.
     */
    static py::object Create(DLPackTensor &&dlTensor, py::object wrappedObj,
                             std::optional<cudaStream_t>              exportStream      = std::nullopt,
                             std::shared_ptr<const StreamHandleOwner> exportStreamOwner = {});

    // Test-only helper: consume a DLPack capsule from a thread that does not
    // hold the GIL. Bound only under cvcuda._test.
    static void ReleaseDLPackCapsuleOffGil(py::capsule capsule);

    const DLTensor &dlTensor() const;

    Shape      shape() const;
    py::tuple  strides() const;
    py::object dtype() const;

    bool load(PyObject *o);
#ifdef NB_FREE_THREADED
    void seedSubmissionState() const;

    bool readOnly() const
    {
        return m_readOnly;
    }

    SubmissionStates submissionStates() const;
    void             shareSubmissionStates(const SubmissionStates &states);
#endif

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
    void setExportStream(cudaStream_t handle, const std::shared_ptr<const StreamHandleOwner> &owner,
                         py::handle wrapper = py::handle{});

#ifndef NB_FREE_THREADED
    const StreamHandleOwner *exportStreamOwner() const;
#endif

    explicit ExternalBuffer(DLPackTensor &&dlTensor);
    ExternalBuffer() = default;
#ifndef NB_FREE_THREADED
    ~ExternalBuffer();
#endif

private:
#ifdef NB_FREE_THREADED
    mutable py::ft_mutex m_interfaceMutex;
    SubmissionStates     m_submissionStates;
    bool                 m_readOnly = false;
#endif
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
    mutable std::optional<cudaStream_t>              m_exportStream;
    mutable std::shared_ptr<const StreamHandleOwner> m_exportStreamOwner;
#ifndef NB_FREE_THREADED
    mutable cudaEvent_t m_exportEvent = nullptr; // Re-recorded per export; the GIL serializes access.
#endif

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

    // Owns the DLManagedTensor for legacy (v0) imports. DLPack requires the
    // deleter to run on the pointer the capsule carried: producers routinely
    // free the enclosing allocation through it, so handing it the address of a
    // copy frees the wrong object.
    struct LegacyDeleter
    {
        void operator()(DLManagedTensor *p) const
        {
            if (p && p->deleter)
                p->deleter(p);
        }
    };

    std::unique_ptr<DLManagedTensor, LegacyDeleter> m_dlManagedLegacy;

    // Returns the __cuda_array_interface__ if the buffer is cuda-accessible,
    // or std::nullopt if it's not.
    std::optional<py::dict> cudaArrayInterface() const;

    // `interface` is the already-fetched `__cuda_array_interface__` object: the
    // caller reads that property exactly once. `producer` is the object owning
    // the memory, retained to keep the buffer alive.
    bool loadCudaArrayInterface(const py::object &producer, const py::object &interface);
    bool loadDLPack(const py::object &object);
    void loadDLPackCapsule(py::capsule &cap);

    // __dlpack__ implementation
    py::capsule dlpack(py::handle stream, py::handle maxVersion) const;

    // __dlpack_device__ implementation
    py::tuple dlpackDevice() const;
};

} // namespace nvcvpy::priv

namespace nanobind::detail {

namespace priv = nvcvpy::priv;

// ExternalBuffer is both a bound Python class and the input adapter for CUDA
// array interface / DLPack producers.  Keep the regular nanobind class caster
// behavior for bound instances while retaining the established implicit input
// conversion for external producers.
template<>
struct type_caster<priv::ExternalBuffer> : type_caster_base<priv::ExternalBuffer>
{
    using Type = priv::ExternalBuffer;

    static constexpr auto Name = const_name("cvcuda.ExternalBuffer");

    bool from_python(handle src, uint32_t flags, cleanup_list *cleanup);

    template<typename T>
    bool can_cast() const noexcept
    {
        return std::is_pointer_v<T> || m_value != nullptr;
    }

    operator Type *() // NOSONAR: nanobind's caster protocol invokes these conversions implicitly.
    {
        return m_value;
    }

    operator Type &() // NOSONAR: nanobind's caster protocol invokes these conversions implicitly.
    {
        return *m_value;
    }

private:
    // Bound arguments live for the Python call; only implicit adapters need an owner.
    Type                 *m_value = nullptr;
    std::shared_ptr<Type> m_owner;
};

} // namespace nanobind::detail

#endif // NVCV_PYTHON_PRIV_EXTERNAL_BUFFER_HPP
