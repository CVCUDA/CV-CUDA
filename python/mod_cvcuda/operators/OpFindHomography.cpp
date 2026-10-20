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

#include "../WorkspaceCache.hpp"
#include "NativeBinding.hpp"
#include "Operators.hpp"

#include <common/Binding.hpp>
#include <common/Hash.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpFindHomography.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace cvcudapy {

// Specialized native state for cvcuda::FindHomography.
// The underlying DeviceState allocates device buffers sized for
// (batchSize, maxNumPoints); a cached op can only be reused when the
// request matches those dimensions exactly, otherwise RunFindHomography
// would issue memsets and kernel launches past the end of the
// allocations. The cache key therefore uses strict equality on both
// ctor arguments (matching the generic OperatorState pattern).
class PyOpFindHomography : public OperatorResource
{
public:
    class Key : public nvcvpy::IKey
    {
    public:
        Key(int batchSize, int maxNumPoints)
            : m_batchSize(batchSize)
            , m_maxNumPoints(maxNumPoints)
        {
        }

    private:
        size_t doGetHash() const override
        {
            return nvcvpy::util::ComputeHash(m_batchSize, m_maxNumPoints);
        }

        bool doIsCompatible(const nvcvpy::IKey &that_) const override
        {
            const auto &that = static_cast<const Key &>(that_);
            return m_batchSize == that.m_batchSize && m_maxNumPoints == that.m_maxNumPoints;
        }

        int m_batchSize;
        int m_maxNumPoints;
    };

    PyOpFindHomography(int batchSize, int maxNumPoints)
        : m_op(batchSize, maxNumPoints)
        , m_batchSize(batchSize)
        , m_maxNumPoints(maxNumPoints)
    {
    }

    inline void submit(cudaStream_t stream, const nvcv::Tensor &srcPts, const nvcv::Tensor &dstPts,
                       const nvcv::Tensor &models) const
    {
        m_op(stream, srcPts, dstPts, models);
    }

    inline void submit(cudaStream_t stream, const nvcv::TensorBatch &srcPts, const nvcv::TensorBatch &dstPts,
                       const nvcv::TensorBatch &models) const
    {
        m_op(stream, srcPts, dstPts, models);
    }

    // A cached op fetched through the capsule API bypasses the cache key, so the
    // caller-supplied dimensions must be re-checked against the ones the device
    // buffers were sized for.
    bool isCompatible(int32_t batchSize, int32_t maxNumPoints) const noexcept
    {
        return m_batchSize == batchSize && m_maxNumPoints == maxNumPoints;
    }

private:
    cvcuda::FindHomography m_op;
    int                    m_batchSize;
    int                    m_maxNumPoints;
};

template<>
struct OperatorCacheUsesCompletionLease<PyOpFindHomography> : std::true_type
{
};

namespace {

using native::Stream;
using native::Tensor;
using native::TensorBatch;

constexpr const char *kFindHomographyOperatorCapsuleName = "FindHomographyOperator";

// The public capsule name is forgeable. Its private context address proves
// that the payload was created by this module before we treat it as a
// shared_ptr<PyOpFindHomography>.
std::byte &FindHomographyOperatorCapsuleContext()
{
    static std::byte context;
    return context;
}

void DestroyFindHomographyOperatorCapsule(PyObject *capsule) noexcept
{
    void *payload = PyCapsule_GetPointer(capsule, kFindHomographyOperatorCapsuleName);
    if (payload == nullptr)
    {
        // Capsule destruction cannot report an exception. This should only be
        // reachable if external code has corrupted a capsule created here.
        PyErr_Clear();
        return;
    }
    delete static_cast<std::shared_ptr<PyOpFindHomography> *>(payload); // NOSONAR: capsule owns the payload.
}

Tensor &FindHomographyInto(Tensor &models, Tensor &srcPts, Tensor &dstPts, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    // Use CreateOperatorEx to use the specialized native state above
    // as template type, instead of the regular cvcuda::OP class used in the CreateOperator function.
    auto batchSize = static_cast<int32_t>(srcPts.impl().shape()[0]);
    auto numPoints = static_cast<int32_t>(srcPts.impl().shape()[1]);

    auto findHomography = CreateOperatorEx<PyOpFindHomography>(batchSize, numPoints);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {srcPts});
    guard.add(LockMode::LOCK_MODE_READ, {dstPts});
    guard.add(LockMode::LOCK_MODE_READWRITE, {models});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*findHomography});

    guard.run([&findHomography, &stream, &srcPts, &dstPts, &models]()
              { findHomography->submit(stream.handle(), srcPts.impl(), dstPts.impl(), models.impl()); });

    return models;
}

std::shared_ptr<Tensor> FindHomography(Tensor &srcPts, Tensor &dstPts, std::optional<Stream *> pstream)
{
    const std::array<int64_t, 3> modelsShape{srcPts.impl().shape()[0], 3, 3};

    auto models = native::CreateTensor({modelsShape.data(), static_cast<int32_t>(modelsShape.size()), nvcv::TENSOR_NHW},
                                       nvcv::TYPE_F32);

    FindHomographyInto(*models, srcPts, dstPts, pstream);
    return models;
}

TensorBatch &VarShapeFindHomographyInto(TensorBatch &models, TensorBatch &srcPts, TensorBatch &dstPts,
                                        std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    // The same native state can be used regardless of Tensors or VarShape.
    int batchSize    = srcPts.numTensors();
    int maxNumPoints = 0;

    for (int i = 0; i < batchSize; i++)
    {
        auto numPoints = static_cast<int>(srcPts.at(i)->impl().shape()[1]);
        if (numPoints > maxNumPoints)
            maxNumPoints = numPoints;
    }

    auto findHomography = CreateOperatorEx<PyOpFindHomography>(batchSize, maxNumPoints);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {srcPts});
    guard.add(LockMode::LOCK_MODE_READ, {dstPts});
    guard.add(LockMode::LOCK_MODE_READWRITE, {models});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*findHomography});

    guard.run([&findHomography, &stream, &srcPts, &dstPts, &models]()
              { findHomography->submit(stream.handle(), srcPts.impl(), dstPts.impl(), models.impl()); });

    return models;
}

std::shared_ptr<TensorBatch> VarShapeFindHomography(TensorBatch &srcPts, TensorBatch &dstPts,
                                                    std::optional<Stream *> pstream)
{
    auto                         models      = TensorBatch::Create(srcPts.numTensors());
    const std::array<int64_t, 3> modelsShape = {1, 3, 3};

    for (int i = 0; i < srcPts.numTensors(); i++)
    {
        auto outTensor = native::CreateTensor(
            {modelsShape.data(), static_cast<int32_t>(modelsShape.size()), nvcv::TENSOR_NHW}, nvcv::TYPE_F32);
        models->pushBack(*outTensor);
    }

    VarShapeFindHomographyInto(*models, srcPts, dstPts, pstream);
    return models;
}

// Get a reusable FindHomography operator that can be passed to findhomography_into_with_op.
// This allows the caller to hold a persistent reference to prevent cache eviction overhead.
// Returns a PyCapsule containing the shared_ptr to the operator.
py::object GetFindHomographyOperator(int32_t batchSize, int32_t numPoints)
{
    auto      op = CreateOperatorEx<PyOpFindHomography>(batchSize, numPoints);
    // Store the shared_ptr on the heap so it can be held by the capsule.
    auto      opPtr = std::make_unique<std::shared_ptr<PyOpFindHomography>>(std::move(op));
    PyObject *raw // NOSONAR: the pointer is handed to nanobind's steal below, which requires a non-const PyObject*.
        = PyCapsule_New(opPtr.get(), kFindHomographyOperatorCapsuleName, &DestroyFindHomographyOperatorCapsule);
    if (raw == nullptr)
    {
        throw py::error_already_set();
    }
    opPtr.release();
    py::object capsule = py::steal<py::object>(raw);

    if (PyCapsule_SetContext(capsule.ptr(), &FindHomographyOperatorCapsuleContext()) != 0)
    {
        throw py::error_already_set();
    }
    return capsule;
}

// Version of FindHomographyInto that accepts a pre-fetched operator.
// This avoids the cache lookup and ResourceGuard overhead that causes bimodal timing.
Tensor &FindHomographyIntoWithOp(Tensor &models, Tensor &srcPts, Tensor &dstPts, py::capsule pyOp,
                                 std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    if (!PyCapsule_IsValid(pyOp.ptr(), kFindHomographyOperatorCapsuleName)
        || PyCapsule_GetContext(pyOp.ptr()) != &FindHomographyOperatorCapsuleContext())
    {
        throw py::type_error("Invalid FindHomography operator capsule");
    }

    const auto *opPtr = static_cast<std::shared_ptr<PyOpFindHomography> *>(
        PyCapsule_GetPointer(pyOp.ptr(), kFindHomographyOperatorCapsuleName));
    if (opPtr == nullptr || !*opPtr)
    {
        PyErr_Clear();
        throw py::type_error("Invalid FindHomography operator capsule");
    }

    auto findHomography = *opPtr;
    if (const auto incompatible =
            [&findHomography](const nvcv::TensorShape &srcShape, const nvcv::TensorShape &dstShape)
        {
            return srcShape.rank() < 2 || dstShape.rank() < 2
                || !findHomography->isCompatible(static_cast<int32_t>(srcShape[0]), static_cast<int32_t>(srcShape[1]))
                || !findHomography->isCompatible(static_cast<int32_t>(dstShape[0]), static_cast<int32_t>(dstShape[1]));
        };
        incompatible(srcPts.impl().shape(), dstPts.impl().shape()))
    {
        throw std::invalid_argument("FindHomography operator dimensions do not match srcPts and dstPts");
    }

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {srcPts});
    guard.add(LockMode::LOCK_MODE_READ, {dstPts});
    guard.add(LockMode::LOCK_MODE_READWRITE, {models});
    // The capsule can be reused directly by concurrent callers, outside the
    // completion-lease cache. Serialize and order its mutable workspace.
    guard.add(LockMode::LOCK_MODE_READWRITE, {*findHomography});

    guard.run([&findHomography, &stream, &srcPts, &dstPts, &models]()
              { findHomography->submit(stream.handle(), srcPts.impl(), dstPts.impl(), models.impl()); });

    return models;
}

} // namespace

void ExportOpFindHomography(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("findhomography", NvtxTrace("cvcuda.findhomography", &FindHomography), "srcPts"_a, "dstPts"_a,
          "stream"_a = nullptr, R"pbdoc(
        Estimates the homography matrix between srcPts and dstPts coordinates on the given cuda stream.


        Args:
            srcPts (cvcuda.Tensor): Input source coordinates tensor containing 2D coordinates in the source image.
            dstPts (cvcuda.Tensor): Input destination coordinates tensor containing 2D coordinates in the target image.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The model homography matrix tensor.

    )pbdoc");

    m.def("findhomography_into", NvtxTrace("cvcuda.findhomography_into", &FindHomographyInto), "models"_a, "srcPts"_a,
          "dstPts"_a, "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Executes the Find Homography operation on the given cuda stream.


        Args:
            models (cvcuda.Tensor): Output model tensor containing 3x3 homography matrices.
            srcPts (cvcuda.Tensor): Input source coordinates tensor containing 2D coordinates in the source image.
            dstPts (cvcuda.Tensor): Input destination coordinates tensor containing 2D coordinates in the target image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The model homography matrix tensor.

    )pbdoc");

    m.def("findhomography", NvtxTrace("cvcuda.findhomography", &VarShapeFindHomography), "srcPts"_a, "dstPts"_a,
          "stream"_a = nullptr, R"pbdoc(
        Executes the Find Homography operation on the given cuda stream.


        Args:
            srcPts (cvcuda.TensorBatch): Input source coordinates tensor containing 2D coordinates in the source image.
            dstPts (cvcuda.TensorBatch): Input destination coordinates tensor containing 2D coordinates in the target image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.TensorBatch: The model homography matrix tensor batch.

    )pbdoc");

    m.def("findhomography_into", NvtxTrace("cvcuda.findhomography_into", &VarShapeFindHomographyInto), "models"_a,
          "srcPts"_a, "dstPts"_a, "stream"_a = nullptr, py::return_value_policy::reference,
          R"pbdoc(
        Executes the Find Homography operation on the given cuda stream.


        Args:
            models (cvcuda.TensorBatch): Output model tensor containing 3x3 homography matrices.
            srcPts (cvcuda.TensorBatch): Input source coordinates tensor containing 2D coordinates in the source image.
            dstPts (cvcuda.TensorBatch): Input destination coordinates tensor containing 2D coordinates in the target image.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.TensorBatch: The model homography matrix tensor batch.


    )pbdoc");

    m.def("get_findhomography_operator", NvtxTrace("cvcuda.get_findhomography_operator", &GetFindHomographyOperator),
          "batch_size"_a, "num_points"_a, R"pbdoc(
        Get a reusable FindHomography operator for the given dimensions.

        This allows holding a persistent reference to the operator to avoid
        the cache eviction overhead that can cause bimodal timing patterns
        when calling findhomography_into repeatedly.

        Args:
            batch_size (int): Number of samples in the batch.
            num_points (int): Number of points per sample.

        Returns:
            object: A FindHomography operator that can be passed to findhomography_into_with_op.

        Example:
            >>> op = cvcuda.get_findhomography_operator(1024, 2048)
            >>> for _ in range(iterations):
            ...     cvcuda.findhomography_into_with_op(models, src, dst, op, stream=stream)
    )pbdoc");

    m.def("findhomography_into_with_op", NvtxTrace("cvcuda.findhomography_into_with_op", &FindHomographyIntoWithOp),
          "models"_a, "srcPts"_a, "dstPts"_a, "operator"_a, "stream"_a = nullptr, py::return_value_policy::reference,
          R"pbdoc(
        Executes the Find Homography operation using a pre-fetched operator.

        This version accepts an operator obtained from get_findhomography_operator(),
        which avoids cache lookup overhead and prevents bimodal timing patterns.


        Args:
            models (cvcuda.Tensor): Output model tensor containing 3x3 homography matrices.
            srcPts (cvcuda.Tensor): Input source coordinates tensor containing 2D coordinates in the source image.
            dstPts (cvcuda.Tensor): Input destination coordinates tensor containing 2D coordinates in the target image.
            operator (object): Pre-fetched operator from get_findhomography_operator().
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The model homography matrix tensor.

    )pbdoc");
}

} // namespace cvcudapy
