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
#include "Operators.hpp"

#include <common/Hash.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpFindHomography.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageFormat.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <nvcv/python/TensorBatch.hpp>
#include <pybind11/stl.h>

#include <memory>

namespace cvcudapy {

// Specialized Python wrapper for cvcuda::FindHomography.
// The underlying DeviceState allocates device buffers sized for
// (batchSize, maxNumPoints); a cached op can only be reused when the
// request matches those dimensions exactly, otherwise RunFindHomography
// would issue memsets and kernel launches past the end of the
// allocations. The cache key therefore uses strict equality on both
// ctor arguments (matching the generic PyOperator pattern).
class PyOpFindHomography : public nvcvpy::Container // NOSONAR: operator wrappers share the Python cache hierarchy.
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
        : m_key(batchSize, maxNumPoints)
        , m_op(batchSize, maxNumPoints)
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

    py::object container() const override
    {
        return py::reinterpret_borrow<py::object>(this->ptr());
    }

    const nvcvpy::IKey &key() const override
    {
        return m_key;
    }

    // Items returned by Cache::fetch have already passed the strict-equality check above,
    // so any one of them can serve the request.
    static std::shared_ptr<nvcvpy::ICacheItem> fetch(std::vector<std::shared_ptr<nvcvpy::ICacheItem>> &cache)
    {
        assert(!cache.empty());
        return cache[0];
    }

private:
    Key                    m_key;
    cvcuda::FindHomography m_op;
};

namespace {

Tensor FindHomographyInto(Tensor &models, Tensor &srcPts, Tensor &dstPts, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    // Use CreateOperatorEx to use the extended create operator function passing the specialized PyOperator above
    // as template type, instead of the regular cvcuda::OP class used in the CreateOperator function.
    auto batchSize = static_cast<int32_t>(srcPts.shape()[0]);
    auto numPoints = static_cast<int32_t>(srcPts.shape()[1]);

    auto findHomography = CreateOperatorEx<PyOpFindHomography>(batchSize, numPoints);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {srcPts});
    guard.add(LockMode::LOCK_MODE_READ, {dstPts});
    guard.add(LockMode::LOCK_MODE_READWRITE, {models});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*findHomography});

    guard.run([&findHomography, &pstream, &srcPts, &dstPts, &models]()
              { findHomography->submit(pstream->cudaHandle(), srcPts, dstPts, models); });

    return models;
}

Tensor FindHomography(Tensor &srcPts, Tensor dstPts, std::optional<Stream> pstream)
{
    Shape modelsShape(3);
    modelsShape[0] = srcPts.shape()[0];
    modelsShape[1] = 3;
    modelsShape[2] = 3;

    Tensor models = Tensor::Create(modelsShape, nvcv::TYPE_F32, nvcv::TENSOR_NHW);

    return FindHomographyInto(models, srcPts, dstPts, pstream);
}

TensorBatch VarShapeFindHomographyInto(TensorBatch &models, TensorBatch &srcPts, TensorBatch &dstPts,
                                       std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    // The same PyOpFindHomography class and CreateOperatorEx function can be used regardless of Tensors or VarShape.
    int batchSize    = srcPts.numTensors();
    int maxNumPoints = 0;

    for (int i = 0; i < batchSize; i++)
    {
        auto numPoints = static_cast<int>(srcPts[i].shape()[1]);
        if (numPoints > maxNumPoints)
            maxNumPoints = numPoints;
    }

    auto findHomography = CreateOperatorEx<PyOpFindHomography>(batchSize, maxNumPoints);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {srcPts});
    guard.add(LockMode::LOCK_MODE_READ, {dstPts});
    guard.add(LockMode::LOCK_MODE_READWRITE, {models});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*findHomography});

    guard.run([&findHomography, &pstream, &srcPts, &dstPts, &models]()
              { findHomography->submit(pstream->cudaHandle(), srcPts, dstPts, models); });

    return models;
}

TensorBatch VarShapeFindHomography(TensorBatch &srcPts, TensorBatch &dstPts, std::optional<Stream> pstream)
{
    TensorBatch models = TensorBatch::Create(srcPts.numTensors());

    Shape modelsShape(3);
    modelsShape[0] = 1;
    modelsShape[1] = 3;
    modelsShape[2] = 3;

    for (int i = 0; i < srcPts.numTensors(); i++)
    {
        Tensor outTensor = Tensor::Create(modelsShape, nvcv::TYPE_F32, nvcv::TENSOR_NHW);
        models.pushBackTensor(outTensor);
    }

    return VarShapeFindHomographyInto(models, srcPts, dstPts, pstream);
}

// Get a reusable FindHomography operator that can be passed to findhomography_into_with_op.
// This allows the caller to hold a persistent reference to prevent cache eviction overhead.
// Returns a PyCapsule containing the shared_ptr to the operator.
py::object GetFindHomographyOperator(int32_t batchSize, int32_t numPoints)
{
    auto op = CreateOperatorEx<PyOpFindHomography>(batchSize, numPoints);
    // Store the shared_ptr on the heap so it can be held by the capsule.
    auto opPtr = std::make_unique<std::shared_ptr<PyOpFindHomography>>(std::move(op));
    return py::capsule(opPtr.release(), "FindHomographyOperator",
                       [](PyObject *capsule)
                       {
                           auto ptr = PyCapsule_GetPointer(capsule, "FindHomographyOperator");
                           std::unique_ptr<std::shared_ptr<PyOpFindHomography>> opPtr(
                               static_cast<std::shared_ptr<PyOpFindHomography> *>(ptr));
                           (void)opPtr;
                       });
}

// Version of FindHomographyInto that accepts a pre-fetched operator.
// This avoids the cache lookup and ResourceGuard overhead that causes bimodal timing.
Tensor FindHomographyIntoWithOp(Tensor &models, Tensor &srcPts, Tensor &dstPts, py::capsule pyOp,
                                std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    // Extract the shared_ptr from the capsule
    const auto *opPtr          = static_cast<std::shared_ptr<PyOpFindHomography> *>(pyOp.get_pointer());
    auto        findHomography = *opPtr;

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {srcPts});
    guard.add(LockMode::LOCK_MODE_READ, {dstPts});
    guard.add(LockMode::LOCK_MODE_READWRITE, {models});
    // NOTE: Do NOT add findHomography to ResourceGuard - the caller holds the reference

    guard.run([&findHomography, &pstream, &srcPts, &dstPts, &models]()
              { findHomography->submit(pstream->cudaHandle(), srcPts, dstPts, models); });

    return models;
}

} // namespace

void ExportOpFindHomography(py::module &m)
{
    using namespace pybind11::literals;

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
          "dstPts"_a, "stream"_a = nullptr, R"pbdoc(
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
          "srcPts"_a, "dstPts"_a, "stream"_a = nullptr,
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
          "models"_a, "srcPts"_a, "dstPts"_a, "operator"_a, "stream"_a = nullptr, R"pbdoc(
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
