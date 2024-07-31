/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Operators.hpp"
#include "WorkspaceCache.hpp"

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

namespace cvcudapy {

namespace {

// Specialized class for cvcuda::FindHomography operator with a better cache Key.
// It allows for reusing an existing operator object from cache if its payload size is >= the required size.
// It also allows to fetch the biggest payload object to be reused while removing all others.
// This is more flexible than using the generic PyOperator class and its Key class.
class PyOpFindHomography : public nvcvpy::Container
{
public:
    // Define a Key class to be used by the cache to fetch similar items for potential reuse.
    class Key : public nvcvpy::IKey
    {
    public:
        // Arguments of the key constructor should match the corresponding cvcuda operator arguments.
        Key(int batchSize, int maxNumPoints) {}

        size_t payloadSize() const
        {
            return 0;
        }

    private:
        size_t doGetHash() const override
        {
            return 0;
        }

        // The comparison of keys is based on the payload size, the one in the cache is "that" key.
        bool doIsCompatible(const nvcvpy::IKey &that_) const override
        {
            return dynamic_cast<const Key *>(&that_) != nullptr;
        }
    };

    // Constructor instantiate the cache key and the operator object.
    PyOpFindHomography(int batchSize, int maxNumPoints)
        : m_key(batchSize, maxNumPoints)
        , m_op(batchSize, maxNumPoints)
    {
    }

    inline void submit(cudaStream_t stream, const nvcv::Tensor &srcPts, const nvcv::Tensor &dstPts,
                       const nvcv::Tensor &models)
    {
        m_op(stream, srcPts, dstPts, models);
    }

    inline void submit(cudaStream_t stream, const nvcv::TensorBatch &srcPts, const nvcv::TensorBatch &dstPts,
                       const nvcv::TensorBatch &models)
    {
        m_op(stream, srcPts, dstPts, models);
    }

    // Required override to get the py object container.
    py::object container() const override
    {
        return *this;
    }

    // Required override to get the key as the base interface class.
    const nvcvpy::IKey &key() const override
    {
        return m_key;
    }

    // The static fetch function can be used to specialize the fetch of a specific object from the cache.
    // It can be used to select the best object among a number of matched cache objects.
    // It can also be used to remove other objects that are not needed in the cache anymore.
    // Here, it fetches the biggest payload OP among cache items and remove all other OPs from the cache.
    // It is ok to remove them since the biggest payload OP can be used to accomodate all of them,
    // so they will never be reused and thus are no longer necessary.
    static std::shared_ptr<nvcvpy::ICacheItem> fetch(std::vector<std::shared_ptr<nvcvpy::ICacheItem>> &cache)
    {
        assert(!cache.empty());

        std::shared_ptr<nvcvpy::ICacheItem> retItem        = cache[0];
        size_t                              maxPayloadSize = 0;

        for (const auto &item : cache)
        {
            const Key &key            = static_cast<const Key &>(item.get()->key());
            size_t     keyPayloadSize = key.payloadSize();

            if (keyPayloadSize > maxPayloadSize)
            {
                maxPayloadSize = keyPayloadSize;
                retItem        = item;
            }
        }

        cache.clear();

        nvcvpy::Cache::removeAllNotInUseMatching(retItem.get()->key());

        return retItem;
    }

private:
    Key                    m_key;
    cvcuda::FindHomography m_op;
};

Tensor FindHomographyInto(Tensor &models, Tensor &srcPts, Tensor &dstPts, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    // Use CreateOperatorEx to use the extended create operator function passing the specialized PyOperator above
    // as template type, instead of the regular cvcuda::OP class used in the CreateOperator function.
    int32_t batchSize = srcPts.shape()[0];
    int32_t numPoints = srcPts.shape()[1];

    auto findHomography = CreateOperatorEx<PyOpFindHomography>(batchSize, numPoints);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {srcPts});
    guard.add(LockMode::LOCK_MODE_READ, {dstPts});
    guard.add(LockMode::LOCK_MODE_READWRITE, {models});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*findHomography});

    findHomography->submit(pstream->cudaHandle(), srcPts, dstPts, models);

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
        int numPoints = srcPts[i].shape()[1];
        if (numPoints > maxNumPoints)
            maxNumPoints = numPoints;
    }

    auto findHomography = CreateOperatorEx<PyOpFindHomography>(batchSize, maxNumPoints);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {srcPts});
    guard.add(LockMode::LOCK_MODE_READ, {dstPts});
    guard.add(LockMode::LOCK_MODE_READWRITE, {models});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*findHomography});

    findHomography->submit(pstream->cudaHandle(), srcPts, dstPts, models);

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
        models.pushBack(outTensor);
    }

    return VarShapeFindHomographyInto(models, srcPts, dstPts, pstream);
}

} // namespace

void ExportOpFindHomography(py::module &m)
{
    using namespace pybind11::literals;

    py::options options;
    options.disable_function_signatures();

    m.def("findhomography", &FindHomography, "srcPts"_a, "dstPts"_a, "stream"_a = nullptr, R"pbdoc(

	cvcuda.findhomography(srcPts: nvcv.Tensor, dstPts: nvcv.Tensor, stream: Optional[nvcv.cuda.Stream] = None) -> nvcv.Tensor

        Estimates the homography matrix between srcPts and dstPts coordinates on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Find Homography operator
            for more details and usage examples.

        Args:
            srcPts (nvcv.Tensor): Input source coordinates tensor containing 2D coordinates in the source image.
            dstPts (nvcv.Tensor): Input destination coordinates tensor containing 2D coordinates in the target image.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The model homography matrix tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("findhomography_into", &FindHomographyInto, "models"_a, "srcPts"_a, "dstPts"_a, "stream"_a = nullptr, R"pbdoc(

	cvcuda.findhomography_into(models: nvcv.Tensor, srcPts: nvcv.Tensor, dstPts: nvcv.Tensor, stream: Optional[nvcv.cuda.Stream] = None)

        Executes the Find Homography operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Find Homography operator
            for more details and usage examples.

        Args:
            models (nvcv.Tensor): Output model tensor containing 3x3 homography matrices.
            srcPts (nvcv.Tensor): Input source coordinates tensor containing 2D coordinates in the source image.
            dstPts (nvcv.Tensor): Input destination coordinates tensor containing 2D coordinates in the target image.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.Tensor: The model homography matrix tensor.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("findhomography", &VarShapeFindHomography, "srcPts"_a, "dstPts"_a, "stream"_a = nullptr, R"pbdoc(

	cvcuda.findhomography(srcPts: nvcv.TensorBatch, dstPts: nvcv.TensorBatch, stream: Optional[nvcv.cuda.Stream] = None) -> TensorBatch

        Executes the Find Homography operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Find Homography operator
            for more details and usage examples.

        Args:
            srcPts (nvcv.TensorBatch): Input source coordinates tensor containing 2D coordinates in the source image.
            dstPts (nvcv.TensorBatch): Input destination coordinates tensor containing 2D coordinates in the target image.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.TensorBatch: The model homography matrix tensor batch.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("findhomography_into", &VarShapeFindHomographyInto, "models"_a, "srcPts"_a, "dstPts"_a, "stream"_a = nullptr,
          R"pbdoc(

	cvcuda.findhomography(models: nvcv.TensorBatch, srcPts: nvcv.TensorBatch, dstPts: nvcv.TensorBatch, stream: Optional[nvcv.cuda.Stream] = None)

        Executes the Find Homography operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Find Homography operator
            for more details and usage examples.

        Args:
            models (nvcv.TensorBatch): Output model tensor containing 3x3 homography matrices.
            srcPts (nvcv.TensorBatch): Input source coordinates tensor containing 2D coordinates in the source image.
            dstPts (nvcv.TensorBatch): Input destination coordinates tensor containing 2D coordinates in the target image.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            nvcv.TensorBatch: The model homography matrix tensor batch.


        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
