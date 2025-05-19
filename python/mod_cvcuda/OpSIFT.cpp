/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/PyUtil.hpp>
#include <cvcuda/OpSIFT.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace cvcudapy {

namespace {

using TupleTensor4 = std::tuple<Tensor, Tensor, Tensor, Tensor>;

class PyOpSIFT : public nvcvpy::Container
{
public:
    class Key : public nvcvpy::IKey
    {
    public:
        Key(const int3 &maxShape, int maxOctaveLayers)
            : m_maxShape{maxShape}
            , m_maxOctaveLayers{maxOctaveLayers}
        {
        }

        bool canBeUsedWith(const int3 &maxShape, int maxOctaveLayers) const
        {
            return (maxShape.x <= m_maxShape.x && maxShape.y <= m_maxShape.y && maxShape.z <= m_maxShape.z
                    && maxOctaveLayers <= m_maxOctaveLayers);
        }

        long long int payloadSize() const
        {
            return static_cast<long long int>(m_maxShape.x) * m_maxShape.y * m_maxShape.z * m_maxOctaveLayers;
        }

    private:
        size_t doGetHash() const override
        {
            return nvcvpy::util::ComputeHash(m_maxShape);
        }

        bool doIsCompatible(const nvcvpy::IKey &that_) const override
        {
            const Key &that = static_cast<const Key &>(that_);
            return that.canBeUsedWith(m_maxShape, m_maxOctaveLayers);
        }

        int3 m_maxShape;
        int  m_maxOctaveLayers;
    };

    PyOpSIFT(const int3 &maxShape, int maxOctaveLayers)
        : m_key(maxShape, maxOctaveLayers)
        , m_op(maxShape, maxOctaveLayers)
    {
    }

    template<class... AA>
    void submit(AA &&...args)
    {
        m_op(std::forward<AA>(args)...);
    }

    py::object container() const override
    {
        return *this;
    }

    const nvcvpy::IKey &key() const override
    {
        return m_key;
    }

    static std::shared_ptr<nvcvpy::ICacheItem> fetch(std::vector<std::shared_ptr<nvcvpy::ICacheItem>> &cache)
    {
        std::shared_ptr<nvcvpy::ICacheItem> retItem = cache[0];

        long long int maxPayloadSize = 0;

        for (const auto &item : cache)
        {
            const Key &key = static_cast<const Key &>(item.get()->key());

            long long int keyPayloadSize = key.payloadSize();

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
    Key          m_key;
    cvcuda::SIFT m_op;
};

// Auxiliary function to get tensor access for input tensor in
auto tensorAccess(Tensor &in)
{
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (!inData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be a valid CUDA strided tensor");
    }
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be a valid image-based tensor");
    }
    return inAccess;
}

TupleTensor4 SIFTInto(Tensor &featCoords, Tensor &featMetadata, Tensor &featDescriptors, Tensor &numFeatures,
                      Tensor &in, int numOctaveLayers, float contrastThreshold, float edgeThreshold, float initSigma,
                      NVCVSIFTFlagType flags, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto inAccess = tensorAccess(in);
    int3 inShape{(int)inAccess->numCols(), (int)inAccess->numRows(), (int)inAccess->numSamples()};
    if (flags == NVCV_SIFT_USE_EXPANDED_INPUT)
    {
        inShape.x *= 2;
        inShape.y *= 2;
    }

    auto op = CreateOperatorEx<PyOpSIFT>(inShape, numOctaveLayers);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {in});
    guard.add(LockMode::LOCK_MODE_WRITE, {featCoords, featMetadata, featDescriptors, numFeatures});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*op});

    op->submit(pstream->cudaHandle(), in, featCoords, featMetadata, featDescriptors, numFeatures, numOctaveLayers,
               contrastThreshold, edgeThreshold, initSigma, flags);

    return TupleTensor4(std::move(featCoords), std::move(featMetadata), std::move(featDescriptors),
                        std::move(numFeatures));
}

// Get default number of maximum features given width and height (5% of total pixels at a minimum of 1)
inline int GetDefaultMaxFeatures(int width, int height)
{
    return std::max(width * height / 20, 1);
}

TupleTensor4 SIFT(Tensor &in, int maxFeatures, int numOctaveLayers, float contrastThreshold, float edgeThreshold,
                  float initSigma, NVCVSIFTFlagType flags, std::optional<Stream> pstream)
{
    auto inAccess   = tensorAccess(in);
    int  numSamples = inAccess->numSamples();

    maxFeatures = maxFeatures == 0 ? GetDefaultMaxFeatures(inAccess->numCols(), inAccess->numRows()) : maxFeatures;

    // Row align must be 1 in below tensors so last 2 dimensions are packed

    // clang-format off

    Tensor featCoords      = Tensor::Create({{numSamples, maxFeatures, 4}, "NMC"}, nvcv::TYPE_F32, 1);
    Tensor featMetadata    = Tensor::Create({{numSamples, maxFeatures, 3}, "NMC"}, nvcv::TYPE_F32, 1);
    Tensor featDescriptors = Tensor::Create({{numSamples, maxFeatures, 128}, "NMD"}, nvcv::TYPE_U8, 1);
    Tensor numFeatures     = Tensor::Create({{numSamples, 1}, "NC"}, nvcv::TYPE_S32, 1);

    // clang-format on

    return SIFTInto(featCoords, featMetadata, featDescriptors, numFeatures, in, numOctaveLayers, contrastThreshold,
                    edgeThreshold, initSigma, flags, pstream);
}

} // namespace

void ExportOpSIFT(py::module &m)
{
    using namespace pybind11::literals;

    m.def("sift", &SIFT, "src"_a, "max_features"_a = 0, "num_octave_layers"_a = 3, "contrast_threshold"_a = 0.03f,
          "edge_threshold"_a = 10.f, "init_sigma"_a = 1.6f, "flags"_a = NVCV_SIFT_USE_EXPANDED_INPUT, py::kw_only(),
          "stream"_a = nullptr, R"pbdoc(

        Executes the SIFT operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the SIFT operator
            for more details and usage examples.

        Args:
            src (nvcv.Tensor): Input tensor to extract features and compute descriptors from.
            max_features (Number, optional): Maximum number of features to be extracted, default is 5% of total
                                             pixels at a minimum of 1.
            num_octave_layers (Number, optional): Number of octave layers, default is 3.
            contrast_threshold (Number, optional): Contrast threshold, default is 0.03.
            edge_threshold (Number, optional): Edge threshold, default is 10.0.
            init_sigma (Number, optional): Initial sigma, default is 1.6.
            flags (cvcuda.SIFT, optional): Flag to whether to expand the input or not, default is
                                           cvcuda.SIFT.USE_EXPANDED_INPUT.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            Tuple[nvcv.Tensor, nvcv.Tensor, nvcv.Tensor, nvcv.Tensor]: A tuple with feature coordinates, metadata, descriptors and
            number of features.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");

    m.def("sift_into", &SIFTInto, "feat_coords"_a, "feat_metadata"_a, "feat_descriptors"_a, "num_features"_a, "src"_a,
          "num_octave_layers"_a = 3, "contrast_threshold"_a = 0.03f, "edge_threshold"_a = 10.f, "init_sigma"_a = 1.6f,
          "flags"_a = NVCV_SIFT_USE_EXPANDED_INPUT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(

        Executes the SIFT operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the SIFT operator
            for more details and usage examples.

        Args:
            feat_coords (nvcv.Tensor): Output tensor with feature coordinates.
            feat_metadata (nvcv.Tensor): Output tensor with feature metadata.
            feat_descriptors (nvcv.Tensor): Output tensor with feature descriptors.
            num_features (nvcv.Tensor): Output tensor with number of features.
            src (nvcv.Tensor): Input tensor to extract features and compute descriptors from.
            num_octave_layers (Number, optional): Number of octave layers, default is 3.
            contrast_threshold (Number, optional): Contrast threshold, default is 0.03.
            edge_threshold (Number, optional): Edge threshold, default is 10.0.
            init_sigma (Number, optional): Initial sigma, default is 1.6.
            flags (cvcuda.SIFT, optional): Flag to whether to expand the input or not, default is
                                           cvcuda.SIFT.USE_EXPANDED_INPUT.
            stream (nvcv.cuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            Tuple[nvcv.Tensor, nvcv.Tensor, nvcv.Tensor, nvcv.Tensor]: A tuple with feature coordinates, metadata, descriptors and
            number of features.

        Caution:
            Restrictions to several arguments may apply. Check the C
            API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
