/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "NativeBinding.hpp"
#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <cvcuda/OpSIFT.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cassert>
#include <memory>
#include <optional>
#include <vector>

namespace cvcudapy {

namespace {

using native::Stream;
using native::Tensor;
using TupleTensor4
    = std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>;
using TupleTensor4Ref = std::tuple<Tensor *, Tensor *, Tensor *, Tensor *>;

class PyOpSIFT : public OperatorResource
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
            const auto &that = static_cast<const Key &>(that_);
            return that.canBeUsedWith(m_maxShape, m_maxOctaveLayers);
        }

        int3 m_maxShape;
        int  m_maxOctaveLayers;
    };

    PyOpSIFT(const int3 &maxShape, int maxOctaveLayers)
        : m_op(maxShape, maxOctaveLayers)
    {
    }

    template<class... AA>
    void submit(AA &&...args)
    {
        m_op(std::forward<AA>(args)...);
    }

    static long long int cacheItemRank(const Key &key)
    {
        return key.payloadSize();
    }

private:
    cvcuda::SIFT m_op;
};

// Auxiliary function to get tensor access for input tensor in
auto tensorAccess(Tensor &in)
{
    auto inData = in.impl().exportData<nvcv::TensorDataStridedCuda>();
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

TupleTensor4Ref SIFTInto(Tensor &featCoords, Tensor &featMetadata, Tensor &featDescriptors, Tensor &numFeatures,
                         Tensor &in, int numOctaveLayers, float contrastThreshold, float edgeThreshold, float initSigma,
                         NVCVSIFTFlagType flags, std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto inAccess = tensorAccess(in);
    int3 inShape{inAccess->numCols(), inAccess->numRows(), static_cast<int>(inAccess->numSamples())};
    if (flags == NVCV_SIFT_USE_EXPANDED_INPUT)
    {
        inShape.x *= 2;
        inShape.y *= 2;
    }

    auto op = CreateOperatorEx<PyOpSIFT>(inShape, numOctaveLayers);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {in});
    guard.add(LockMode::LOCK_MODE_WRITE, {featCoords, featMetadata, featDescriptors, numFeatures});
    guard.add(LockMode::LOCK_MODE_READWRITE, {*op});

    guard.run(
        [&op, &stream, &in, &featCoords, &featMetadata, &featDescriptors, &numFeatures, &numOctaveLayers,
         &contrastThreshold, &edgeThreshold, &initSigma, &flags]()
        {
            op->submit(stream.handle(), in.impl(), featCoords.impl(), featMetadata.impl(), featDescriptors.impl(),
                       numFeatures.impl(), numOctaveLayers, contrastThreshold, edgeThreshold, initSigma, flags);
        });

    return {&featCoords, &featMetadata, &featDescriptors, &numFeatures};
}

// Get default number of maximum features given width and height (5% of total pixels at a minimum of 1)
inline int GetDefaultMaxFeatures(int width, int height)
{
    return std::max(width * height / 20, 1);
}

TupleTensor4 SIFT(Tensor &in, int maxFeatures, int numOctaveLayers, float contrastThreshold, float edgeThreshold,
                  float initSigma, NVCVSIFTFlagType flags, std::optional<Stream *> pstream)
{
    auto inAccess   = tensorAccess(in);
    auto numSamples = static_cast<int>(inAccess->numSamples());

    maxFeatures = maxFeatures == 0 ? GetDefaultMaxFeatures(inAccess->numCols(), inAccess->numRows()) : maxFeatures;

    // Row align must be 1 in below tensors so last 2 dimensions are packed

    // clang-format off

    auto featCoords      = native::CreateTensor({{numSamples, maxFeatures, 4}, "NMC"}, nvcv::TYPE_F32, 1);
    auto featMetadata    = native::CreateTensor({{numSamples, maxFeatures, 3}, "NMC"}, nvcv::TYPE_F32, 1);
    auto featDescriptors = native::CreateTensor({{numSamples, maxFeatures, 128}, "NMD"}, nvcv::TYPE_U8, 1);
    auto numFeatures     = native::CreateTensor({{numSamples, 1}, "NC"}, nvcv::TYPE_S32, 1);

    // clang-format on

    SIFTInto(*featCoords, *featMetadata, *featDescriptors, *numFeatures, in, numOctaveLayers, contrastThreshold,
             edgeThreshold, initSigma, flags, pstream);
    return {std::move(featCoords), std::move(featMetadata), std::move(featDescriptors), std::move(numFeatures)};
}

} // namespace

void ExportOpSIFT(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("sift", NvtxTrace("cvcuda.sift", &SIFT), "src"_a, "max_features"_a = 0, "num_octave_layers"_a = 3,
          "contrast_threshold"_a = 0.03f, "edge_threshold"_a = 10.f, "init_sigma"_a = 1.6f,
          "flags"_a = NVCV_SIFT_USE_EXPANDED_INPUT, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the SIFT operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor to extract features and compute descriptors from.
            max_features (Number, optional): Maximum number of features to be extracted, default is 5% of total
                                             pixels at a minimum of 1.
            num_octave_layers (Number, optional): Number of octave layers, default is 3.
            contrast_threshold (Number, optional): Contrast threshold, default is 0.03.
            edge_threshold (Number, optional): Edge threshold, default is 10.0.
            init_sigma (Number, optional): Initial sigma, default is 1.6.
            flags (cvcuda.SIFT, optional): Flag to whether to expand the input or not, default is
                                           cvcuda.SIFT.USE_EXPANDED_INPUT.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            Tuple[cvcuda.Tensor, cvcuda.Tensor, cvcuda.Tensor, cvcuda.Tensor]: A tuple with feature coordinates, metadata, descriptors and
            number of features.

    )pbdoc");

    m.def("sift_into", NvtxTrace("cvcuda.sift_into", &SIFTInto), "feat_coords"_a, "feat_metadata"_a,
          "feat_descriptors"_a, "num_features"_a, "src"_a, "num_octave_layers"_a = 3, "contrast_threshold"_a = 0.03f,
          "edge_threshold"_a = 10.f, "init_sigma"_a = 1.6f, "flags"_a = NVCV_SIFT_USE_EXPANDED_INPUT, py::kw_only(),
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Executes the SIFT operation on the given cuda stream.


        Args:
            feat_coords (cvcuda.Tensor): Output tensor with feature coordinates.
            feat_metadata (cvcuda.Tensor): Output tensor with feature metadata.
            feat_descriptors (cvcuda.Tensor): Output tensor with feature descriptors.
            num_features (cvcuda.Tensor): Output tensor with number of features.
            src (cvcuda.Tensor): Input tensor to extract features and compute descriptors from.
            num_octave_layers (Number, optional): Number of octave layers, default is 3.
            contrast_threshold (Number, optional): Contrast threshold, default is 0.03.
            edge_threshold (Number, optional): Edge threshold, default is 10.0.
            init_sigma (Number, optional): Initial sigma, default is 1.6.
            flags (cvcuda.SIFT, optional): Flag to whether to expand the input or not, default is
                                           cvcuda.SIFT.USE_EXPANDED_INPUT.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            Tuple[cvcuda.Tensor, cvcuda.Tensor, cvcuda.Tensor, cvcuda.Tensor]: A tuple with feature coordinates, metadata, descriptors and
            number of features.

    )pbdoc");
}

} // namespace cvcudapy
