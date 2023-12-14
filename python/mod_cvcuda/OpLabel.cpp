/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpLabel.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

#include <optional>

namespace cvcudapy {

using TupleTensor3 = std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>>;

namespace {

TupleTensor3 LabelInto(Tensor &output, std::optional<Tensor> count, std::optional<Tensor> stats, Tensor &input,
                       NVCVConnectivityType connectivity, NVCVLabelType assignLabels, std::optional<Tensor> bgLabel,
                       std::optional<Tensor> minThresh, std::optional<Tensor> maxThresh, std::optional<Tensor> minSize,
                       std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::Label>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*op});

    if (count)
    {
        guard.add(LockMode::LOCK_WRITE, {*count});
    }
    if (stats)
    {
        guard.add(LockMode::LOCK_WRITE, {*stats});
    }
    if (bgLabel)
    {
        guard.add(LockMode::LOCK_READ, {*bgLabel});
    }
    if (minThresh)
    {
        guard.add(LockMode::LOCK_READ, {*minThresh});
    }
    if (maxThresh)
    {
        guard.add(LockMode::LOCK_READ, {*maxThresh});
    }
    if (minSize)
    {
        guard.add(LockMode::LOCK_READ, {*minSize});
    }

    op->submit(pstream->cudaHandle(), input, output, (bgLabel ? *bgLabel : nvcv::Tensor{nullptr}),
               (minThresh ? *minThresh : nvcv::Tensor{nullptr}), (maxThresh ? *maxThresh : nvcv::Tensor{nullptr}),
               (minSize ? *minSize : nvcv::Tensor{nullptr}), (count ? *count : nvcv::Tensor{nullptr}),
               (stats ? *stats : nvcv::Tensor{nullptr}), connectivity, assignLabels);

    return TupleTensor3(std::move(output), count, stats);
}

TupleTensor3 Label(Tensor &input, NVCVConnectivityType connectivity, NVCVLabelType assignLabels, bool count, bool stats,
                   int maxLabels, std::optional<Tensor> bgLabel, std::optional<Tensor> minThresh,
                   std::optional<Tensor> maxThresh, std::optional<Tensor> minSize, std::optional<Stream> pstream)
{
    constexpr nvcv::DataType outType = nvcv::TYPE_U32;

    auto inputData = input.exportData<nvcv::TensorDataStridedCuda>();
    if (!inputData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be a valid CUDA strided tensor");
    }
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inputData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be a valid image-based tensor");
    }
    int numSamples = inAccess->numSamples();

    Tensor                output = Tensor::Create(input.shape(), outType);
    std::optional<Tensor> countTensor, statsTensor;

    if (count)
    {
        countTensor = Tensor::Create({{numSamples}, "N"}, outType);
    }
    if (stats)
    {
        int numStats = 1;
        if (connectivity == NVCV_CONNECTIVITY_4_2D || connectivity == NVCV_CONNECTIVITY_8_2D)
        {
            numStats = 6;
        }
        if (connectivity == NVCV_CONNECTIVITY_6_3D || connectivity == NVCV_CONNECTIVITY_26_3D)
        {
            numStats = 8;
        }

        statsTensor = Tensor::Create(
            {
                {numSamples, maxLabels, numStats},
                "NMA"
        },
            outType);
    }

    return LabelInto(output, countTensor, statsTensor, input, connectivity, assignLabels, bgLabel, minThresh, maxThresh,
                     minSize, pstream);
}

} // namespace

void ExportOpLabel(py::module &m)
{
    using namespace pybind11::literals;

    m.def("label", &Label, "src"_a, "connectivity"_a = NVCV_CONNECTIVITY_4_2D, "assign_labels"_a = NVCV_LABEL_FAST,
          py::kw_only(), "count"_a = false, "stats"_a = false, "max_labels"_a = 10000, "bg_label"_a = nullptr,
          "min_thresh"_a = nullptr, "max_thresh"_a = nullptr, "min_size"_a = nullptr, "stream"_a = nullptr, R"pbdoc(

        Executes the Label operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Label operator for more details and usage examples.

        Args:
            src (Tensor): Input tensor to label connected-component regions.
            connectivity (cvcuda.ConnectivityType, optional): Choice to control connectivity of input elements,
                                                              default is cvcuda.CONNECTIVITY_4_2D.
            assign_labels (cvcuda.LABEL, optional): Choice on how labels are assigned,
                                                    default is cvcuda.LABEL.FAST.
            count (bool, optional): Use True to return the count of valid labeled regions.
            stats (bool, optional): Use True to return the statistics of valid labeled regions.
            max_labels (Number, optional): Maximum number of labels to compute statistics for, default is 10000.
            bg_label (Tensor, optional): Background tensor to define input values to be considered background
                                         labels and thus ignored.
            min_thresh (Tensor, optional): Minimum threshold tensor to mask input values below it to be 0, and others 1.
            max_thresh (Tensor, optional): Maximum threshold tensor to mask input values above it to be 0, and others 1.
            min_size (Tensor, optional): Minimum size tensor to remove islands, i.e. labeled regions with number of
                                         elements less than the minimum size.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple wih output labels, count of regions and their statistics.
                                           The count or stats tensors may be None if theirs arguments are False.

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA operator.
    )pbdoc");

    m.def("label_into", &LabelInto, "dst"_a, "count"_a = nullptr, "stats"_a = nullptr, "src"_a,
          "connectivity"_a = NVCV_CONNECTIVITY_4_2D, "assign_labels"_a = NVCV_LABEL_FAST, py::kw_only(),
          "bg_label"_a = nullptr, "min_thresh"_a = nullptr, "max_thresh"_a = nullptr, "min_size"_a = nullptr,
          "stream"_a = nullptr, R"pbdoc(

        Executes the Label operation on the given cuda stream.

        See also:
            Refer to the CV-CUDA C API reference for the Label operator for more details and usage examples.

        Args:
            dst (Tensor): Output tensor with labels.
            count (Tensor, optional): Output tensor with count number of labeled regions.
            stats (Tensor, optional): Output tensor with statistics for each labeled region.
            src (Tensor): Input tensor to label connected-component regions.
            connectivity (cvcuda.ConnectivityType, optional): Choice to control connectivity of input elements,
                                                              default is cvcuda.CONNECTIVITY_4_2D.
            assign_labels (cvcuda.LABEL, optional): Choice on how labels are assigned,
                                                    default is cvcuda.LABEL.FAST.
            bg_label (Tensor, optional): Background tensor to define input values to be considered background
                                         labels and thus ignored.
            min_thresh (Tensor, optional): Minimum threshold tensor to mask input values below it to be 0, and others 1.
            max_thresh (Tensor, optional): Maximum threshold tensor to mask input values above it to be 0, and others 1.
            min_size (Tensor, optional): Minimum size tensor to remove islands, i.e. labeled regions with number of
                                         elements less than the minimum size.
            stream (Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple wih output labels, count of regions and their statistics.
                                           The count or stats tensors may be None if theirs arguments are None.

        Caution:
            Restrictions to several arguments may apply. Check the C API references of the CV-CUDA operator.
    )pbdoc");
}

} // namespace cvcudapy
