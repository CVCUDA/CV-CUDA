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
                       NVCVConnectivityType connectivity, NVCVLabelType assignLabels, NVCVLabelMaskType maskType,
                       const std::optional<Tensor> &bgLabel, const std::optional<Tensor> &minThresh,
                       const std::optional<Tensor> &maxThresh, const std::optional<Tensor> &minSize,
                       const std::optional<Tensor> &mask, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::Label>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});

    if (count)
    {
        guard.add(LockMode::LOCK_MODE_WRITE, {*count});
    }
    if (stats)
    {
        guard.add(LockMode::LOCK_MODE_WRITE, {*stats});
    }
    if (bgLabel)
    {
        guard.add(LockMode::LOCK_MODE_READ, {*bgLabel});
    }
    if (minThresh)
    {
        guard.add(LockMode::LOCK_MODE_READ, {*minThresh});
    }
    if (maxThresh)
    {
        guard.add(LockMode::LOCK_MODE_READ, {*maxThresh});
    }
    if (minSize)
    {
        guard.add(LockMode::LOCK_MODE_READ, {*minSize});
    }
    if (mask)
    {
        guard.add(LockMode::LOCK_MODE_READ, {*mask});
    }

    guard.run(
        [&bgLabel, &minThresh, &maxThresh, &minSize, &count, &stats, &mask, &op, &pstream, &input, &output,
         &connectivity, &assignLabels, &maskType]()
        {
            const nvcv::Tensor nullTensor{nullptr};
            op->submit(pstream->cudaHandle(), input, output, AsNvcvTensor(bgLabel, nullTensor),
                       AsNvcvTensor(minThresh, nullTensor), AsNvcvTensor(maxThresh, nullTensor),
                       AsNvcvTensor(minSize, nullTensor), AsNvcvTensor(count, nullTensor),
                       AsNvcvTensor(stats, nullTensor), AsNvcvTensor(mask, nullTensor), connectivity, assignLabels,
                       maskType);
        });

    return TupleTensor3(std::move(output), count, stats);
}

TupleTensor3 Label(Tensor &input, NVCVConnectivityType connectivity, NVCVLabelType assignLabels,
                   NVCVLabelMaskType maskType, bool count, bool stats, int maxLabels,
                   const std::optional<Tensor> &bgLabel, const std::optional<Tensor> &minThresh,
                   const std::optional<Tensor> &maxThresh, const std::optional<Tensor> &minSize,
                   const std::optional<Tensor> &mask, std::optional<Stream> pstream)
{
    constexpr nvcv::DataType outType = nvcv::TYPE_S32;

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
    auto numSamples = static_cast<int>(inAccess->numSamples());

    Tensor                output = Tensor::Create(input.shape(), outType);
    std::optional<Tensor> countTensor;
    std::optional<Tensor> statsTensor;

    if (count)
    {
        countTensor = Tensor::Create({{numSamples}, "N"}, outType);
    }
    if (stats)
    {
        int numStats = 1;
        if (connectivity == NVCV_CONNECTIVITY_4_2D || connectivity == NVCV_CONNECTIVITY_8_2D)
        {
            numStats = 7;
        }
        if (connectivity == NVCV_CONNECTIVITY_6_3D || connectivity == NVCV_CONNECTIVITY_26_3D)
        {
            numStats = 9;
        }

        statsTensor = Tensor::Create(
            {
                {numSamples, maxLabels, numStats},
                "NMA"
        },
            outType);
    }

    return LabelInto(output, countTensor, statsTensor, input, connectivity, assignLabels, maskType, bgLabel, minThresh,
                     maxThresh, minSize, mask, pstream);
}

} // namespace

void ExportOpLabel(py::module &m)
{
    using namespace pybind11::literals;

    py::enum_<NVCVLabelMaskType>(m, "LabelMaskType", py::arithmetic())
        .value("REMOVE_ISLANDS_OUTSIDE_MASK_ONLY", NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY)
        .export_values();

    m.def("label", NvtxTrace("cvcuda.label", &Label), "src"_a, "connectivity"_a = NVCV_CONNECTIVITY_4_2D,
          "assign_labels"_a = NVCV_LABEL_FAST, "mask_type"_a = NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY, py::kw_only(),
          "count"_a = false, "stats"_a = false, "max_labels"_a = 10000, "bg_label"_a = nullptr,
          "min_thresh"_a = nullptr, "max_thresh"_a = nullptr, "min_size"_a = nullptr, "mask"_a = nullptr,
          "stream"_a = nullptr, R"pbdoc(
        Executes the Label operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor to label connected-component regions.
            connectivity (cvcuda.ConnectivityType, optional): Choice to control connectivity of input elements,
                                                              default is cvcuda.CONNECTIVITY_4_2D.
            assign_labels (cvcuda.LABEL, optional): Choice on how labels are assigned,
                                                    default is cvcuda.LABEL.FAST.
            mask_type (cvcuda.LabelMaskType, optional): Choice on how the mask is used,
                                                        default is cvcuda.REMOVE_ISLANDS_OUTSIDE_MASK_ONLY.
            count (bool, optional): Use True to return the count of valid labeled regions.
            stats (bool, optional): Use True to return the statistics of valid labeled regions.
            max_labels (Number, optional): Maximum number of labels to compute statistics for, default is 10000.
            bg_label (cvcuda.Tensor, optional): Background tensor to define input values to be considered background
                                         labels and thus ignored.
            min_thresh (cvcuda.Tensor, optional): Minimum threshold tensor to mask input values below it to be 0, and others 1.
            max_thresh (cvcuda.Tensor, optional): Maximum threshold tensor to mask input values above it to be 0, and others 1.
            min_size (cvcuda.Tensor, optional): Minimum size tensor to remove islands, i.e. labeled regions with number of
                                         elements less than the minimum size.
            mask (cvcuda.Tensor, optional): Mask tensor, its behavior is controlled by \ref mask_type.  One choice is to
                                     control island removal in addition to \ref min_size, i.e. regions with at
                                     least one element inside the mask (non-zero values) are not removed in case
                                     mask_type is cvcuda.REMOVE_ISLANDS_OUTSIDE_MASK_ONLY.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            Tuple[cvcuda.Tensor, cvcuda.Tensor, cvcuda.Tensor]: A tuple with output labels, count of regions and their statistics.
                                           The count or stats tensors may be None if theirs arguments are False.

    )pbdoc");

    m.def("label_into", NvtxTrace("cvcuda.label_into", &LabelInto), "dst"_a, "count"_a = nullptr, "stats"_a = nullptr,
          "src"_a, "connectivity"_a = NVCV_CONNECTIVITY_4_2D, "assign_labels"_a = NVCV_LABEL_FAST,
          "mask_type"_a = NVCV_REMOVE_ISLANDS_OUTSIDE_MASK_ONLY, py::kw_only(), "bg_label"_a = nullptr,
          "min_thresh"_a = nullptr, "max_thresh"_a = nullptr, "min_size"_a = nullptr, "mask"_a = nullptr,
          "stream"_a = nullptr, R"pbdoc(
        Executes the Label operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor with labels.
            count (cvcuda.Tensor, optional): Output tensor with count number of labeled regions.
            stats (cvcuda.Tensor, optional): Output tensor with statistics for each labeled region.
            src (cvcuda.Tensor): Input tensor to label connected-component regions.
            connectivity (cvcuda.ConnectivityType, optional): Choice to control connectivity of input elements,
                                                              default is cvcuda.CONNECTIVITY_4_2D.
            assign_labels (cvcuda.LABEL, optional): Choice on how labels are assigned,
                                                    default is cvcuda.LABEL.FAST.
            mask_type (cvcuda.LabelMaskType, optional): Choice on how the mask is used,
                                                        default is cvcuda.REMOVE_ISLANDS_OUTSIDE_MASK_ONLY.
            bg_label (cvcuda.Tensor, optional): Background tensor to define input values to be considered background
                                         labels and thus ignored.
            min_thresh (cvcuda.Tensor, optional): Minimum threshold tensor to mask input values below it to be 0, and others 1.
            max_thresh (cvcuda.Tensor, optional): Maximum threshold tensor to mask input values above it to be 0, and others 1.
            min_size (cvcuda.Tensor, optional): Minimum size tensor to remove islands, i.e. labeled regions with number of
                                         elements less than the minimum size.
            mask (cvcuda.Tensor, optional): Mask tensor, its behavior is controlled by \ref mask_type.  One choice is to
                                     control island removal in addition to \ref min_size, i.e. regions with at
                                     least one element inside the mask (non-zero values) are not removed in case
                                     mask_type is cvcuda.REMOVE_ISLANDS_OUTSIDE_MASK_ONLY.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            Tuple[cvcuda.Tensor, cvcuda.Tensor, cvcuda.Tensor]: A tuple with output labels, count of regions and their statistics.
                                           The count or stats tensors may be None if theirs arguments are None.

    )pbdoc");
}

} // namespace cvcudapy
