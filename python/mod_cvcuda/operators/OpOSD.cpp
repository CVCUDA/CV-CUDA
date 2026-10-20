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
#include <cvcuda/OpOSD.hpp>
#include <cvcuda/priv/Types.hpp>

#include <memory>

namespace cvcudapy {

template<>
struct OperatorCacheUsesCompletionLease<cvcuda::OSD> : std::true_type
{
};

namespace {
using ElementsImpl = cvcuda::priv::NVCVElementsImpl;
using native::Stream;
using native::Tensor;

// Segment and PolyLine commands contain device pointers owned by Elements.
// Model that owner as a resource so the pending-release queue retains it until OSD completes.
class ElementsLifetimeResource final : public OperatorResource
{
public:
    explicit ElementsLifetimeResource(std::shared_ptr<ElementsImpl> elements)
        : m_elements(std::move(elements))
    {
    }

    std::mutex *submissionMutex() const override
    {
        return nullptr;
    }

private:
    std::shared_ptr<ElementsImpl> m_elements;
};

Tensor &OSDInto(Tensor &output, Tensor &input, const std::shared_ptr<ElementsImpl> &elements,
                std::optional<Stream *> pstream)
{
    Stream &stream = native::GetStream(pstream);

    auto op               = CreateOperator<cvcuda::OSD>();
    auto elementsLifetime = std::make_shared<ElementsLifetimeResource>(elements);

    ResourceGuard guard(stream);
    guard.add(LockMode::LOCK_MODE_READ, {input});
    guard.add(LockMode::LOCK_MODE_WRITE, {output});
    guard.add(LockMode::LOCK_MODE_NONE, {*op});
    guard.add(LockMode::LOCK_MODE_NONE, {*elementsLifetime});

    auto elementsHandle = static_cast<NVCVElements>(static_cast<void *>(elements.get()));
    guard.run([&op, &stream, &input, &output, elementsHandle]()
              { op->submit(stream.handle(), input.impl(), output.impl(), elementsHandle); });

    return output;
}

std::shared_ptr<Tensor> OSD(Tensor &input, const std::shared_ptr<ElementsImpl> &elements,
                            std::optional<Stream *> pstream)
{
    auto output = native::CreateTensor(input.impl().shape(), input.dtype());

    OSDInto(*output, input, elements, pstream);
    return output;
}

} // namespace

void ExportOpOSD(py::module_ &m)
{
    using namespace nanobind::literals;

    m.def("osd", NvtxTrace("cvcuda.osd", &OSD), "src"_a, "elements"_a, py::kw_only(), "stream"_a = nullptr, R"pbdoc(
        Executes the OSD operation on the given cuda stream.


        Args:
            src (cvcuda.Tensor): Input tensor containing one or more images.
            elements (cvcuda.Elements): OSD elements in reference to the input tensor.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor.

    )pbdoc");

    m.def("osd_into", NvtxTrace("cvcuda.osd_into", &OSDInto), "dst"_a, "src"_a, "elements"_a, py::kw_only(),
          "stream"_a = nullptr, py::return_value_policy::reference, R"pbdoc(
        Executes the OSD operation on the given cuda stream.


        Args:
            dst (cvcuda.Tensor): Output tensor to store the result of the operation.
            src (cvcuda.Tensor): Input tensor containing one or more images.
            elements (cvcuda.Elements): OSD elements in reference to the input tensor.
            stream (cvcuda.Stream, optional): CUDA Stream on which to perform the operation.

        Returns:
            cvcuda.Tensor: The output tensor (same as dst).
    )pbdoc");
}

} // namespace cvcudapy
