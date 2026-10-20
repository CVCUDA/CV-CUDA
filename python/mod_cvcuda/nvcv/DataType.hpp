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

#ifndef NVCV_PYTHON_DATATYPE_HPP
#define NVCV_PYTHON_DATATYPE_HPP

#include <common/Binding.hpp>
#include <nvcv/DataType.hpp>

#include <optional>

namespace nvcv {
size_t ComputeHash(const nvcv::DataType &dtype);
}

namespace nvcvpy::priv {
namespace py = nanobind;

struct PyDataType
{
    nvcv::DataType value;
};

void ExportDataType(py::module_ &m);

nvcv::Packing MakeHomogeneousPacking(int bits, int numChannels);

std::optional<nvcv::DataType> ToNVCVDataType(const py::numpy_dtype &dt);
std::optional<nvcv::DataType> ToNVCVDataType(char kind, int itemSize);
py::numpy_dtype               ToDType(nvcv::DataType dtype);

// Normalize `dtype` to its canonical supported representation. Equivalent to
// ToNVCVDataType(ToDType(dtype)) without materializing a numpy dtype; returns
// nullopt when the type has no supported representation.
std::optional<nvcv::DataType> NormalizeDataType(const nvcv::DataType &dtype);

} // namespace nvcvpy::priv

namespace nanobind::detail {

template<>
struct NB_EXPORT type_caster<nvcv::DataType>
{
    NB_TYPE_CASTER(nvcv::DataType, const_name("cvcuda.Type"));

    bool          from_python(handle src, uint32_t flags, cleanup_list *cleanup) noexcept;
    static handle from_cpp(nvcv::DataType type, rv_policy policy, cleanup_list *cleanup) noexcept;
};

} // namespace nanobind::detail

#endif // NVCV_PYTHON_DATATYPE_HPP
