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

#ifndef NVCV_CORE_PRIV_TLS_HPP
#define NVCV_CORE_PRIV_TLS_HPP

#include <nvcv/Status.h>

#include <array>
#include <exception>

namespace nvcv::priv {

struct CoreTLS // NOSONAR: TLS buffers stay flat so C API string-return helpers can reuse stable storage.
{
    NVCVStatus                                       lastErrorStatus;
    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> lastErrorMessage;

    std::array<char, 1024> bufColorSpecName;
    std::array<char, 128>  bufColorModelName;
    std::array<char, 128>  bufChromaLocationName;
    std::array<char, 128>  bufRawPatternName;
    std::array<char, 128>  bufColorSpaceName;
    std::array<char, 128>  bufColorTransferFunctionName;
    std::array<char, 128>  bufColorRangeName;
    std::array<char, 128>  bufWhitePointName;
    std::array<char, 128>  bufYCbCrEncodingName;
    std::array<char, 128>  bufChromaSubsamplingName;

    std::array<char, 128> bufDataKindName;
    std::array<char, 128> bufMemLayoutName;
    std::array<char, 128> bufChannelName;
    std::array<char, 128> bufSwizzleName;
    std::array<char, 128> bufByteOrderName;
    std::array<char, 128> bufPackingName;

    std::array<char, 1024> bufDataTypeName;
    std::array<char, 1024> bufImageFormatName;
    std::array<char, 1024> bufAlphaTypeName;
    std::array<char, 1024> bufExtraChannelTypeName;

    std::array<char, 128> bufResourceTypeName;
};

CoreTLS &GetCoreTLS() noexcept;

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_TLS_HPP
