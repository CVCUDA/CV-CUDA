/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <exception>

namespace nvcv::priv {

struct CoreTLS
{
    NVCVStatus lastErrorStatus;
    char       lastErrorMessage[NVCV_MAX_STATUS_MESSAGE_LENGTH];

    char bufColorSpecName[1024];
    char bufColorModelName[128];
    char bufChromaLocationName[128];
    char bufRawPatternName[128];
    char bufColorSpaceName[128];
    char bufColorTransferFunctionName[128];
    char bufColorRangeName[128];
    char bufWhitePointName[128];
    char bufYCbCrEncodingName[128];
    char bufChromaSubsamplingName[128];

    char bufDataKindName[128];
    char bufMemLayoutName[128];
    char bufChannelName[128];
    char bufSwizzleName[128];
    char bufByteOrderName[128];
    char bufPackingName[128];

    char bufDataTypeName[1024];
    char bufImageFormatName[1024];
};

CoreTLS &GetCoreTLS() noexcept;

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_TLS_HPP
