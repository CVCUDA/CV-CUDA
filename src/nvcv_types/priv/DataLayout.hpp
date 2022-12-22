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

#ifndef NVCV_FORMAT_PRIV_DATA_LAYOUT_HPP
#define NVCV_FORMAT_PRIV_DATA_LAYOUT_HPP

#include <nvcv/DataLayout.h>

#include <array>
#include <cstdint>
#include <iosfwd>
#include <optional>

namespace nvcv::priv {

std::optional<NVCVPacking> MakeNVCVPacking(const NVCVPackingParams &params) noexcept;
std::optional<NVCVPacking> MakeNVCVPacking(int bitsX, int bitsY = 0, int bitsZ = 0, int bitsW = 0) noexcept;
NVCVSwizzle MakeNVCVSwizzle(NVCVChannel x, NVCVChannel y = NVCV_CHANNEL_0, NVCVChannel z = NVCV_CHANNEL_0,
                            NVCVChannel w = NVCV_CHANNEL_0) noexcept;

bool IsSubWord(const NVCVPackingParams &p);

int GetBitsPerPixel(NVCVPacking packing) noexcept;

NVCVPackingParams GetPackingParams(NVCVPacking packing) noexcept;

int GetAlignment(NVCVPacking packing) noexcept;

NVCVChannel GetSwizzleChannel(NVCVSwizzle swizzle, int idx) noexcept;

std::array<NVCVChannel, 4> GetChannels(NVCVSwizzle swizzle) noexcept;

int GetNumChannels(NVCVSwizzle swizzle) noexcept;

int GetBlockHeightLog2(NVCVMemLayout memLayout) noexcept;

int GetNumComponents(NVCVPacking packing) noexcept;
int GetNumChannels(NVCVPacking packing) noexcept;

std::array<int32_t, 4> GetBitsPerComponent(NVCVPacking packing) noexcept;

NVCVSwizzle MergePlaneSwizzles(NVCVSwizzle sw0, NVCVSwizzle sw1 = NVCV_SWIZZLE_0000,
                               NVCVSwizzle sw2 = NVCV_SWIZZLE_0000, NVCVSwizzle sw3 = NVCV_SWIZZLE_0000);

// Flips byte order in memory space and return the resulting swizzle.
// Optionally an offset+length in component space can be specified, it'll
// restrict the flipping only to these components alone.
NVCVSwizzle FlipByteOrder(NVCVSwizzle swizzle, int off = 0, int len = 4) noexcept;

const char *GetName(NVCVDataKind dataKind);
const char *GetName(NVCVPacking packing);
const char *GetName(NVCVMemLayout memLayout);
const char *GetName(NVCVChannel swizzleChannel);
const char *GetName(NVCVSwizzle swizzle);
const char *GetName(NVCVByteOrder byteOrder);

} // namespace nvcv::priv

std::ostream &operator<<(std::ostream &out, NVCVDataKind dataKind);
std::ostream &operator<<(std::ostream &out, NVCVPacking packing);
std::ostream &operator<<(std::ostream &out, NVCVMemLayout memLayout);
std::ostream &operator<<(std::ostream &out, NVCVChannel swizzleChannel);
std::ostream &operator<<(std::ostream &out, NVCVSwizzle swizzle);
std::ostream &operator<<(std::ostream &out, NVCVByteOrder byteOrder);

#endif // NVCV_FORMAT_PRIV_DATA_LAYOUT_HPP
