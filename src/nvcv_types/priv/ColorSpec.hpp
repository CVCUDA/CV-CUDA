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

#ifndef NVCV_FORMAT_PRIV_COLORSPEC_HPP
#define NVCV_FORMAT_PRIV_COLORSPEC_HPP

#include <nvcv/ColorSpec.h>

#include <memory> // for std::pair

namespace nvcv::priv {

struct ChromaLoc
{
    NVCVChromaLocation horiz, vert;
};

class ColorSpec
{
public:
    constexpr ColorSpec(NVCVColorSpec cspec)
        : m_cspec{cspec}
    {
    }

    ColorSpec(NVCVColorSpace cspace, NVCVYCbCrEncoding encoding, NVCVColorTransferFunction xferfunc,
              NVCVColorRange range, const ChromaLoc &loc) noexcept;

    operator NVCVColorSpec() const noexcept;

    ChromaLoc chromaLoc() const noexcept;
    ColorSpec chromaLoc(const ChromaLoc &newLoc) const;

    NVCVColorSpace colorSpace() const noexcept;
    ColorSpec      colorSpace(NVCVColorSpace newColorSpace) const;

    NVCVYCbCrEncoding YCbCrEncoding() const noexcept;
    ColorSpec         YCbCrEncoding(NVCVYCbCrEncoding newEncoding) const;

    NVCVColorTransferFunction xferFunc() const noexcept;
    ColorSpec                 xferFunc(NVCVColorTransferFunction newXferFunc) const;

    NVCVColorRange colorRange() const noexcept;
    ColorSpec      colorRange(NVCVColorRange range) const;

    NVCVWhitePoint whitePoint() const noexcept;

private:
    NVCVColorSpec m_cspec;
};

std::ostream &operator<<(std::ostream &out, ColorSpec cspec);

NVCVChromaSubsampling MakeNVCVChromaSubsampling(int samplesHoriz, int samplesVert);
std::pair<int, int>   GetChromaSamples(NVCVChromaSubsampling css);

bool NeedsColorspec(NVCVColorModel cmodel);

const char *GetName(NVCVColorModel colorModel);
const char *GetName(NVCVColorSpec colorSpec);
const char *GetName(NVCVChromaSubsampling chromaSub);
const char *GetName(NVCVColorTransferFunction xferFunc);
const char *GetName(NVCVYCbCrEncoding cstd);
const char *GetName(NVCVColorRange range);
const char *GetName(NVCVWhitePoint whitePoint);
const char *GetName(NVCVColorSpace color_space);
const char *GetName(NVCVChromaLocation loc);
const char *GetName(NVCVRawPattern raw);

} // namespace nvcv::priv

std::ostream &operator<<(std::ostream &out, NVCVColorModel colorModel);
std::ostream &operator<<(std::ostream &out, NVCVColorSpec colorSpec);
std::ostream &operator<<(std::ostream &out, NVCVChromaSubsampling chromaSub);
std::ostream &operator<<(std::ostream &out, NVCVColorTransferFunction xferFunc);
std::ostream &operator<<(std::ostream &out, NVCVYCbCrEncoding cstd);
std::ostream &operator<<(std::ostream &out, NVCVColorRange range);
std::ostream &operator<<(std::ostream &out, NVCVWhitePoint whitePoint);
std::ostream &operator<<(std::ostream &out, NVCVColorSpace color_space);
std::ostream &operator<<(std::ostream &out, NVCVChromaLocation loc);
std::ostream &operator<<(std::ostream &out, NVCVRawPattern raw);

// To be used inside gdb, as sometimes it has problems resolving the correct
// overload based on the parameter type.
std::string StrNVCVColorSpec(NVCVColorSpec cspec);

#endif // NVCV_FORMAT_PRIV_COLORSPEC_HPP
