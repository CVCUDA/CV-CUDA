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

#ifndef CVCUDA_TYPES_HPP
#define CVCUDA_TYPES_HPP

#include "SafeSize.hpp"

#include <cuda_runtime.h>
#include <cvcuda/Types.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace cvcuda::priv {

#define checkERR(call) check_error(call, #call, __LINE__, __FILE__) // NOSONAR: std::source_location is C++20.

inline static bool check_error(cudaError_t e, const char *call, int line, const char *file)
{
    if (e != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, cudaGetErrorString(e),
                cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

// Default font, user can install via below command:
//      sudo apt-get update
//      sudo apt-get install ttf-dejavu fonts-dejavu
inline constexpr char DEFAULT_OSD_FONT[] = "DejaVuSansMono";

inline size_t NVCVPolyLinePointCount(int32_t numPoints)
{
    return CheckedMulMany({2U, CheckedPositiveToSize(numPoints, "NVCVPolyLine numPoints")},
                          "NVCVPolyLine point count overflow");
}

inline size_t NVCVPolyLineByteCount(int32_t numPoints)
{
    return CheckedMulMany({NVCVPolyLinePointCount(numPoints), sizeof(int32_t)},
                          "NVCVPolyLine allocation size overflow");
}

class NVCVText
{
public:
    std::string   utf8Text;  // Text to draw in utf8 format.
    int32_t       fontSize;  // Font size for the text.
    std::string   fontName;  // Font name for the text.
    NVCVPointI    tlPos;     // Top-left corner point for label text, \ref NVCVPointI.
    NVCVColorRGBA fontColor; // Font color of the text.
    NVCVColorRGBA bgColor;   // Background color of text box.

    NVCVText(const char *_utf8Text, int32_t _fontSize, const char *_fontName, NVCVPointI _tlPos,
             NVCVColorRGBA _fontColor, NVCVColorRGBA _bgColor)
        : utf8Text(_utf8Text)
        , fontSize(_fontSize)
        , fontName(_fontName)
        , tlPos(_tlPos)
        , fontColor(_fontColor)
        , bgColor(_bgColor){};
};

class NVCVSegment
{
public:
    NVCVBoxI      box;            // Bounding box of segment, \ref NVCVBoxI.
    int32_t       thickness;      // Line thickness of segment outter rect.
    float        *dSeg = nullptr; // Device pointer for segment mask, cannot be nullptr.
                                  // Array length: segWidth * segHeight
                                  // Format:
                                  //      Score_00, Score_01, ..., Score_0k, ...
                                  //      Score_10, Score_11, ..., Score_kk, ...
                                  //          ... ,     ... , ...,     ... , ...
    int32_t       segWidth;       // Segment mask width.
    int32_t       segHeight;      // Segment mask height.
    float         segThreshold;   // Segment threshold.
    NVCVColorRGBA borderColor;    // Line color of segment outter rect.
    NVCVColorRGBA segColor;       // Segment mask color.

    NVCVSegment(NVCVBoxI _box, int32_t _thickness, const float *_hSeg, int32_t _segWidth, int32_t _segHeight,
                float _segThreshold, NVCVColorRGBA _borderColor, NVCVColorRGBA _segColor)
        : box(_box)
        , thickness(_thickness)
        , segWidth(_segWidth)
        , segHeight(_segHeight)
        , segThreshold(_segThreshold)
        , borderColor(_borderColor)
        , segColor(_segColor)
    {
        checkERR(cudaMalloc(&dSeg, static_cast<size_t>(segWidth) * segHeight * sizeof(float)));
        checkERR(
            cudaMemcpy(dSeg, _hSeg, static_cast<size_t>(segWidth) * segHeight * sizeof(float), cudaMemcpyHostToDevice));
    }

    NVCVSegment(const NVCVSegment &segment)
        : box(segment.box)
        , thickness(segment.thickness)
        , segWidth(segment.segWidth)
        , segHeight(segment.segHeight)
        , segThreshold(segment.segThreshold)
        , borderColor(segment.borderColor)
        , segColor(segment.segColor)
    {
        checkERR(cudaMalloc(&dSeg, static_cast<size_t>(segWidth) * segHeight * sizeof(float)));
        checkERR(cudaMemcpy(dSeg, segment.dSeg, static_cast<size_t>(segWidth) * segHeight * sizeof(float),
                            cudaMemcpyDeviceToDevice));
    }

    NVCVSegment &operator=(const NVCVSegment &) = delete;

    ~NVCVSegment()
    {
        if (dSeg != nullptr)
        {
            checkERR(cudaFree(dSeg));
            dSeg = nullptr;
        }
    };
};

class NVCVPolyLine
{
public:
    std::vector<int32_t> hPoints;           // Host polyline points' xy.
                                            // Array length: 2 * numPoints.
                                            // Format : X0, Y0, X1, Y1, ..., Xk, Yk, ...
    int32_t             *dPoints = nullptr; // Device pointer for polyline points' xy.
                                            // Can be nullptr only if fillColor.a == 0.
                                            // Array length: 2 * numPoints.
                                            // Format: X0, Y0, X1, Y1, ..., Xk, Yk, ...
    int32_t              numPoints;         // Number of polyline points.
    int32_t              thickness;         // Polyline thickness.
    bool                 isClosed;          // Connect p(0) to p(n-1) or not.
    NVCVColorRGBA        borderColor;       // Line color of polyline border.
    NVCVColorRGBA        fillColor;         // Fill color of poly fill area.
    bool                 interpolation;     // Default: true

    NVCVPolyLine(int32_t *_hPoints, int32_t _numPoints, int32_t _thickness, bool _isClosed, NVCVColorRGBA _borderColor,
                 NVCVColorRGBA _fillColor, bool _interpolation)
        : numPoints(_numPoints)
        , thickness(_thickness)
        , isClosed(_isClosed)
        , borderColor(_borderColor)
        , fillColor(_fillColor)
        , interpolation(_interpolation)
    {
        const size_t pointCount = NVCVPolyLinePointCount(numPoints);
        const size_t pointBytes = NVCVPolyLineByteCount(numPoints);
        hPoints.assign(_hPoints, _hPoints + pointCount);
        checkERR(cudaMalloc(&dPoints, pointBytes));
        checkERR(cudaMemcpy(dPoints, hPoints.data(), pointBytes, cudaMemcpyHostToDevice));
    }

    NVCVPolyLine(const NVCVPolyLine &pl)
        : hPoints(pl.hPoints)
        , numPoints(pl.numPoints)
        , thickness(pl.thickness)
        , isClosed(pl.isClosed)
        , borderColor(pl.borderColor)
        , fillColor(pl.fillColor)
        , interpolation(pl.interpolation)
    {
        const size_t pointBytes = NVCVPolyLineByteCount(numPoints);
        checkERR(cudaMalloc(&dPoints, pointBytes));
        checkERR(cudaMemcpy(dPoints, pl.dPoints, pointBytes, cudaMemcpyDeviceToDevice));
    }

    NVCVPolyLine &operator=(const NVCVPolyLine &) = delete;

    ~NVCVPolyLine()
    {
        if (dPoints != nullptr)
        {
            checkERR(cudaFree(dPoints));
            dPoints = nullptr;
        }
    };
};

class NVCVClock
{
public:
    NVCVClockFormat clockFormat; // Pre-defined clock format.
    long            time;        // Clock time.
    int32_t         fontSize;    // Font size.
    std::string     font;        // Font name.
    NVCVPointI      tlPos;       // Top-left corner point, \ref NVCVPointI.
    NVCVColorRGBA   fontColor;   // Font color of the text.
    NVCVColorRGBA   bgColor;     // Background color of text box.

    NVCVClock(NVCVClockFormat _clockFormat, long _time, int32_t _fontSize, const char *_font, NVCVPointI _tlPos,
              NVCVColorRGBA _fontColor, NVCVColorRGBA _bgColor)
        : clockFormat(_clockFormat)
        , time(_time)
        , fontSize(_fontSize)
        , font(_font)
        , tlPos(_tlPos)
        , fontColor(_fontColor)
        , bgColor(_bgColor){};
};

class NVCVElement
{
public:
    using Data = std::variant<std::monostate, NVCVBndBoxI, NVCVText, NVCVSegment, NVCVPoint, NVCVLine, NVCVPolyLine,
                              NVCVRotatedBox, NVCVCircle, NVCVArrow, NVCVClock>;

    explicit NVCVElement(NVCVOSDType osd_type);
    NVCVElement(NVCVOSDType osd_type, std::nullptr_t);

    template<typename ElementData>
    NVCVElement(NVCVOSDType osd_type, const ElementData *src);

    NVCVElement(const NVCVElement &)            = delete;
    NVCVElement &operator=(const NVCVElement &) = delete;
    ~NVCVElement()                              = default;

    NVCVOSDType type() const;
    Data       &data();

private:
    void setData(const NVCVBndBoxI &src);
    void setData(const NVCVText &src);
    void setData(const NVCVSegment &src);
    void setData(const NVCVPoint &src);
    void setData(const NVCVLine &src);
    void setData(const NVCVPolyLine &src);
    void setData(const NVCVRotatedBox &src);
    void setData(const NVCVCircle &src);
    void setData(const NVCVArrow &src);
    void setData(const NVCVClock &src);

    /*
        *  type:
        *      NVCV_OSD_RECT           -   \ref NVCVBndBoxI.
        *      NVCV_OSD_TEXT           -   \ref NVCVText.
        *      NVCV_OSD_SEGMENT        -   \ref NVCVSegment.
        *      NVCV_OSD_POINT          -   \ref NVCVPoint.
        *      NVCV_OSD_LINE           -   \ref NVCVLine.
        *      NVCV_OSD_POLYLINE       -   \ref NVCVPolyLine.
        *      NVCV_OSD_ROTATED_RECT   -   \ref NVCVRotatedBox.
        *      NVCV_OSD_CIRCLE         -   \ref NVCVCircle.
        *      NVCV_OSD_ARROW          -   \ref NVCVArrow.
        *      NVCV_OSD_CLOCK          -   \ref NVCVClock.
        */
    NVCVOSDType m_type; // OSD element type to draw.
    Data        m_data; // OSD element data.
};

inline NVCVElement::NVCVElement(NVCVOSDType osd_type)
    : m_type(osd_type)
{
}

inline NVCVElement::NVCVElement(NVCVOSDType osd_type, std::nullptr_t)
    : NVCVElement(osd_type)
{
}

template<typename ElementData>
inline NVCVElement::NVCVElement(NVCVOSDType osd_type, const ElementData *src)
    : NVCVElement(osd_type)
{
    if (src != nullptr)
    {
        setData(*src);
    }
}

inline void NVCVElement::setData(const NVCVBndBoxI &src)
{
    if (m_type == NVCVOSDType::NVCV_OSD_RECT)
    {
        m_data.emplace<NVCVBndBoxI>(src);
    }
}

inline void NVCVElement::setData(const NVCVText &src)
{
    if (m_type == NVCVOSDType::NVCV_OSD_TEXT)
    {
        m_data.emplace<NVCVText>(src);
    }
}

inline void NVCVElement::setData(const NVCVSegment &src)
{
    if (m_type == NVCVOSDType::NVCV_OSD_SEGMENT)
    {
        m_data.emplace<NVCVSegment>(src);
    }
}

inline void NVCVElement::setData(const NVCVPoint &src)
{
    if (m_type == NVCVOSDType::NVCV_OSD_POINT)
    {
        m_data.emplace<NVCVPoint>(src);
    }
}

inline void NVCVElement::setData(const NVCVLine &src)
{
    if (m_type == NVCVOSDType::NVCV_OSD_LINE)
    {
        m_data.emplace<NVCVLine>(src);
    }
}

inline void NVCVElement::setData(const NVCVPolyLine &src)
{
    if (m_type == NVCVOSDType::NVCV_OSD_POLYLINE)
    {
        m_data.emplace<NVCVPolyLine>(src);
    }
}

inline void NVCVElement::setData(const NVCVRotatedBox &src)
{
    if (m_type == NVCVOSDType::NVCV_OSD_ROTATED_RECT)
    {
        m_data.emplace<NVCVRotatedBox>(src);
    }
}

inline void NVCVElement::setData(const NVCVCircle &src)
{
    if (m_type == NVCVOSDType::NVCV_OSD_CIRCLE)
    {
        m_data.emplace<NVCVCircle>(src);
    }
}

inline void NVCVElement::setData(const NVCVArrow &src)
{
    if (m_type == NVCVOSDType::NVCV_OSD_ARROW)
    {
        m_data.emplace<NVCVArrow>(src);
    }
}

inline void NVCVElement::setData(const NVCVClock &src)
{
    if (m_type == NVCVOSDType::NVCV_OSD_CLOCK)
    {
        m_data.emplace<NVCVClock>(src);
    }
}

inline NVCVOSDType NVCVElement::type() const
{
    return m_type;
}

inline NVCVElement::Data &NVCVElement::data()
{
    return m_data;
}

class NVCVBlurBoxesImpl
{
public:
    explicit NVCVBlurBoxesImpl(const std::vector<std::vector<NVCVBlurBoxI>> &blurboxes_vec);
    NVCVBlurBoxesImpl(const NVCVBlurBoxesImpl &)            = delete;
    NVCVBlurBoxesImpl &operator=(const NVCVBlurBoxesImpl &) = delete;
    ~NVCVBlurBoxesImpl();

    int32_t      batch() const;
    int32_t      numBoxesAt(int32_t b) const;
    NVCVBlurBoxI boxAt(int32_t b, int32_t i) const;

private:
    std::vector<std::vector<NVCVBlurBoxI>> m_blurboxes_vec;
};

inline NVCVBlurBoxesImpl::NVCVBlurBoxesImpl(const std::vector<std::vector<NVCVBlurBoxI>> &blurboxes_vec)
    : m_blurboxes_vec(blurboxes_vec)
{
}

inline NVCVBlurBoxesImpl::~NVCVBlurBoxesImpl()
{
    std::vector<std::vector<NVCVBlurBoxI>> tmp;
    m_blurboxes_vec.swap(tmp);
}

inline int32_t NVCVBlurBoxesImpl::batch() const
{
    return static_cast<int32_t>(m_blurboxes_vec.size());
}

inline int32_t NVCVBlurBoxesImpl::numBoxesAt(int32_t b) const
{
    return static_cast<int32_t>(m_blurboxes_vec[b].size());
}

inline NVCVBlurBoxI NVCVBlurBoxesImpl::boxAt(int32_t b, int32_t i) const
{
    return m_blurboxes_vec[b][i];
}

class NVCVBndBoxesImpl
{
public:
    explicit NVCVBndBoxesImpl(const std::vector<std::vector<NVCVBndBoxI>> &bndboxes_vec);
    NVCVBndBoxesImpl(const NVCVBndBoxesImpl &)            = delete;
    NVCVBndBoxesImpl &operator=(const NVCVBndBoxesImpl &) = delete;
    ~NVCVBndBoxesImpl();

    int32_t     batch() const;
    int32_t     numBoxesAt(int32_t b) const;
    NVCVBndBoxI boxAt(int32_t b, int32_t i) const;

private:
    std::vector<std::vector<NVCVBndBoxI>> m_bndboxes_vec;
};

inline NVCVBndBoxesImpl::NVCVBndBoxesImpl(const std::vector<std::vector<NVCVBndBoxI>> &bndboxes_vec)
    : m_bndboxes_vec(bndboxes_vec)
{
}

inline NVCVBndBoxesImpl::~NVCVBndBoxesImpl()
{
    std::vector<std::vector<NVCVBndBoxI>> tmp;
    m_bndboxes_vec.swap(tmp);
}

inline int32_t NVCVBndBoxesImpl::batch() const
{
    return static_cast<int32_t>(m_bndboxes_vec.size());
}

inline int32_t NVCVBndBoxesImpl::numBoxesAt(int32_t b) const
{
    return static_cast<int32_t>(m_bndboxes_vec[b].size());
}

inline NVCVBndBoxI NVCVBndBoxesImpl::boxAt(int32_t b, int32_t i) const
{
    return m_bndboxes_vec[b][i];
}

class NVCVElementsImpl
{
public:
    explicit NVCVElementsImpl(const std::vector<std::vector<std::shared_ptr<NVCVElement>>> &elements_vec);
    NVCVElementsImpl(const NVCVElementsImpl &)            = delete;
    NVCVElementsImpl &operator=(const NVCVElementsImpl &) = delete;
    ~NVCVElementsImpl();

    int32_t                      batch() const;
    int32_t                      numElementsAt(int32_t b) const;
    std::shared_ptr<NVCVElement> elementAt(int32_t b, int32_t i) const;

private:
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> m_elements_vec;
};

inline NVCVElementsImpl::NVCVElementsImpl(const std::vector<std::vector<std::shared_ptr<NVCVElement>>> &elements_vec)
    : m_elements_vec(elements_vec)
{
}

inline NVCVElementsImpl::~NVCVElementsImpl()
{
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> tmp;
    m_elements_vec.swap(tmp);
}

inline int32_t NVCVElementsImpl::batch() const
{
    return static_cast<int32_t>(m_elements_vec.size());
}

inline int32_t NVCVElementsImpl::numElementsAt(int32_t b) const
{
    return static_cast<int32_t>(m_elements_vec[b].size());
}

inline std::shared_ptr<NVCVElement> NVCVElementsImpl::elementAt(int32_t b, int32_t i) const
{
    return m_elements_vec[b][i];
}

} // namespace cvcuda::priv

#endif // CVCUDA_TYPES_HPP
