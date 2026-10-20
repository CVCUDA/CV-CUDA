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

#include "OsdElement.hpp"

#include <common/Binding.hpp>
#include <common/String.hpp>
#include <cuda_runtime.h>
#include <cvcuda/Types.h>
#include <cvcuda/priv/Types.hpp>

#include <cstddef>
#include <stdexcept>

namespace cvcudapy {

#define checkRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)

inline static bool check_runtime(cudaError_t e, const char *call, int line, const char *file)
{
    if (e != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, cudaGetErrorString(e),
                cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

namespace {

class OsdElementError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

NVCVPointI pytopoint(py::tuple point)
{
    if (point.size() > 2 || point.empty())
        throw py::value_error("Invalid point size.");

    NVCVPointI ret;
    memset(&ret, 0, sizeof(ret));

    if (!point.empty())
    {
        ret.x = py::cast<int>(point[0]);
    }
    if (point.size() > 1)
    {
        ret.y = py::cast<int>(point[1]);
    }
    return ret;
}

NVCVBoxI pytobox(py::tuple box)
{
    if (box.size() > 4 || box.empty())
        throw py::value_error("Invalid box size.");

    NVCVBoxI ret;
    memset(&ret, 0, sizeof(ret));

    if (!box.empty())
    {
        ret.x = py::cast<int>(box[0]);
    }
    if (box.size() > 1)
    {
        ret.y = py::cast<int>(box[1]);
    }
    if (box.size() > 2)
    {
        ret.width = py::cast<int>(box[2]);
    }
    if (box.size() > 3)
    {
        ret.height = py::cast<int>(box[3]);
    }
    return ret;
}

NVCVColorRGBA pytocolor(py::tuple color)
{
    if (color.size() > 4 || color.empty())
        throw py::value_error("Invalid color size.");

    NVCVColorRGBA ret;
    memset(&ret, 0, sizeof(ret));
    ret.a = 255;

    auto *pr = reinterpret_cast<std::byte *>(&ret);
    for (size_t i = 0; i < color.size(); ++i)
    {
        pr[i] = static_cast<std::byte>(py::cast<unsigned char>(color[i]));
    }
    return ret;
}

// Inverse of pytopoint/pytobox/pytocolor.  Used for `def_property_readonly`
// accessors so nanobind-stubgen sees a Python-importable `tuple` instead of
// raw C-API type names like `NVCVPointI`.
py::tuple pointtotuple(const NVCVPointI &p)
{
    return py::make_tuple(p.x, p.y);
}

py::tuple boxtotuple(const NVCVBoxI &b)
{
    return py::make_tuple(b.x, b.y, b.width, b.height);
}

py::tuple colortotuple(const NVCVColorRGBA &c)
{
    return py::make_tuple(c.r, c.g, c.b, c.a);
}

} // namespace

void ExportBoxBlur(py::module_ &m)
{
    using namespace py::literals;
    using namespace cvcuda::priv;

    py::class_<NVCVBlurBoxI>(m, "BlurBoxI", "BlurBoxI", py::is_weak_referenceable())
        .def(py::new_(
                 [](py::tuple box, int kernelSize)
                 {
                     NVCVBlurBoxI blurbox;
                     blurbox.box        = pytobox(box);
                     blurbox.kernelSize = kernelSize;
                     return blurbox;
                 }),
             "box"_a, "kernelSize"_a)
        .def_prop_ro(
            "box", [](const NVCVBlurBoxI &self) { return boxtotuple(self.box); },
            "Tuple describing a box: x-coordinate, y-coordinate, width, height.")
        .def_ro("kernelSize", &NVCVBlurBoxI::kernelSize, "Kernel sizes of mean filter.");

    py::class_<NVCVBlurBoxesImpl>(m, "BlurBoxesI", py::is_weak_referenceable())
        .def(py::new_([](const std::vector<std::vector<NVCVBlurBoxI>> &blurboxes_vec)
                      { return std::make_shared<NVCVBlurBoxesImpl>(blurboxes_vec); }),
             "boxes"_a);
}

void ExportOSD(py::module_ &m)
{
    using namespace py::literals;
    using namespace cvcuda::priv;

    py::class_<NVCVBndBoxI>(m, "BndBoxI", "BndBoxI", py::is_weak_referenceable())
        .def(py::new_(
                 [](py::tuple box, int thickness, py::tuple borderColor, py::tuple fillColor)
                 {
                     NVCVBndBoxI bndbox;
                     bndbox.box         = pytobox(box);
                     bndbox.thickness   = thickness;
                     bndbox.borderColor = pytocolor(borderColor);
                     bndbox.fillColor   = pytocolor(fillColor);
                     return bndbox;
                 }),
             "box"_a, "thickness"_a, "borderColor"_a, "fillColor"_a)
        .def_prop_ro(
            "box", [](const NVCVBndBoxI &self) { return boxtotuple(self.box); },
            "Tuple describing a box: x-coordinate, y-coordinate, width, height.")
        .def_ro("thickness", &NVCVBndBoxI::thickness, "Border thickness of bounding box.")
        .def_prop_ro(
            "borderColor", [](const NVCVBndBoxI &self) { return colortotuple(self.borderColor); },
            "Border color of bounding box.")
        .def_prop_ro(
            "fillColor", [](const NVCVBndBoxI &self) { return colortotuple(self.fillColor); },
            "Filled color of bounding box.");

    py::class_<NVCVBndBoxesImpl>(m, "BndBoxesI", py::is_weak_referenceable())
        .def(py::new_([](const std::vector<std::vector<NVCVBndBoxI>> &bndboxes_vec)
                      { return std::make_shared<NVCVBndBoxesImpl>(bndboxes_vec); }),
             "boxes"_a);

    py::class_<NVCVText>(m, "Label", py::is_weak_referenceable())
        .def(py::new_(
                 [](const char *utf8Text, int32_t fontSize, const char *fontName, py::tuple tlPos, py::tuple fontColor,
                    py::tuple bgColor) {
                     return NVCVText(utf8Text, fontSize, fontName, pytopoint(tlPos), pytocolor(fontColor),
                                     pytocolor(bgColor));
                 }),
             "utf8Text"_a, "fontSize"_a, py::arg("fontName") = "DejaVuSansMono", "tlPos"_a, "fontColor"_a, "bgColor"_a);

    py::class_<NVCVSegment>(m, "Segment", py::is_weak_referenceable())
        .def(py::new_(
                 [](py::tuple box, int32_t thickness, const py::ndarray<float, py::c_contig, py::device::cpu> &segArray,
                    float segThreshold, py::tuple borderColor, py::tuple segColor)
                 {
                     if (segArray.ndim() != 2)
                     {
                         throw OsdElementError("segArray dims must be 2!");
                     }

                     return NVCVSegment(pytobox(box), thickness, segArray.data(),
                                        static_cast<int32_t>(segArray.shape(0)),
                                        static_cast<int32_t>(segArray.shape(1)), segThreshold, pytocolor(borderColor),
                                        pytocolor(segColor));
                 }),
             "box"_a, "thickness"_a, "segArray"_a, "segThreshold"_a, "borderColor"_a, "segColor"_a);

    py::class_<NVCVPoint>(m, "Point", "Point", py::is_weak_referenceable())
        .def(py::new_(
                 [](py::tuple centerPos, int32_t radius, py::tuple color)
                 {
                     NVCVPoint point;
                     point.centerPos = pytopoint(centerPos);
                     point.radius    = radius;
                     point.color     = pytocolor(color);
                     return point;
                 }),
             "centerPos"_a, "radius"_a, "color"_a)
        .def_prop_ro(
            "centerPos", [](const NVCVPoint &self) { return pointtotuple(self.centerPos); }, "Center point.")
        .def_ro("radius", &NVCVPoint::radius, "Point size.")
        .def_prop_ro(
            "color", [](const NVCVPoint &self) { return colortotuple(self.color); }, "Point color.");

    py::class_<NVCVLine>(m, "Line", "Line", py::is_weak_referenceable())
        .def(py::new_(
                 [](py::tuple pos0, py::tuple pos1, int32_t thickness, py::tuple color, bool interpolation)
                 {
                     NVCVLine line;
                     line.pos0          = pytopoint(pos0);
                     line.pos1          = pytopoint(pos1);
                     line.thickness     = thickness;
                     line.color         = pytocolor(color);
                     line.interpolation = interpolation;
                     return line;
                 }),
             "pos0"_a, "pos1"_a, "thickness"_a, "color"_a, py::arg("interpolation") = true)
        .def_prop_ro(
            "pos0", [](const NVCVLine &self) { return pointtotuple(self.pos0); }, "Start point.")
        .def_prop_ro(
            "pos1", [](const NVCVLine &self) { return pointtotuple(self.pos1); }, "End point.")
        .def_ro("thickness", &NVCVLine::thickness, "Line thickness.")
        .def_prop_ro(
            "color", [](const NVCVLine &self) { return colortotuple(self.color); }, "Line color.")
        .def_ro("interpolation", &NVCVLine::interpolation, "Default: true.");

    py::class_<NVCVPolyLine>(m, "PolyLine", py::is_weak_referenceable())
        .def(py::new_(
                 [](const py::ndarray<int, py::c_contig, py::device::cpu> &points, int32_t thickness, bool isClosed,
                    py::tuple borderColor, py::tuple fillColor, bool interpolation)
                 {
                     if (points.ndim() != 2 || points.shape(1) != 2)
                     {
                         throw OsdElementError("points dims and shape[1] must be 2!");
                     }

                     return NVCVPolyLine(points.data(), static_cast<int32_t>(points.shape(0)), thickness, isClosed,
                                         pytocolor(borderColor), pytocolor(fillColor), interpolation);
                 }),
             "points"_a, "thickness"_a, "isClosed"_a, "borderColor"_a, "fillColor"_a, py::arg("interpolation") = true);

    py::class_<NVCVRotatedBox>(m, "RotatedBox", "RotatedBox", py::is_weak_referenceable())
        .def(py::new_(
                 [](py::tuple centerPos, int32_t width, int32_t height, float yaw, int32_t thickness,
                    py::tuple borderColor, py::tuple bgColor, bool interpolation)
                 {
                     NVCVRotatedBox rb;
                     rb.centerPos     = pytopoint(centerPos);
                     rb.width         = width;
                     rb.height        = height;
                     rb.yaw           = yaw;
                     rb.thickness     = thickness;
                     rb.borderColor   = pytocolor(borderColor);
                     rb.bgColor       = pytocolor(bgColor);
                     rb.interpolation = interpolation;
                     return rb;
                 }),
             "centerPos"_a, "width"_a, "height"_a, "yaw"_a, "thickness"_a, "borderColor"_a, "bgColor"_a,
             py::arg("interpolation") = false)
        .def_prop_ro(
            "centerPos", [](const NVCVRotatedBox &self) { return pointtotuple(self.centerPos); }, "Center point.")
        .def_ro("width", &NVCVRotatedBox::width, "Box width.")
        .def_ro("height", &NVCVRotatedBox::height, "Box height.")
        .def_ro("yaw", &NVCVRotatedBox::yaw, "Box yaw.")
        .def_ro("thickness", &NVCVRotatedBox::thickness, "Box border thickness.")
        .def_prop_ro(
            "borderColor", [](const NVCVRotatedBox &self) { return colortotuple(self.borderColor); },
            "Rotated box border color.")
        .def_prop_ro(
            "bgColor", [](const NVCVRotatedBox &self) { return colortotuple(self.bgColor); },
            "Rotated box filled color.")
        .def_ro("interpolation", &NVCVRotatedBox::interpolation, "Default: false.");

    py::class_<NVCVCircle>(m, "Circle", "Circle", py::is_weak_referenceable())
        .def(py::new_(
                 [](py::tuple centerPos, int32_t radius, int32_t thickness, py::tuple borderColor, py::tuple bgColor)
                 {
                     NVCVCircle circle;
                     circle.centerPos   = pytopoint(centerPos);
                     circle.radius      = radius;
                     circle.thickness   = thickness;
                     circle.borderColor = pytocolor(borderColor);
                     circle.bgColor     = pytocolor(bgColor);
                     return circle;
                 }),
             "centerPos"_a, "radius"_a, "thickness"_a, "borderColor"_a, "bgColor"_a)
        .def_prop_ro(
            "centerPos", [](const NVCVCircle &self) { return pointtotuple(self.centerPos); }, "Center point.")
        .def_ro("radius", &NVCVCircle::radius, "Circle radius.")
        .def_ro("thickness", &NVCVCircle::thickness, "Circle thickness.")
        .def_prop_ro(
            "borderColor", [](const NVCVCircle &self) { return colortotuple(self.borderColor); },
            "Circle border color.")
        .def_prop_ro(
            "bgColor", [](const NVCVCircle &self) { return colortotuple(self.bgColor); }, "Circle filled color.");

    py::class_<NVCVArrow>(m, "Arrow", "Arrow", py::is_weak_referenceable())
        .def(py::new_(
                 [](py::tuple pos0, py::tuple pos1, int32_t arrowSize, int32_t thickness, py::tuple color,
                    bool interpolation)
                 {
                     NVCVArrow arrow;
                     arrow.pos0          = pytopoint(pos0);
                     arrow.pos1          = pytopoint(pos1);
                     arrow.arrowSize     = arrowSize;
                     arrow.thickness     = thickness;
                     arrow.color         = pytocolor(color);
                     arrow.interpolation = interpolation;
                     return arrow;
                 }),
             "pos0"_a, "pos1"_a, "arrowSize"_a, "thickness"_a, "color"_a, py::arg("interpolation") = false)
        .def_prop_ro(
            "pos0", [](const NVCVArrow &self) { return pointtotuple(self.pos0); }, "Start point.")
        .def_prop_ro(
            "pos1", [](const NVCVArrow &self) { return pointtotuple(self.pos1); }, "End point.")
        .def_ro("arrowSize", &NVCVArrow::arrowSize, "Arrow size.")
        .def_ro("thickness", &NVCVArrow::thickness, "Arrow line thickness.")
        .def_prop_ro(
            "color", [](const NVCVArrow &self) { return colortotuple(self.color); }, "Arrow line color.")
        .def_ro("interpolation", &NVCVArrow::interpolation, "Default: false.");

    py::enum_compat<NVCVClockFormat>(m, "ClockFormat")
        .value("YYMMDD_HHMMSS", NVCVClockFormat::YYMMDD_HHMMSS)
        .value("YYMMDD", NVCVClockFormat::YYMMDD)
        .value("HHMMSS", NVCVClockFormat::HHMMSS);

    py::class_<NVCVClock>(m, "Clock", py::is_weak_referenceable())
        .def(py::new_(
                 [](NVCVClockFormat clockFormat, long time, int32_t fontSize, const char *font, py::tuple tlPos,
                    py::tuple fontColor, py::tuple bgColor) {
                     return NVCVClock(clockFormat, time, fontSize, font, pytopoint(tlPos), pytocolor(fontColor),
                                      pytocolor(bgColor));
                 }),
             "clockFormat"_a, "time"_a, "fontSize"_a, py::arg("font") = "DejaVuSansMono", "tlPos"_a, "fontColor"_a,
             "bgColor"_a);

    py::class_<NVCVElementsImpl>(m, "Elements", py::is_weak_referenceable())
        .def(py::new_(
                 [](const std::vector<py::list> &elements_list_vec)
                 {
                     std::vector<std::vector<std::shared_ptr<NVCVElement>>> elements_vec;
                     for (const auto &elements_list : elements_list_vec)
                     {
                         std::vector<std::shared_ptr<NVCVElement>> curVec;
                         for (const auto &item : elements_list)
                         {
                             std::shared_ptr<NVCVElement> element;
                             if (nanobind::isinstance<NVCVBndBoxI>(item))
                             {
                                 auto rect = py::cast<NVCVBndBoxI>(item);
                                 element   = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &rect);
                             }
                             else if (nanobind::isinstance<NVCVText>(item))
                             {
                                 auto text = py::cast<NVCVText>(item);
                                 element   = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_TEXT, &text);
                             }
                             else if (nanobind::isinstance<NVCVSegment>(item))
                             {
                                 auto segment = py::cast<NVCVSegment>(item);
                                 element      = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_SEGMENT, &segment);
                             }
                             else if (nanobind::isinstance<NVCVPoint>(item))
                             {
                                 auto point = py::cast<NVCVPoint>(item);
                                 element    = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_POINT, &point);
                             }
                             else if (nanobind::isinstance<NVCVLine>(item))
                             {
                                 auto line = py::cast<NVCVLine>(item);
                                 element   = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_LINE, &line);
                             }
                             else if (nanobind::isinstance<NVCVPolyLine>(item))
                             {
                                 auto pl = py::cast<NVCVPolyLine>(item);
                                 element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_POLYLINE, &pl);
                             }
                             else if (nanobind::isinstance<NVCVRotatedBox>(item))
                             {
                                 auto rb = py::cast<NVCVRotatedBox>(item);
                                 element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_ROTATED_RECT, &rb);
                             }
                             else if (nanobind::isinstance<NVCVCircle>(item))
                             {
                                 auto circle = py::cast<NVCVCircle>(item);
                                 element     = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_CIRCLE, &circle);
                             }
                             else if (nanobind::isinstance<NVCVArrow>(item))
                             {
                                 auto arrow = py::cast<NVCVArrow>(item);
                                 element    = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_ARROW, &arrow);
                             }
                             else if (nanobind::isinstance<NVCVClock>(item))
                             {
                                 auto clock = py::cast<NVCVClock>(item);
                                 element    = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_CLOCK, &clock);
                             }
                             else
                             {
                                 element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_NONE, nullptr);
                             }
                             curVec.emplace_back(element);
                         }
                         elements_vec.emplace_back(curVec);
                     }

                     return std::make_shared<NVCVElementsImpl>(elements_vec);
                 }),
             "elements"_a);
}

} // namespace cvcudapy
