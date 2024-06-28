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

#include "OsdElement.hpp"

#include <common/String.hpp>
#include <cuda_runtime.h>
#include <cvcuda/Types.h>
#include <cvcuda/priv/Types.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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

static NVCVPointI pytopoint(py::tuple point)
{
    if (point.size() > 2 || point.size() == 0)
        throw py::value_error("Invalid point size.");

    NVCVPointI ret;
    memset(&ret, 0, sizeof(ret));

    int *pr = (int *)&ret;
    for (size_t i = 0; i < point.size(); ++i)
    {
        pr[i] = point[i].cast<int>();
    }
    return ret;
}

static NVCVBoxI pytobox(py::tuple box)
{
    if (box.size() > 4 || box.size() == 0)
        throw py::value_error("Invalid box size.");

    NVCVBoxI ret;
    memset(&ret, 0, sizeof(ret));

    int *pr = (int *)&ret;
    for (size_t i = 0; i < box.size(); ++i)
    {
        pr[i] = box[i].cast<int>();
    }
    return ret;
}

static NVCVColorRGBA pytocolor(py::tuple color)
{
    if (color.size() > 4 || color.size() == 0)
        throw py::value_error("Invalid color size.");

    NVCVColorRGBA ret;
    memset(&ret, 0, sizeof(ret));
    ret.a = 255;

    unsigned char *pr = (unsigned char *)&ret;
    for (size_t i = 0; i < color.size(); ++i)
    {
        pr[i] = color[i].cast<unsigned char>();
    }
    return ret;
}

} // namespace

void ExportBoxBlur(py::module &m)
{
    using namespace py::literals;
    using namespace cvcuda::priv;

    py::class_<NVCVBlurBoxI>(m, "BlurBoxI", "BlurBoxI")
        .def(py::init(
                 [](py::tuple box, int kernelSize)
                 {
                     NVCVBlurBoxI blurbox;
                     blurbox.box        = pytobox(box);
                     blurbox.kernelSize = kernelSize;
                     return blurbox;
                 }),
             "box"_a, "kernelSize"_a)
        .def_readonly("box", &NVCVBlurBoxI::box, "Tuple describing a box: x-coordinate, y-coordinate, width, height.")
        .def_readonly("kernelSize", &NVCVBlurBoxI::kernelSize, "Kernel sizes of mean filter.");

    py::class_<NVCVBlurBoxesImpl, std::shared_ptr<NVCVBlurBoxesImpl>>(m, "BlurBoxesI")
        .def(py::init([](const std::vector<std::vector<NVCVBlurBoxI>> &blurboxes_vec)
                      { return std::make_shared<NVCVBlurBoxesImpl>(blurboxes_vec); }),
             "boxes"_a);
}

void ExportOSD(py::module &m)
{
    using namespace py::literals;
    using namespace cvcuda::priv;

    py::class_<NVCVBndBoxI>(m, "BndBoxI", "BndBoxI")
        .def(py::init(
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
        .def_readonly("box", &NVCVBndBoxI::box, "Tuple describing a box: x-coordinate, y-coordinate, width, height.")
        .def_readonly("thickness", &NVCVBndBoxI::thickness, "Border thickness of bounding box.")
        .def_readonly("borderColor", &NVCVBndBoxI::borderColor, "Border color of bounding box.")
        .def_readonly("fillColor", &NVCVBndBoxI::fillColor, "Filled color of bounding box.");

    py::class_<NVCVBndBoxesImpl, std::shared_ptr<NVCVBndBoxesImpl>>(m, "BndBoxesI")
        .def(py::init([](const std::vector<std::vector<NVCVBndBoxI>> &bndboxes_vec)
                      { return std::make_shared<NVCVBndBoxesImpl>(bndboxes_vec); }),
             "boxes"_a);

    py::class_<NVCVText, std::shared_ptr<NVCVText>>(m, "Label")
        .def(py::init(
                 [](const char *utf8Text, int32_t fontSize, const char *fontName, py::tuple tlPos, py::tuple fontColor,
                    py::tuple bgColor) {
                     return NVCVText(utf8Text, fontSize, fontName, pytopoint(tlPos), pytocolor(fontColor),
                                     pytocolor(bgColor));
                 }),
             "utf8Text"_a, "fontSize"_a, py::arg("fontName") = "DejaVuSansMono", "tlPos"_a, "fontColor"_a, "bgColor"_a);

    py::class_<NVCVSegment>(m, "Segment")
        .def(py::init(
                 [](py::tuple box, int32_t thickness, py::array_t<float> segArray, float segThreshold,
                    py::tuple borderColor, py::tuple segColor)
                 {
                     py::buffer_info hSeg = segArray.request();
                     if (hSeg.ndim != 2)
                     {
                         throw std::runtime_error("segArray dims must be 2!");
                     }

                     return NVCVSegment(pytobox(box), thickness, (float *)hSeg.ptr, hSeg.shape[0], hSeg.shape[1],
                                        segThreshold, pytocolor(borderColor), pytocolor(segColor));
                 }),
             "box"_a, "thickness"_a, "segArray"_a, "segThreshold"_a, "borderColor"_a, "segColor"_a);

    py::class_<NVCVPoint>(m, "Point", "Point")
        .def(py::init(
                 [](py::tuple centerPos, int32_t radius, py::tuple color)
                 {
                     NVCVPoint point;
                     point.centerPos = pytopoint(centerPos);
                     point.radius    = radius;
                     point.color     = pytocolor(color);
                     return point;
                 }),
             "centerPos"_a, "radius"_a, "color"_a)
        .def_readonly("centerPos", &NVCVPoint::centerPos, "Center point.")
        .def_readonly("radius", &NVCVPoint::radius, "Point size.")
        .def_readonly("color", &NVCVPoint::color, "Point color.");

    py::class_<NVCVLine>(m, "Line", "Line")
        .def(py::init(
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
        .def_readonly("pos0", &NVCVLine::pos0, "Start point.")
        .def_readonly("pos1", &NVCVLine::pos1, "End point.")
        .def_readonly("thickness", &NVCVLine::thickness, "Line thickness.")
        .def_readonly("color", &NVCVLine::color, "Line color.")
        .def_readonly("interpolation", &NVCVLine::interpolation, "Default: true.");

    py::class_<NVCVPolyLine>(m, "PolyLine")
        .def(py::init(
                 [](py::array_t<int> points, int32_t thickness, bool isClosed, py::tuple borderColor,
                    py::tuple fillColor, bool interpolation)
                 {
                     py::buffer_info points_info = points.request();
                     if (points_info.ndim != 2 || points_info.shape[1] != 2)
                     {
                         throw std::runtime_error("points dims and shape[1] must be 2!");
                     }

                     return NVCVPolyLine((int32_t *)points_info.ptr, points_info.shape[0], thickness, isClosed,
                                         pytocolor(borderColor), pytocolor(fillColor), interpolation);
                 }),
             "points"_a, "thickness"_a, "isClosed"_a, "borderColor"_a, "fillColor"_a, py::arg("interpolation") = true);

    py::class_<NVCVRotatedBox>(m, "RotatedBox", "RotatedBox")
        .def(py::init(
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
        .def_readonly("centerPos", &NVCVRotatedBox::centerPos, "Center point.")
        .def_readonly("width", &NVCVRotatedBox::width, "Box width.")
        .def_readonly("height", &NVCVRotatedBox::height, "Box height.")
        .def_readonly("yaw", &NVCVRotatedBox::yaw, "Box yaw.")
        .def_readonly("thickness", &NVCVRotatedBox::thickness, "Box border thickness.")
        .def_readonly("borderColor", &NVCVRotatedBox::borderColor, "Circle border color.")
        .def_readonly("bgColor", &NVCVRotatedBox::bgColor, "Circle filled color.")
        .def_readonly("interpolation", &NVCVRotatedBox::interpolation, "Default: false.");

    py::class_<NVCVCircle>(m, "Circle", "Circle")
        .def(py::init(
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
        .def_readonly("centerPos", &NVCVCircle::centerPos, "Center point.")
        .def_readonly("radius", &NVCVCircle::radius, "Circle radius.")
        .def_readonly("thickness", &NVCVCircle::thickness, "Circle thickness.")
        .def_readonly("borderColor", &NVCVCircle::borderColor, "Circle border color.")
        .def_readonly("bgColor", &NVCVCircle::bgColor, "Circle filled color.");

    py::class_<NVCVArrow>(m, "Arrow", "Arrow")
        .def(py::init(
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
        .def_readonly("pos0", &NVCVArrow::pos0, "Start point.")
        .def_readonly("pos1", &NVCVArrow::pos1, "End point.")
        .def_readonly("arrowSize", &NVCVArrow::arrowSize, "Arrow size.")
        .def_readonly("thickness", &NVCVArrow::thickness, "Arrow line thickness.")
        .def_readonly("color", &NVCVArrow::color, "Arrow line color.")
        .def_readonly("interpolation", &NVCVArrow::interpolation, "Default: false.");

    py::enum_<NVCVClockFormat>(m, "ClockFormat")
        .value("YYMMDD_HHMMSS", NVCVClockFormat::YYMMDD_HHMMSS)
        .value("YYMMDD", NVCVClockFormat::YYMMDD)
        .value("HHMMSS", NVCVClockFormat::HHMMSS);

    py::class_<NVCVClock>(m, "Clock")
        .def(py::init(
                 [](NVCVClockFormat clockFormat, long time, int32_t fontSize, const char *font, py::tuple tlPos,
                    py::tuple fontColor, py::tuple bgColor) {
                     return NVCVClock(clockFormat, time, fontSize, font, pytopoint(tlPos), pytocolor(fontColor),
                                      pytocolor(bgColor));
                 }),
             "clockFormat"_a, "time"_a, "fontSize"_a, py::arg("font") = "DejaVuSansMono", "tlPos"_a, "fontColor"_a,
             "bgColor"_a);

    py::class_<NVCVElementsImpl, std::shared_ptr<NVCVElementsImpl>>(m, "Elements")
        .def(py::init(
                 [](const std::vector<py::list> &elements_list_vec)
                 {
                     std::vector<std::vector<std::shared_ptr<NVCVElement>>> elements_vec;
                     for (const auto &elements_list : elements_list_vec)
                     {
                         std::vector<std::shared_ptr<NVCVElement>> curVec;
                         for (size_t i = 0; i < elements_list.size(); ++i)
                         {
                             std::shared_ptr<NVCVElement> element;
                             if (pybind11::isinstance<NVCVBndBoxI>(elements_list[i]))
                             {
                                 auto rect = elements_list[i].cast<NVCVBndBoxI>();
                                 element   = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &rect);
                             }
                             else if (pybind11::isinstance<NVCVText>(elements_list[i]))
                             {
                                 auto text = elements_list[i].cast<NVCVText>();
                                 element   = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_TEXT, &text);
                             }
                             else if (pybind11::isinstance<NVCVSegment>(elements_list[i]))
                             {
                                 auto segment = elements_list[i].cast<NVCVSegment>();
                                 element      = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_SEGMENT, &segment);
                             }
                             else if (pybind11::isinstance<NVCVPoint>(elements_list[i]))
                             {
                                 auto point = elements_list[i].cast<NVCVPoint>();
                                 element    = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_POINT, &point);
                             }
                             else if (pybind11::isinstance<NVCVLine>(elements_list[i]))
                             {
                                 auto line = elements_list[i].cast<NVCVLine>();
                                 element   = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_LINE, &line);
                             }
                             else if (pybind11::isinstance<NVCVPolyLine>(elements_list[i]))
                             {
                                 auto pl = elements_list[i].cast<NVCVPolyLine>();
                                 element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_POLYLINE, &pl);
                             }
                             else if (pybind11::isinstance<NVCVRotatedBox>(elements_list[i]))
                             {
                                 auto rb = elements_list[i].cast<NVCVRotatedBox>();
                                 element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_ROTATED_RECT, &rb);
                             }
                             else if (pybind11::isinstance<NVCVCircle>(elements_list[i]))
                             {
                                 auto circle = elements_list[i].cast<NVCVCircle>();
                                 element     = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_CIRCLE, &circle);
                             }
                             else if (pybind11::isinstance<NVCVArrow>(elements_list[i]))
                             {
                                 auto arrow = elements_list[i].cast<NVCVArrow>();
                                 element    = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_ARROW, &arrow);
                             }
                             else if (pybind11::isinstance<NVCVClock>(elements_list[i]))
                             {
                                 auto clock = elements_list[i].cast<NVCVClock>();
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
