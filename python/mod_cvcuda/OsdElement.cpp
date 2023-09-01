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

void ExportBndBox(py::module &m)
{
    using namespace py::literals;

    py::class_<NVCVBndBoxI>(m, "BndBoxI")
        .def(py::init([]() { return NVCVBndBoxI{}; }))
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

    py::class_<NVCVBndBoxesI>(m, "BndBoxesI")
        .def(py::init([]() { return NVCVBndBoxesI{}; }))
        .def(py::init(
                 [](std::vector<int> numBoxes_vec, std::vector<NVCVBndBoxI> bndboxes_vec)
                 {
                     NVCVBndBoxesI bndboxes;

                     bndboxes.batch    = numBoxes_vec.size();
                     bndboxes.numBoxes = new int[bndboxes.batch];
                     memcpy(bndboxes.numBoxes, numBoxes_vec.data(), numBoxes_vec.size() * sizeof(int));

                     int total_box_num = bndboxes_vec.size();
                     bndboxes.boxes    = new NVCVBndBoxI[total_box_num];
                     memcpy(bndboxes.boxes, bndboxes_vec.data(), bndboxes_vec.size() * sizeof(NVCVBndBoxI));

                     return bndboxes;
                 }),
             "numBoxes"_a, "boxes"_a)
        .def_readonly("batch", &NVCVBndBoxesI::batch, "Number of images in the image batch.")
        .def_readonly("numBoxes", &NVCVBndBoxesI::numBoxes, "Number array of bounding boxes for image batch.")
        .def_readonly("boxes", &NVCVBndBoxesI::boxes, "Bounding box array for image batch, \ref NVCVBndBoxI.");
}

void ExportBoxBlur(py::module &m)
{
    using namespace py::literals;

    py::class_<NVCVBlurBoxI>(m, "BlurBoxI")
        .def(py::init([]() { return NVCVBlurBoxI{}; }))
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

    py::class_<NVCVBlurBoxesI>(m, "BlurBoxesI")
        .def(py::init([]() { return NVCVBlurBoxesI{}; }))
        .def(py::init(
                 [](std::vector<int> numBoxes_vec, std::vector<NVCVBlurBoxI> blurboxes_vec)
                 {
                     NVCVBlurBoxesI blurboxes;

                     blurboxes.batch    = numBoxes_vec.size();
                     blurboxes.numBoxes = new int[blurboxes.batch];
                     memcpy(blurboxes.numBoxes, numBoxes_vec.data(), numBoxes_vec.size() * sizeof(int));

                     int total_box_num = blurboxes_vec.size();
                     blurboxes.boxes   = new NVCVBlurBoxI[total_box_num];
                     memcpy(blurboxes.boxes, blurboxes_vec.data(), blurboxes_vec.size() * sizeof(NVCVBlurBoxI));

                     return blurboxes;
                 }),
             "numBoxes"_a, "boxes"_a)
        .def_readonly("batch", &NVCVBlurBoxesI::batch, "Number of images in the image batch.")
        .def_readonly("numBoxes", &NVCVBlurBoxesI::numBoxes, "Number array of blurring boxes for image batch.")
        .def_readonly("boxes", &NVCVBlurBoxesI::boxes, "Blurring box array for image batch, \ref NVCVBlurBoxI.");
}

void ExportOSD(py::module &m)
{
    using namespace py::literals;

    py::class_<NVCVText>(m, "Label")
        .def(py::init([]() { return NVCVText{}; }))
        .def(py::init(
                 [](const char *utf8Text, int32_t fontSize, const char *fontName, py::tuple tlPos, py::tuple fontColor,
                    py::tuple bgColor)
                 {
                     NVCVText label;
                     label.utf8Text = (const char *)malloc(strlen(utf8Text));
                     memcpy(const_cast<char *>(label.utf8Text), utf8Text, strlen(utf8Text) + 1);
                     label.fontName = (const char *)malloc(strlen(fontName));
                     memcpy(const_cast<char *>(label.fontName), fontName, strlen(fontName) + 1);
                     label.fontSize  = fontSize;
                     label.tlPos     = pytopoint(tlPos);
                     label.fontColor = pytocolor(fontColor);
                     label.bgColor   = pytocolor(bgColor);
                     return label;
                 }),
             "utf8Text"_a, "fontSize"_a, py::arg("fontName") = "DejaVuSansMono", "tlPos"_a, "fontColor"_a, "bgColor"_a)
        .def_readonly("utf8Text", &NVCVText::utf8Text, "Label text in utf8 format.")
        .def_readonly("fontSize", &NVCVText::fontSize, "Font size of label text.")
        .def_readonly("fontName", &NVCVText::fontName, "Font name of label text, default: DejaVuSansMono.")
        .def_readonly("tlPos", &NVCVText::tlPos, "Top-left corner point for label text.")
        .def_readonly("fontColor", &NVCVText::fontColor, "Font color of label text.")
        .def_readonly("bgColor", &NVCVText::bgColor, "Back color of label text.");

    py::class_<NVCVSegment>(m, "Segment")
        .def(py::init([]() { return NVCVSegment{}; }))
        .def(py::init(
                 [](py::tuple box, int32_t thickness, py::array_t<float> segArray, float segThreshold,
                    py::tuple borderColor, py::tuple segColor)
                 {
                     NVCVSegment segment;
                     segment.box       = pytobox(box);
                     segment.thickness = thickness;

                     py::buffer_info hSeg = segArray.request();
                     if (hSeg.ndim != 2)
                     {
                         throw std::runtime_error("segArray dims must be 2!");
                     }
                     segment.segWidth  = hSeg.shape[0];
                     segment.segHeight = hSeg.shape[1];

                     checkRuntime(cudaMalloc(&segment.dSeg, segment.segWidth * segment.segHeight * sizeof(float)));
                     checkRuntime(cudaMemcpy(segment.dSeg, hSeg.ptr,
                                             segment.segWidth * segment.segHeight * sizeof(float),
                                             cudaMemcpyHostToDevice));

                     segment.segThreshold = segThreshold;
                     segment.borderColor  = pytocolor(borderColor);
                     segment.segColor     = pytocolor(segColor);
                     return segment;
                 }),
             "box"_a, "thickness"_a, "segArray"_a, "segThreshold"_a, "borderColor"_a, "segColor"_a)
        .def_readonly("box", &NVCVSegment::box, "Bounding box of segment.")
        .def_readonly("thickness", &NVCVSegment::thickness, "Line thickness of segment outter rect.")
        .def_readonly("dSeg", &NVCVSegment::dSeg, "Device pointer for segment mask.")
        .def_readonly("segWidth", &NVCVSegment::segWidth, "Segment mask width.")
        .def_readonly("segHeight", &NVCVSegment::segHeight, "Segment mask height.")
        .def_readonly("segThreshold", &NVCVSegment::segThreshold, "Segment threshold.")
        .def_readonly("borderColor", &NVCVSegment::borderColor, "Line color of segment outter rect.")
        .def_readonly("segColor", &NVCVSegment::segColor, "Segment mask color.");

    py::class_<NVCVPoint>(m, "Point")
        .def(py::init([]() { return NVCVPoint{}; }))
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

    py::class_<NVCVLine>(m, "Line")
        .def(py::init([]() { return NVCVLine{}; }))
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
        .def(py::init([]() { return NVCVPolyLine{}; }))
        .def(py::init(
                 [](py::array_t<int> points, int32_t thickness, bool isClosed, py::tuple borderColor,
                    py::tuple fillColor, bool interpolation)
                 {
                     NVCVPolyLine pl;

                     py::buffer_info points_info = points.request();
                     if (points_info.ndim != 2 || points_info.shape[1] != 2)
                     {
                         throw std::runtime_error("points dims and shape[1] must be 2!");
                     }

                     pl.numPoints = points_info.shape[0];
                     pl.hPoints   = new int[pl.numPoints * 2];
                     checkRuntime(cudaMalloc(&pl.dPoints, 2 * pl.numPoints * sizeof(int)));

                     memcpy(pl.hPoints, points_info.ptr, 2 * pl.numPoints * sizeof(int));
                     checkRuntime(cudaMemcpy(pl.dPoints, points_info.ptr, 2 * pl.numPoints * sizeof(int),
                                             cudaMemcpyHostToDevice));

                     pl.thickness     = thickness;
                     pl.isClosed      = isClosed;
                     pl.borderColor   = pytocolor(borderColor);
                     pl.fillColor     = pytocolor(fillColor);
                     pl.interpolation = interpolation;
                     return pl;
                 }),
             "points"_a, "thickness"_a, "isClosed"_a, "borderColor"_a, "fillColor"_a, py::arg("interpolation") = true)
        .def_readonly("hPoints", &NVCVPolyLine::hPoints, "Host pointer for polyline points.")
        .def_readonly("dPoints", &NVCVPolyLine::dPoints, "Device pointer for polyline points.")
        .def_readonly("numPoints", &NVCVPolyLine::numPoints, "Number of polyline points.")
        .def_readonly("thickness", &NVCVPolyLine::thickness, "Polyline thickness.")
        .def_readonly("isClosed", &NVCVPolyLine::isClosed, "Connect p(0) to p(n-1) or not.")
        .def_readonly("borderColor", &NVCVPolyLine::borderColor, "Line color of polyline border.")
        .def_readonly("fillColor", &NVCVPolyLine::fillColor, "Fill color of poly fill area.")
        .def_readonly("interpolation", &NVCVPolyLine::interpolation, "Default: true.");

    py::class_<NVCVRotatedBox>(m, "RotatedBox")
        .def(py::init([]() { return NVCVRotatedBox{}; }))
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

    py::class_<NVCVCircle>(m, "Circle")
        .def(py::init([]() { return NVCVCircle{}; }))
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

    py::class_<NVCVArrow>(m, "Arrow")
        .def(py::init([]() { return NVCVArrow{}; }))
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
        .def(py::init([]() { return NVCVClock{}; }))
        .def(py::init(
                 [](NVCVClockFormat clockFormat, long time, int32_t fontSize, const char *font, py::tuple tlPos,
                    py::tuple fontColor, py::tuple bgColor)
                 {
                     NVCVClock clock;
                     clock.clockFormat = clockFormat;
                     clock.time        = time;
                     clock.fontSize    = fontSize;
                     clock.font        = (const char *)malloc(strlen(font));
                     memcpy(const_cast<char *>(clock.font), font, strlen(font) + 1);
                     clock.tlPos     = pytopoint(tlPos);
                     clock.fontColor = pytocolor(fontColor);
                     clock.bgColor   = pytocolor(bgColor);
                     return clock;
                 }),
             "clockFormat"_a, "time"_a, "fontSize"_a, py::arg("font") = "DejaVuSansMono", "tlPos"_a, "fontColor"_a,
             "bgColor"_a)
        .def_readonly("clockFormat", &NVCVClock::clockFormat, "Pre-defined clock format.")
        .def_readonly("time", &NVCVClock::time, "Clock time.")
        .def_readonly("fontSize", &NVCVClock::fontSize, "Font size.")
        .def_readonly("font", &NVCVClock::font, "Font name, default: DejaVuSansMono.")
        .def_readonly("tlPos", &NVCVClock::tlPos, "Top-left corner point of the text.")
        .def_readonly("fontColor", &NVCVClock::fontColor, "Font color of the text.")
        .def_readonly("bgColor", &NVCVClock::bgColor, "Background color of text box.");

    py::class_<NVCVElement>(m, "Element")
        .def(py::init([]() { return NVCVElement{}; }))
        .def(py::init(
                 [](NVCVOSDType type, void *data)
                 {
                     NVCVElement element;
                     element.type = type;
                     element.data = data;
                     return element;
                 }),
             "type"_a, "data"_a)
        .def_readonly("type", &NVCVElement::type, "Element type.")
        .def_readonly("data", &NVCVElement::data, "Element data pointer.");

    py::class_<NVCVElements>(m, "Elements")
        .def(py::init([]() { return NVCVElements{}; }))
        .def(py::init(
                 [](std::vector<int> numElements_vec, py::tuple elements_list)
                 {
                     NVCVElements ctx;

                     ctx.batch       = numElements_vec.size();
                     ctx.numElements = new int[ctx.batch];
                     memcpy(ctx.numElements, numElements_vec.data(), numElements_vec.size() * sizeof(int));

                     int total_element_num = elements_list.size();
                     ctx.elements          = new NVCVElement[total_element_num];

                     for (size_t i = 0; i < elements_list.size(); ++i)
                     {
                         if (pybind11::isinstance<NVCVBndBoxI>(elements_list[i]))
                         {
                             ctx.elements[i].type = NVCVOSDType::NVCV_OSD_RECT;
                             ctx.elements[i].data = new NVCVBndBoxI();
                             auto bndbox          = elements_list[i].cast<NVCVBndBoxI>();
                             memcpy(ctx.elements[i].data, &bndbox, sizeof(NVCVBndBoxI));
                         }
                         else if (pybind11::isinstance<NVCVText>(elements_list[i]))
                         {
                             ctx.elements[i].type = NVCVOSDType::NVCV_OSD_TEXT;
                             ctx.elements[i].data = new NVCVText();
                             auto text            = elements_list[i].cast<NVCVText>();
                             memcpy(ctx.elements[i].data, &text, sizeof(NVCVText));
                         }
                         else if (pybind11::isinstance<NVCVSegment>(elements_list[i]))
                         {
                             ctx.elements[i].type = NVCVOSDType::NVCV_OSD_SEGMENT;
                             ctx.elements[i].data = new NVCVSegment();
                             auto segment         = elements_list[i].cast<NVCVSegment>();
                             memcpy(ctx.elements[i].data, &segment, sizeof(NVCVSegment));
                         }
                         else if (pybind11::isinstance<NVCVPoint>(elements_list[i]))
                         {
                             ctx.elements[i].type = NVCVOSDType::NVCV_OSD_POINT;
                             ctx.elements[i].data = new NVCVPoint();
                             auto point           = elements_list[i].cast<NVCVPoint>();
                             memcpy(ctx.elements[i].data, &point, sizeof(NVCVPoint));
                         }
                         else if (pybind11::isinstance<NVCVLine>(elements_list[i]))
                         {
                             ctx.elements[i].type = NVCVOSDType::NVCV_OSD_LINE;
                             ctx.elements[i].data = new NVCVLine();
                             auto line            = elements_list[i].cast<NVCVLine>();
                             memcpy(ctx.elements[i].data, &line, sizeof(NVCVLine));
                         }
                         else if (pybind11::isinstance<NVCVPolyLine>(elements_list[i]))
                         {
                             ctx.elements[i].type = NVCVOSDType::NVCV_OSD_POLYLINE;
                             ctx.elements[i].data = new NVCVPolyLine();
                             auto pl              = elements_list[i].cast<NVCVPolyLine>();
                             memcpy(ctx.elements[i].data, &pl, sizeof(NVCVPolyLine));
                         }
                         else if (pybind11::isinstance<NVCVRotatedBox>(elements_list[i]))
                         {
                             ctx.elements[i].type = NVCVOSDType::NVCV_OSD_ROTATED_RECT;
                             ctx.elements[i].data = new NVCVRotatedBox();
                             auto pl              = elements_list[i].cast<NVCVRotatedBox>();
                             memcpy(ctx.elements[i].data, &pl, sizeof(NVCVRotatedBox));
                         }
                         else if (pybind11::isinstance<NVCVCircle>(elements_list[i]))
                         {
                             ctx.elements[i].type = NVCVOSDType::NVCV_OSD_CIRCLE;
                             ctx.elements[i].data = new NVCVCircle();
                             auto circle          = elements_list[i].cast<NVCVCircle>();
                             memcpy(ctx.elements[i].data, &circle, sizeof(NVCVCircle));
                         }
                         else if (pybind11::isinstance<NVCVArrow>(elements_list[i]))
                         {
                             ctx.elements[i].type = NVCVOSDType::NVCV_OSD_ARROW;
                             ctx.elements[i].data = new NVCVArrow();
                             auto arrow           = elements_list[i].cast<NVCVArrow>();
                             memcpy(ctx.elements[i].data, &arrow, sizeof(NVCVArrow));
                         }
                         else if (pybind11::isinstance<NVCVClock>(elements_list[i]))
                         {
                             ctx.elements[i].type = NVCVOSDType::NVCV_OSD_CLOCK;
                             ctx.elements[i].data = new NVCVClock();
                             auto clock           = elements_list[i].cast<NVCVClock>();
                             memcpy(ctx.elements[i].data, &clock, sizeof(NVCVClock));
                         }
                         else
                         {
                             ctx.elements[i].type = NVCVOSDType::NVCV_OSD_NONE;
                         }
                     }

                     return ctx;
                 }),
             "numElements"_a, "elements"_a)
        .def_readonly("batch", &NVCVElements::batch, "Number of images in the image batch.")
        .def_readonly("numElements", &NVCVElements::numElements, "Number array of OSD elements for image batch.")
        .def_readonly("elements", &NVCVElements::elements, "OSD elements array for image batch, \ref NVCVElement.");
}

} // namespace cvcudapy
