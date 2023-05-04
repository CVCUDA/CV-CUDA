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
#include <cvcuda/Types.h>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {

static NVCVBoxI pytobox(py::tuple box)
{
    if (box.size() > 4 || box.size() == 0)
        throw py::value_error("Invalid color size.");

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
        .def_readwrite("box", &NVCVBndBoxI::box, "Tuple describing a box: x-coordinate, y-coordinate, width, height.")
        .def_readwrite("thickness", &NVCVBndBoxI::thickness, "Border thickness of bounding box.")
        .def_readwrite("borderColor", &NVCVBndBoxI::borderColor, "Border color of bounding box.")
        .def_readwrite("fillColor", &NVCVBndBoxI::fillColor, "Filled color of bounding box.");

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
        .def_readwrite("batch", &NVCVBndBoxesI::batch, "Number of images in the image batch.")
        .def_readwrite("numBoxes", &NVCVBndBoxesI::numBoxes, "Number array of bounding boxes for image batch.")
        .def_readwrite("boxes", &NVCVBndBoxesI::boxes, "Bounding box array for image batch, \ref NVCVBndBoxI.");
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
        .def_readwrite("box", &NVCVBlurBoxI::box, "Tuple describing a box: x-coordinate, y-coordinate, width, height.")
        .def_readwrite("kernelSize", &NVCVBlurBoxI::kernelSize, "Kernel sizes of mean filter.");

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
        .def_readwrite("batch", &NVCVBlurBoxesI::batch, "Number of images in the image batch.")
        .def_readwrite("numBoxes", &NVCVBlurBoxesI::numBoxes, "Number array of blurring boxes for image batch.")
        .def_readwrite("boxes", &NVCVBlurBoxesI::boxes, "Blurring box array for image batch, \ref NVCVBlurBoxI.");
}

} // namespace cvcudapy
