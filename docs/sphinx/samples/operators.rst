..
   # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   # SPDX-License-Identifier: Apache-2.0
   #
   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at
   #
   # http://www.apache.org/licenses/LICENSE-2.0
   #
   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.

:orphan:

.. _sample_operators:

Operator Samples Overview
=========================

Each operator sample reads an image from disk, runs a single CV-CUDA operator on the GPU,
and writes the result back to disk.

Resampling & Geometry
---------------------

:doc:`operators/center_crop` · :doc:`operators/copymakeborder` · :doc:`operators/customcrop` ·
:doc:`operators/flip` · :doc:`operators/hq_resize` · :doc:`operators/pillowresize` ·
:doc:`operators/random_resized_crop` · :doc:`operators/remap` · :doc:`operators/resize` ·
:doc:`operators/rotate` · :doc:`operators/warp_affine` · :doc:`operators/warp_perspective`

Filtering & Blur
----------------

:doc:`operators/averageblur` · :doc:`operators/bilateral_filter` · :doc:`operators/boxblur` ·
:doc:`operators/conv2d` · :doc:`operators/gaussian` · :doc:`operators/joint_bilateral_filter` ·
:doc:`operators/laplacian` · :doc:`operators/median_blur` · :doc:`operators/morphology`

Color & Photometric
-------------------

:doc:`operators/advcvtcolor` · :doc:`operators/brightness_contrast` ·
:doc:`operators/channelreorder` · :doc:`operators/clahe` · :doc:`operators/color_twist` ·
:doc:`operators/convertto` · :doc:`operators/cvtcolor` · :doc:`operators/gamma_contrast` ·
:doc:`operators/histogrameq` · :doc:`operators/normalize`

Thresholding & Segmentation
---------------------------

:doc:`operators/adaptivethreshold` · :doc:`operators/label` · :doc:`operators/threshold`

Noise, Restoration & Augmentation
----------------------------------

:doc:`operators/erase` · :doc:`operators/gaussiannoise` · :doc:`operators/inpaint`

Compositing & Drawing
---------------------

:doc:`operators/bndbox` · :doc:`operators/composite` · :doc:`operators/osd`

Layout & Preprocessing
-----------------------

:doc:`operators/crop_flip_normalize_reformat` · :doc:`operators/reformat` ·
:doc:`operators/resize_crop_convert_reformat` · :doc:`operators/stack`
