..
  # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. list-table:: List of operators
   :widths: 30 70
   :header-rows: 1

   * - Pre/Post-Processing Operators
     - Definition
   * - Adaptive Thresholding (:py:func:`cvcuda.adaptivethreshold`)
     - Chooses threshold based on smaller regions in the neighborhood of each pixel.
   * - Advanced Color Format Conversions (:py:func:`cvcuda.advcvtcolor`)
     - Performs color conversion from interleaved RGB/BGR <-> YUV/YVU and semi planar. Supported standards: BT.601. BT.709. BT.2020
   * - AverageBlur (:py:func:`cvcuda.averageblur`)
     - Reduces image noise using an average filter
   * - BilateralFilter (:py:func:`cvcuda.bilateral_filter`)
     - Reduces image noise while preserving strong edges
   * - Bounding Box (:py:func:`cvcuda.bndbox`)
     - Draws an rectangular border using the X-Y coordinates and dimensions typically to define the location and size of an object in an image
   * - Box Blurring (:py:func:`cvcuda.boxblur`)
     - Overlays a blurred rectangle using the X-Y coordinates and dimensions that define the location and size of an object in an image
   * - Brightness_Contrast (:py:func:`cvcuda.brightness_contrast`)
     - Adjusts brightness and contrast of an image
   * - CenterCrop (:py:func:`cvcuda.center_crop`)
     - Crops an image at its center
   * - ChannelReorder (:py:func:`cvcuda.channelreorder`)
     - Shuffles the order of image channels
   * - Color_Twist (:py:func:`cvcuda.color_twist`)
     - Adjusts the hue saturation brightness and contrast of an image
   * - Composite (:py:func:`cvcuda.composite`)
     - Composites two images together
   * - Conv2D (:py:func:`cvcuda.conv2d`)
     - Convolves an image with a provided kernel
   * - CopyMakeBorder (:py:func:`cvcuda.copymakeborder`)
     - Creates a border around an image
   * - CustomCrop (:py:func:`cvcuda.customcrop`)
     - Crops an image with a given region-of-interest
   * - CvtColor (:py:func:`cvcuda.cvtcolor`)
     - Converts an image from one color space to another
   * - DataTypeConvert (:py:func:`cvcuda.convertto`)
     - Converts an image's data type with optional scaling
   * - Erase (:py:func:`cvcuda.erase`)
     - Erases image regions
   * - Flip (:py:func:`cvcuda.flip`)
     - Flips a 2D image around its axis
   * - GammaContrast (:py:func:`cvcuda.gamma_contrast`)
     - Adjusts image contrast
   * - Gaussian (:py:func:`cvcuda.gaussian`)
     - Applies a gaussian blur filter to the image
   * - Gaussian Noise (:py:func:`cvcuda.gaussiannoise`)
     - Generates a statistical noise with a normal (Gaussian) distribution
   * - Histogram (:py:func:`cvcuda.histogram`)
     - Provides a grayscale value distribution showing the frequency of occurrence of each gray value.
   * - Histogram Equalizer (:py:func:`cvcuda.histogrameq`)
     - Allows effective spreading out the intensity range of the image typically used to improve contrast
   * - HqResize (:py:func:`cvcuda.hq_resize`)
     - Performs advanced resizing supporting 2D and 3D data, tensors, tensor batches, and varshape image batches (2D only). Supports nearest neighbor, linear, cubic, Gaussian and Lanczos interpolation, with optional antialiasing when down-sampling.
   * - Inpainting (:py:func:`cvcuda.inpaint`)
     - Performs inpainting by replacing a pixel by normalized weighted sum of all the known pixels in the neighborhood
   * - Joint Bilateral Filter (:py:func:`cvcuda.joint_bilateral_filter`)
     - Reduces image noise while preserving strong edges based on a guidance image
   * - Label (:py:func:`cvcuda.label`)
     - Labels connected regions in an image using 4-way connectivity for foreground and 8-way for background pixels
   * - Laplacian (:py:func:`cvcuda.laplacian`)
     - Applies a Laplace transform to an image
   * - MedianBlur (:py:func:`cvcuda.median_blur`)
     - Reduces an image's salt-and-pepper noise
   * - MinArea Rect (:py:func:`cvcuda.minarearect`)
     - Finds the minimum area rotated rectangle typically used to draw bounding rectangle with minimum area
   * - MinMaxLoc (:py:func:`cvcuda.min_max_loc`)
     - Finds the maximum and minimum values in a given array
   * - Morphology (:py:func:`cvcuda.morphology`)
     - Performs morphological erode and dilate transformations
   * - Non-Maximum Suppression (:py:func:`cvcuda.nms`)
     - Enables selecting a single entity out of many overlapping ones typically used for selecting from multiple bounding boxes during object detection
   * - Normalize (:py:func:`cvcuda.normalize`)
     - Normalizes an image pixel's range
   * - OSD (:py:func:`cvcuda.osd`)
     - Displays an overlay on the image of of different forms including polyline line text rotated rectangle segmented mask
   * - PadStack (:py:func:`cvcuda.padandstack`)
     - Stacks several images into a tensor with border extension
   * - PairwiseMatcher (:py:func:`cvcuda.match`)
     - Matches features computed separately (e.g. via the SIFT operator) in two images, e.g. using the brute force method
   * - PillowResize (:py:func:`cvcuda.pillowresize`)
     - Changes the size and scale of an image using python-pillow algorithm
   * - RandomResizedCrop (:py:func:`cvcuda.random_resized_crop`)
     - Crops a random portion of an image and resizes it to a specified size.
   * - Reformat (:py:func:`cvcuda.reformat`)
     - Converts a planar image into non-planar and vice versa
   * - Remap (:py:func:`cvcuda.remap`)
     - Maps pixels in an image with one projection to another projection in a new image.
   * - Resize (:py:func:`cvcuda.resize`)
     - Changes the size and scale of an image
   * - ResizeCropConvertReformat (:py:func:`cvcuda.resize_crop_convert_reformat`)
     - Performs fused Resize-Crop-Convert-Reformat sequence with optional channel reordering.
   * - Rotate (:py:func:`cvcuda.rotate`)
     - Rotates a 2D array in multiples of 90 degrees
   * - SIFT (:py:func:`cvcuda.sift`)
     - Identifies and matches features in images that are invariant to scale rotation and affine distortion.
   * - Stack (:py:func:`cvcuda.stack`)
     - Combines multiple images into a single batch tensor
   * - Thresholding (:py:func:`cvcuda.threshold`)
     - Chooses a global threshold value that is the same for all pixels across the image.
   * - WarpAffine (:py:func:`cvcuda.warp_affine`)
     - Applies an affine transformation to an image
   * - WarpPerspective (:py:func:`cvcuda.warp_perspective`)
     - Applies a perspective transformation to an image
