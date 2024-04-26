
[//]: # "SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"
[//]: # ""
[//]: # "Licensed under the Apache License, Version 2.0 (the 'License');"
[//]: # "you may not use this file except in compliance with the License."
[//]: # "You may obtain a copy of the License at"
[//]: # "http://www.apache.org/licenses/LICENSE-2.0"
[//]: # ""
[//]: # "Unless required by applicable law or agreed to in writing, software"
[//]: # "distributed under the License is distributed on an 'AS IS' BASIS"
[//]: # "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied."
[//]: # "See the License for the specific language governing permissions and"
[//]: # "limitations under the License."


# CV-CUDA Developer Guide

## What is CV-CUDA?

CV-CUDA™ is an open-source, graphics processing unit (GPU)-accelerated library
for cloud-scale image processing and computer vision developed jointly by NVIDIA
and the ByteDance Applied Machine Learning teams.  CV-CUDA helps developers build
highly efficient pre- and post-processing pipelines that can improve throughput
by more than 10x while lowering cloud computing costs.

CV-CUDA includes:

- A unified, specialized set of high-performance CV and image processing kernels
- C, C++, and Python APIs
- Batching support, with variable shape images
- Zero-copy interfaces to PyTorch
- Sample applications

## What Pre- and Post-Processing Operators Are Included?

| Pre/Post-Processing Operators | Definition |
|-------------------------------|------------|
| Adaptive Thresholding | Chooses threshold based on smaller regions in the neighborhood of each pixel. |
| Advanced Color Format Conversions | Performs color conversion from interleaved RGB/BGR <-> YUV/YVU and semi planar. Supported standards: BT.601. BT.709. BT.2020 |
| AverageBlur | Reduces image noise using an average filter |
| BilateralFilter | Reduces image noise while preserving strong edges |
| Bounding Box | Draws an rectangular border using the X-Y coordinates and dimensions typically to define the location and size of an object in an image |
| Box Blurring | Overlays a blurred rectangle using the X-Y coordinates and dimensions that define the location and size of an object in an image |
| Brightness_Contrast | Adjusts brightness and contrast of an image |
| CenterCrop | Crops an image at its center |
| ChannelReorder | Shuffles the order of image channels |
| Color_Twist | Adjusts the hue saturation brightness and contrast of an image |
| Composite | Composites two images together |
| Conv2D | Convolves an image with a provided kernel |
| CopyMakeBorder | Creates a border around an image |
| CustomCrop | Crops an image with a given region-of-interest |
| CvtColor | Converts an image from one color space to another |
| DataTypeConvert | Converts an image’s data type, with optional scaling |
| Erase | Erases image regions |
| Flip | Flips a 2D image around its axis |
| GammaContrast | Adjusts image contrast |
| Gaussian | Applies a gaussian blur filter to the image |
| Gaussian Noise | Generates a statistical noise with a normal (Gaussian) distribution |
| Histogram | Provides a grayscale value distribution showing the frequency of occurrence of each gray value. |
| Histogram Equalizer | Allows effective spreading out the intensity range of the image typically used to improve contrast |
| HqResize | Performs advanced resizing supporting 2D and 3D data, tensors, tensor batches, and varshape image batches (2D only). Supports nearest neighbor, linear, cubic, Gaussian and Lanczos interpolation, with optional antialiasing when down-sampling |
| Inpainting | Performs inpainting by replacing a pixel by normalized weighted sum of all the known pixels in the neighborhood |
| Joint Bilateral Filter | Reduces image noise while preserving strong edges based on a guidance image |
| Label | Labels connected regions in an image using 4-way connectivity for foreground and 8-way for background pixels |
| Laplacian | Applies a Laplace transform to an image |
| MedianBlur | Reduces an image’s salt-and-pepper noise |
| MinArea Rect | Finds the minimum area rotated rectangle typically used to draw bounding rectangle with minimum area |
| MinMaxLoc | Finds the maximum and minimum values in a given array |
| Morphology | Performs morphological erode and dilate transformations |
| Morphology (close) | Performs morphological operation that involves dilation followed by erosion on an image |
| Morphology (open) | Performs morphological operation that involves erosion followed by dilation on an image |
| Non-Maximum Suppression | Enables selecting a single entity out of many overlapping ones typically used for selecting from multiple bounding boxes during object detection |
| Normalize | Normalizes an image pixel’s range |
| OSD (Polyline Line Text Rotated Rect Segmented Mask) | Displays an overlay on the image of different forms including polyline line text rotated rectangle segmented mask |
| PadStack | Stacks several images into a tensor with border extension |
| PairwiseMatcher | Matches features computed separately (e.g. via the SIFT operator) in two images, e.g. using the brute force method |
| PillowResize | Changes the size and scale of an image using python-pillow algorithm |
| RandomResizedCrop | Crops a random portion of an image and resizes it to a specified size. |
| Reformat | Converts a planar image into non-planar and vice versa |
| Remap | Maps pixels in an image with one projection to another projection in a new image. |
| Resize | Changes the size and scale of an image |
| Rotate | Rotates a 2D array in multiples of 90 degrees |
| SIFT | Identifies and describes features in images that are invariant to scale rotation and affine distortion. |
| Thresholding | Chooses a global threshold value that is the same for all pixels across the image. |
| WarpAffine | Applies an affine transformation to an image |
| WarpPerspective | Applies a perspective transformation to an image |

## Where Are the Release Notes?

CV-CUDA release notes can be
found [here](https://github.com/CVCUDA/CV-CUDA/releases)

## Where Can I Get Help?

An awesome product requires excellent support. File requests for enhancements and bug reports
[here](https://github.com/CVCUDA/CV-CUDA/issues/new/choose).

We are providing limited, direct, support to select enterprises using CV-CUDA.
To apply for direct enterprise developer engagement from NVIDIA , please fill
out the early access developer application
[here](http://developer.nvidia.com/cv-cuda/early-access).

## What Other Computer Vision Products Does NVIDIA Offer?

NVIDIA offers a number of computer vision products

In addition to cloud-scale computer vision and image processing, NVIDIA offers:

- [DALI](https://developer.nvidia.com/dali) (Data Loading Library), a portable,
  holistic framework for accelerated data loading and augmentation in deep
  learning workflows involving images, videos, and audio data.
- [VPI](https://developer.nvidia.com/embedded/vpi) (Vision Programming
  Interface), an accelerated computer vision and image processing software
  library primarily for embedded/edge applications.
- [cuCIM](https://developer.nvidia.com/multidimensional-image-processing)
  (Compute Unified Device Architecture Clara Image), an open source,
  accelerated computer vision and image processing library for multidimensional
  images in biomedical, geospatial, material life science, and remote sensing
  use cases.
- [NPP](https://developer.nvidia.com/npp) (NVIDIA Performance Primitives), an
  image, signal, and video processing library that accelerates and performs
  domain-specific functions.

If you want to learn more about what computer vision solutions are available,
review the computer vision solutions landing page.

---

<font size="1">
<b>Notice</b>

The information provided in this specification is believed to be accurate and
reliable as of the date provided. However, NVIDIA Corporation (“NVIDIA”) does
not give any representations or warranties, expressed or implied, as to the
accuracy or completeness of such information. NVIDIA shall have no liability for
the consequences or use of such information or for any infringement of patents
or other rights of third parties that may result from its use. This publication
supersedes and replaces all other specifications for the product that may have
been previously supplied.

NVIDIA reserves the right to make corrections, modifications, enhancements,
improvements, and other changes to this specification, at any time and/or to
discontinue any product or service without notice. Customer should obtain the
latest relevant specification before placing orders and should verify that such
information is current and complete.

NVIDIA products are sold subject to the NVIDIA standard terms and conditions of
sale supplied at the time of order acknowledgement, unless otherwise agreed in
an individual sales agreement signed by authorized representatives of NVIDIA and
customer. NVIDIA hereby expressly objects to applying any customer general terms
and conditions with regards to the purchase of the NVIDIA product referenced in
this specification.

NVIDIA products are not designed, authorized or warranted to be suitable for use
in medical, military, aircraft, space or life support equipment, nor in
applications where failure or malfunction of the NVIDIA product can reasonably
be expected to result in personal injury, death or property or environmental
damage. NVIDIA accepts no liability for inclusion and/or use of NVIDIA products
in such equipment or applications and therefore such inclusion and/or use is at
customer’s own risk.

NVIDIA makes no representation or warranty that products based on these
specifications will be suitable for any specified use without further testing or
modification. Testing of all parameters of each product is not necessarily
performed by NVIDIA. It is customer’s sole responsibility to ensure the product
is suitable and fit for the application planned by customer and to do the
necessary testing for the application in order to avoid a default of the
application or the product. Weaknesses in customer’s product designs may affect
the quality and reliability of the NVIDIA product and may result in additional
or different conditions and/or requirements beyond those contained in this
specification. NVIDIA does not accept any liability related to any default,
damage, costs or problem which may be based on or attributable to: (i) the use
of the NVIDIA product in any manner that is contrary to this specification, or
(ii) customer product designs.

No license, either expressed or implied, is granted under any NVIDIA patent
right, copyright, or other NVIDIA intellectual property right under this
specification. Information published by NVIDIA regarding third-party products or
services does not constitute a license from NVIDIA to use such products or
services or a warranty or endorsement thereof. Use of such information may
require a license from a third party under the patents or other intellectual
property rights of the third party, or a license from NVIDIA under the patents
or other intellectual property rights of NVIDIA. Reproduction of information in
this specification is permissible only if reproduction is approved by NVIDIA in
writing, is reproduced without alteration, and is accompanied by all associated
conditions, limitations, and notices.

ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS, DIAGNOSTICS,
LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY, “MATERIALS”) ARE BEING
PROVIDED “AS IS.” NVIDIA MAKES NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR
OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED
WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR
PURPOSE. Notwithstanding any damages that customer might incur for any reason
whatsoever, NVIDIA’s aggregate and cumulative liability towards customer for the
products described herein shall be limited in accordance with the NVIDIA terms
and conditions of sale for the product.

<b>Trademarks</b>

NVIDIA, the NVIDIA logo, NVIDIA CV-CUDA, and NVIDIA TensorRT are trademarks
and/or registered trademarks of NVIDIA Corporation in the U.S. and other
countries. Other company and product names may be trademarks of the respective
companies with which they are associated.

<b>Copyright</b>

© 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
</font>
