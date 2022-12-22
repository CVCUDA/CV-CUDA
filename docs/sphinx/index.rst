..
  # SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _cvcuda_doc_system:

CV-CUDA
============================

NVIDIA CV-CUDA™ is an open-source project for building cloud-scale `Artificial Intelligence (AI) imaging and Computer Vision (CV) <https://developer.nvidia.com/computer-vision>`_ applications. It uses graphics processing unit (GPU) acceleration to help developers build highly efficient pre- and post-processing pipelines. It can improve throughput by more than 10x while lowering cloud computing costs.

CV-CUDA includes:

*  A unified, specialized set of high-performance CV and image processing kernels
*  C, C++, and Python APIs
*  Batching support, with variable shape images
*  Zero-copy interfaces to PyTorch
*  Sample applications: object classification and image segmentation


.. image:: content/cvcuda_arch.jpg
   :width: 610



CV-CUDA Pre- and Post-Processing Operators
------------------

CV-CUDA offers more than 20 Computer Vision and Image Processing operators. Find the operator that is right for your workflow below.


.. csv-table::
   :file: content/cvcuda_oplist.csv
   :widths: 30, 70
   :header-rows: 1


Where Are the Release Notes?
------------------

An awesome product requires excellent support.  CV-CUDA release notes can be found `here <https://github.com/CVCUDA/CV-CUDA/releases/tag/v0.2.0-alpha>`_.


Where Can I Get Help?
------------------

File requests for enhancements and bug reports `here <https://github.com/CVCUDA/CV-CUDA/issues/new/choose>`_.


What Other Computer Vision Products Does NVIDIA Offer?
------------------

NVIDIA offers a number of products for accelerating computer vision and image processing applications. In addition to CV-CUDA, some of the others include:

*  `DALI <https://developer.nvidia.com/dali>`_ (Data Loading Library), a portable, holistic framework for accelerated data loading and augmentation in deep learning workflows involving images, videos, and audio data.
*  `VPI <https://developer.nvidia.com/embedded/vpi>`_ (Vision Programming Interface), an accelerated computer vision and image processing software library primarily for embedded/edge applications.
*  `cuCIM <https://developer.nvidia.com/multidimensional-image-processing>`_ (Compute Unified Device Architecture Clara Image), an open source, accelerated computer vision and image processing library for multidimensional images in biomedical, geospatial, material life science, and remote sensing use cases.
*  `NPP <https://developer.nvidia.com/npp>`_ (NVIDIA Performance Primitives), an image, signal, and video processing library that accelerates and performs domain-specific functions.

If you want to learn more about what computer vision solutions are available, review the `computer vision solutions landing page <https://developer.nvidia.com/computer-vision>`_.


Notice
--------------------
The information provided in this specification is believed to be accurate and reliable as of the date provided. However, NVIDIA Corporation (“NVIDIA”) does not give any representations or warranties, expressed or implied, as to the accuracy or completeness of such information. NVIDIA shall have no liability for the consequences or use of such information or for any infringement of patents or other rights of third parties that may result from its use. This publication supersedes and replaces all other specifications for the product that may have been previously supplied.

NVIDIA reserves the right to make corrections, modifications, enhancements, improvements, and other changes to this specification, at any time and/or to discontinue any product or service without notice. Customer should obtain the latest relevant specification before placing orders and should verify that such information is current and complete.

NVIDIA products are sold subject to the NVIDIA standard terms and conditions of sale supplied at the time of order acknowledgement, unless otherwise agreed in an individual sales agreement signed by authorized representatives of NVIDIA and customer. NVIDIA hereby expressly objects to applying any customer general terms and conditions with regards to the purchase of the NVIDIA product referenced in this specification.

NVIDIA products are not designed, authorized or warranted to be suitable for use in medical, military, aircraft, space or life support equipment, nor in applications where failure or malfunction of the NVIDIA product can reasonably be expected to result in personal injury, death or property or environmental damage. NVIDIA accepts no liability for inclusion and/or use of NVIDIA products in such equipment or applications and therefore such inclusion and/or use is at customer’s own risk.

NVIDIA makes no representation or warranty that products based on these specifications will be suitable for any specified use without further testing or modification. Testing of all parameters of each product is not necessarily performed by NVIDIA. It is customer’s sole responsibility to ensure the product is suitable and fit for the application planned by customer and to do the necessary testing for the application in order to avoid a default of the application or the product. Weaknesses in customer’s product designs may affect the quality and reliability of the NVIDIA product and may result in additional or different conditions and/or requirements beyond those contained in this specification. NVIDIA does not accept any liability related to any default, damage, costs or problem which may be based on or attributable to: (i) the use of the NVIDIA product in any manner that is contrary to this specification, or (ii) customer product designs.

No license, either expressed or implied, is granted under any NVIDIA patent right, copyright, or other NVIDIA intellectual property right under this specification. Information published by NVIDIA regarding third-party products or services does not constitute a license from NVIDIA to use such products or services or a warranty or endorsement thereof. Use of such information may require a license from a third party under the patents or other intellectual property rights of the third party, or a license from NVIDIA under the patents or other intellectual property rights of NVIDIA. Reproduction of information in this specification is permissible only if reproduction is approved by NVIDIA in writing, is reproduced without alteration, and is accompanied by all associated conditions, limitations, and notices.

ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS, DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY, “MATERIALS”) ARE BEING PROVIDED “AS IS.” NVIDIA MAKES NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE. Notwithstanding any damages that customer might incur for any reason whatsoever, NVIDIA’s aggregate and cumulative liability towards customer for the products described herein shall be limited in accordance with the NVIDIA terms and conditions of sale for the product.


Trademarks
--------------------

NVIDIA, the NVIDIA logo, NVIDIA CV-CUDA, and NVIDIA TensorRT are trademarks and/or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Other company and product names may be trademarks of the respective companies with which they are associated.


Copyright
--------------------
© 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.



.. toctree::
    :caption: Beginner's Guide
    :maxdepth: 1
    :hidden:

    Installation <installation>
    Getting Started <getting_started>

.. toctree::
    :caption: API Documentation
    :maxdepth: 2
    :hidden:

    C Modules <modules/c_modules>
    C++ Modules <modules/cpp_modules>
    Index <_exhale_api/cvcuda_api>

.. toctree::
    :caption: Release Notes
    :maxdepth: 1
    :hidden:

    Pre-Alpha <relnotes/v0.1.0-prealpha>
    Alpha <relnotes/v0.2.0-alpha>
