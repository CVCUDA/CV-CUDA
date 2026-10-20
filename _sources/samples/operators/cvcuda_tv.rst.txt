..
   # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _sample_cvcuda_tv:

``cvcuda.tv``
=============

These runnable samples demonstrate augmentation, color, geometry, and
preprocessing pipelines using the ``cvcuda.tv`` functional API and
``cvcuda.tv.Compose``. They use the repository's default image and accept the
standard sample ``--input`` and ``--output`` options. Run any script from the repository root, for example:

.. code-block:: bash

   python3 samples/operators/cvcuda_tv_color.py

Augmentation
------------

.. literalinclude:: ../../../../samples/operators/cvcuda_tv_augment.py
   :language: python
   :start-after: docs_tag: begin_example
   :end-before: docs_tag: end_example
   :dedent: 4

Color
-----

.. literalinclude:: ../../../../samples/operators/cvcuda_tv_color.py
   :language: python
   :start-after: docs_tag: begin_example
   :end-before: docs_tag: end_example
   :dedent: 4

Geometry
--------

.. literalinclude:: ../../../../samples/operators/cvcuda_tv_geometry.py
   :language: python
   :start-after: docs_tag: begin_example
   :end-before: docs_tag: end_example
   :dedent: 4

Miscellaneous Preprocessing
---------------------------

.. literalinclude:: ../../../../samples/operators/cvcuda_tv_misc.py
   :language: python
   :start-after: docs_tag: begin_example
   :end-before: docs_tag: end_example
   :dedent: 4

TorchVision Pipelines
---------------------

.. literalinclude:: ../../../../samples/operators/cvcuda_tv_compose.py
   :language: python
   :start-after: docs_tag: begin_example
   :end-before: docs_tag: end_example
   :dedent: 4

See Also
--------

* :doc:`Using cvcuda.tv <../../cvcuda_tv>`
* :doc:`TV API reference <../../modules/python/tv>`
