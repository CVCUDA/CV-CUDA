..
   # SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _preprocessor_cvcuda:

Classification Post-processing Pipeline
====================

The classification post processing pipeline is a relatively lightweight one with sorting being the only operation happening in it.

**The exact post-processing operations are:** ::

   Sorting the probabilities to get the top N -> Print the top N classes with the confidence scores

Since the network outputs the class probabilities (0-1) for all the classes supported by the network, we must first sort those in the descending order and take out the top-N from it. These operations will be done using PyTorch math and the results will be logged to the stdout.


.. literalinclude:: ../../../../../samples/classification/python/pipelines.py
   :language: python
   :linenos:
   :start-after: begin_proces_probs
   :end-before: end_proces_probs
   :dedent:
