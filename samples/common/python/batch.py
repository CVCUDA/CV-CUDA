# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Union, List
import numpy as np
import cvcuda
import torch


class Batch:
    """
    This object helps us keep track of the data associated with a batch used
    throughout the deep learning pipelines of CVCUDA.
    In addition of tracking the data tensors associated with the batch, it
    allows tracking the index of the batch and any filename information one
    wants to attach (i.e. which files did the data come from).
    """

    def __init__(
        self,
        batch_idx: int,
        data: Union[cvcuda.Tensor, np.ndarray, torch.Tensor],
        fileinfo: Union[str, List[str]],
    ):
        """
        Initializes a new instance of the `Batch` class.
        :param batch_idx: A zero based int specifying the index of this batch.
        :param data: The data associated with this batch. Either a torch/CVCUDA tensor or a numpy array.
        :param fileinfo: Either a string or list or strings specifying any filename information of this batch.
        """
        self.batch_idx = batch_idx
        self.data = data
        self.fileinfo = fileinfo
