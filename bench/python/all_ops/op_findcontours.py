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

# NOTE: One must import PyCuda driver first, before CVCUDA or VPF otherwise
# things may throw unexpected errors.
import pycuda.driver as cuda  # noqa: F401

from bench_utils import AbstractOpBase
import cvcuda
import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class OpFindContours(AbstractOpBase):
    def setup(self, input):
        grayscale_input = read_image(
            os.path.join(self.assets_dir, "countour_lines.jpg")
        )
        grayscale_input = grayscale_input.moveaxis(
            0, -1
        ).contiguous()  # From CHW to HWC
        # Binarize the grayscale_input
        grayscale_input[grayscale_input <= 50] = 0
        grayscale_input[grayscale_input > 50] = 255

        grayscale_input = [grayscale_input.clone() for _ in range(input.shape[0])]
        grayscale_input = torch.stack(grayscale_input)
        grayscale_input = grayscale_input.cuda(self.device_id)
        self.grayscale_input = cvcuda.as_tensor(grayscale_input, "NHWC")

    def run(self, input):
        return cvcuda.find_contours(self.grayscale_input)

    def visualize(self):
        """
        Attempts to visualize the output produced by the operator as an image by writing it
        down to the disk. May raise exceptions if visualization is not successful.
        """
        output_dir = self._setup_clear_output_dir(filename_ends_with="_op_out.jpg")
        # Convert the inputs and outputs to numpy arrays first.
        # input shape: NHWC
        # out[0] = points_info shape: NxMx2 (M == max points, 2 for x and y coordinates)
        # out[1] = contours_info shape: NxC where
        #       (C == max contours, number of non-zero elements are number of contours)
        input_npy = (
            torch.as_tensor(
                self.grayscale_input.cuda(), device="cuda:%d" % self.device_id
            )
            .cpu()
            .numpy()
        )
        points_npy = (
            torch.as_tensor(self.op_output[0].cuda(), device="cuda:%d" % self.device_id)
            .cpu()
            .numpy()
        )
        num_contours_npy = (
            torch.as_tensor(self.op_output[1].cuda(), device="cuda:%d" % self.device_id)
            .cpu()
            .numpy()
        )

        # Loop over all the images...
        for i, img in enumerate(input_npy):

            # Grab the information on the points and the contours of this image.
            points_info = points_npy[i]
            contours_info = num_contours_npy[i]

            # Keep only the non-zero entries from contours_info
            contours_info = contours_info[np.nonzero(contours_info)]
            # Use the num_points in contours_info to split the points_info
            # Since the values in num_points are not start-stop indices of the points
            # we need to use cumsum to fix it and use it inside the split function
            valid_points = np.split(points_info, contours_info.cumsum())
            # Last element in valid_points is the remainder of the points so need to drop it.
            all_contours = valid_points[:-1]  # This list stores OpenCV style contours.

            plt.figure(figsize=(img.shape[1] / 100.0, img.shape[0] / 100.0))
            plt.gca().invert_yaxis()

            plt.plot(0, 0, color="white")
            plt.plot(img.shape[1], img.shape[0], color="white")
            for contour in all_contours:
                x, y = contour[:, 0], contour[:, 1]
                plt.plot(x, y, color="green", linewidth=2)

            # Save using PIL
            out_file_name = "img_%d_op_out.jpg" % i
            plt.savefig(os.path.join(output_dir, out_file_name))
            plt.close()
