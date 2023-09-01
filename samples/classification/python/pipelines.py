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

import numpy as np
import logging
import cvcuda
import torch


class PreprocessorCvcuda:
    # docs_tag: begin_init_preprocessorcvcuda
    def __init__(self, device_id, cvcuda_perf):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id
        self.cvcuda_perf = cvcuda_perf
        self.mean_tensor = torch.Tensor([0.485, 0.456, 0.406])
        self.mean_tensor = self.mean_tensor.reshape(1, 1, 1, 3).cuda(self.device_id)
        self.mean_tensor = cvcuda.as_tensor(self.mean_tensor, "NHWC")
        self.stddev_tensor = torch.Tensor([0.229, 0.224, 0.225])
        self.stddev_tensor = self.stddev_tensor.reshape(1, 1, 1, 3).cuda(self.device_id)
        self.stddev_tensor = cvcuda.as_tensor(self.stddev_tensor, "NHWC")

        self.logger.info("Using CVCUDA as preprocessor.")
        # docs_tag: end_init_preprocessorcvcuda

    # docs_tag: begin_call_preprocessorcvcuda
    def __call__(self, frame_nhwc, out_size):
        self.cvcuda_perf.push_range("preprocess.cvcuda")

        # docs_tag: begin_tensor_conversion
        # Need to check what type of input we have received:
        # 1) CVCUDA tensor --> Nothing needs to be done.
        # 2) Numpy Array --> Convert to torch tensor first and then CVCUDA tensor
        # 3) Torch Tensor --> Convert to CVCUDA tensor
        if isinstance(frame_nhwc, torch.Tensor):
            frame_nhwc = cvcuda.as_tensor(frame_nhwc, "NHWC")
        elif isinstance(frame_nhwc, np.ndarray):
            frame_nhwc = cvcuda.as_tensor(
                torch.as_tensor(frame_nhwc).to(
                    device="cuda:%d" % self.device_id, non_blocking=True
                ),
                "NHWC",
            )
        # docs_tag: end_tensor_conversion

        # docs_tag: begin_preproc_pipeline
        # Resize the tensor to a different size.
        # NOTE: This resize is done after the data has been converted to a NHWC Tensor format
        #       That means the height and width of the frames/images are already same, unlike
        #       a python list of HWC tensors.
        #       This resize is only going to help it downscale to a fixed size and not
        #       to help resize images with different sizes to a fixed size. If you have a folder
        #       full of images with all different sizes, it would be best to run this sample with
        #       batch size of 1. That way, this resize operation will be able to resize all the images.
        resized = cvcuda.resize(
            frame_nhwc,
            (
                frame_nhwc.shape[0],
                out_size[1],
                out_size[0],
                frame_nhwc.shape[3],
            ),
            cvcuda.Interp.LINEAR,
        )

        # Convert to floating point range 0-1.
        normalized = cvcuda.convertto(resized, np.float32, scale=1 / 255)

        # Normalize with mean and std-dev.
        normalized = cvcuda.normalize(
            normalized,
            base=self.mean_tensor,
            scale=self.stddev_tensor,
            flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
        )

        # Convert it to NCHW layout and return it.
        normalized = cvcuda.reformat(normalized, "NCHW")

        self.cvcuda_perf.pop_range()

        # Return 3 pieces of information:
        #   1. The original nhwc frame
        #   2. The resized frame
        #   3. The normalized frame.
        return (
            frame_nhwc,
            resized,
            normalized,
        )
        # docs_tag: end_preproc_pipeline


class PostprocessorCvcuda:
    def __init__(
        self,
        output_layout,
        device_id,
        cvcuda_perf,
    ):
        self.logger = logging.getLogger(__name__)
        if output_layout != "NCHW":
            raise RuntimeError(
                "Unknown post-processing output layout: %s" % output_layout
            )
        self.output_layout = output_layout
        self.device_id = device_id
        self.cvcuda_perf = cvcuda_perf

        self.logger.info("Using CVCUDA as post-processor.")

    # docs_tag: begin_call_postprocessorcvcuda
    def __call__(self, probabilities, top_n, labels):
        self.cvcuda_perf.push_range("postprocess.cvcuda")

        # docs_tag: begin_proces_probs
        # We assume that everything other than probabilities will be a CVCUDA tensor.
        # probabilities has to be a torch tensor because we need to perform a few
        # math operations on it. Even if the TensorRT backend was used to run inference
        # it would have generated output as Torch.tensor

        actual_batch_size = probabilities.shape[0]

        # Sort output scores in descending order
        _, indices = torch.sort(probabilities, descending=True)

        probabilities = probabilities.cpu().numpy()
        indices = indices.cpu().numpy()

        # tag: Display Top N Results
        for img_idx in range(actual_batch_size):
            self.logger.info(
                "Classification probabilities for the image: %d of %d"
                % (img_idx + 1, actual_batch_size)
            )

            # Display Top N Results
            for idx in indices[img_idx][:top_n]:
                self.logger.info(
                    "\t%s: %2.3f%%"
                    % (labels[idx], round(probabilities[img_idx][idx] * 100.0, 3)),
                )

        # docs_tag: end_proces_probs

        self.cvcuda_perf.pop_range()  # postprocess

        # docs_tag: end_postproc_pipeline
