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
import nvtx


class PreprocessorCvcuda:
    # docs_tag: begin_init_preprocessorcvcuda
    def __init__(self, device_id):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id
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
        nvtx.push_range("preprocess.cvcuda")

        # docs_tag: begin_tensor_conversion
        # Need to check what type of input we have received:
        # 1) CVCUDA tensor --> Nothing needs to be done.
        # 2) Numpy Array --> Convert to torch tensor first and then CVCUDA tensor
        # 3) Torch Tensor --> Convert to CVCUDA tensor
        if isinstance(frame_nhwc, torch.Tensor):
            frame_nhwc = cvcuda.as_tensor(frame_nhwc, "NHWC")
            has_copy = False
        elif isinstance(frame_nhwc, np.ndarray):
            has_copy = True  # noqa: F841
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

        nvtx.pop_range()

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
        gpu_output,
        device_id,
    ):
        self.logger = logging.getLogger(__name__)
        if output_layout != "NCHW" and output_layout != "NHWC":
            raise RuntimeError(
                "Unknown post-processing output layout: %s" % output_layout
            )
        self.gpu_output = gpu_output
        self.output_layout = output_layout
        self.device_id = device_id

        self.logger.info("Using CVCUDA as post-processor.")

    # docs_tag: begin_call_postprocessorcvcuda
    def __call__(self, probabilities, frame_nhwc, resized_tensor, class_index):
        nvtx.push_range("postprocess.cvcuda")

        # docs_tag: begin_proces_probs
        # We assume that everything other than probabilities will be a CVCUDA tensor.
        # probabilities has to be a torch tensor because we need to perform a few
        # math operations on it. Even if the TensorRT backend was used to run inference
        # it would have generated output as Torch.tensor

        actual_batch_size = resized_tensor.shape[0]

        class_probs = probabilities[:actual_batch_size, class_index, :, :]
        class_probs = torch.unsqueeze(class_probs, dim=-1)
        class_probs *= 255
        class_probs = class_probs.type(torch.uint8)

        cvcuda_class_masks = cvcuda.as_tensor(class_probs.cuda(), "NHWC")
        # docs_tag: end_proces_probs

        # docs_tag: begin_postproc_pipeline
        # Upscale the resulting masks to the full resolution of the input image.
        cvcuda_class_masks_upscaled = cvcuda.resize(
            cvcuda_class_masks,
            (frame_nhwc.shape[0], frame_nhwc.shape[1], frame_nhwc.shape[2], 1),
            cvcuda.Interp.NEAREST,
        )

        # Blur the down-scaled input images and upscale them back to their original resolution.
        # A part of this will be used to create a background blur effect later when the
        # overlay happens.
        # Note: We apply blur on the low-res version of the images to save computation time.
        cvcuda_blurred_input_imgs = cvcuda.gaussian(
            resized_tensor, kernel_size=(15, 15), sigma=(5, 5)
        )
        cvcuda_blurred_input_imgs = cvcuda.resize(
            cvcuda_blurred_input_imgs,
            (frame_nhwc.shape[0], frame_nhwc.shape[1], frame_nhwc.shape[2], 3),
            cvcuda.Interp.LINEAR,
        )

        # Next we apply joint bilateral filter on the up-scaled masks with the gray version of the
        # input image as guidance to smooth out the edges of the masks. This is needed because
        # the mask was generated in lower resolution and then up-scaled. Joint bilateral will help
        # in smoothing out the edges, resulting in a nice smooth mask.
        cvcuda_frame_nhwc = cvcuda.as_tensor(frame_nhwc.cuda(), "NHWC")

        cvcuda_image_tensor_nhwc_gray = cvcuda.cvtcolor(
            cvcuda_frame_nhwc, cvcuda.ColorConversion.BGR2GRAY
        )

        cvcuda_jb_masks = cvcuda.joint_bilateral_filter(
            cvcuda_class_masks_upscaled,
            cvcuda_image_tensor_nhwc_gray,
            diameter=5,
            sigma_color=50,
            sigma_space=1,
        )

        # Create an overlay image. We do this by selectively blurring out pixels
        # in the input image where the class mask prediction was absent (i.e. False)
        # We already have all the things required for this: The input images,
        # the blurred version of the input images and the upscale version
        # of the mask.
        cvcuda_composite_imgs_nhwc = cvcuda.composite(
            cvcuda_frame_nhwc,
            cvcuda_blurred_input_imgs,
            cvcuda_jb_masks,
            3,
        )

        # Based on the output requirements, we return appropriate tensors.
        if self.output_layout == "NCHW":
            cvcuda_composite_imgs_out = cvcuda.reformat(
                cvcuda_composite_imgs_nhwc, "NCHW"
            )
        else:
            assert self.output_layout == "NHWC"
            cvcuda_composite_imgs_out = cvcuda_composite_imgs_nhwc

        if self.gpu_output:
            cvcuda_composite_imgs_out = torch.as_tensor(
                cvcuda_composite_imgs_out.cuda(), device="cuda:%d" % self.device_id
            )
        else:
            cvcuda_composite_imgs_out = (
                torch.as_tensor(cvcuda_composite_imgs_out.cuda()).cpu().numpy()
            )

        nvtx.pop_range()  # postprocess

        # docs_tag: end_postproc_pipeline

        return cvcuda_composite_imgs_out
