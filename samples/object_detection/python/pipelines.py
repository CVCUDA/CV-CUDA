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

import logging
import numpy as np
import cvcuda
import torch


class PreprocessorCvcuda:
    # docs_tag: begin_init_preprocessorcvcuda
    def __init__(self, device_id, cvcuda_perf):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id
        self.cvcuda_perf = cvcuda_perf
        self.scale = 1 / 255

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
        confidence_threshold,
        iou_threshold,
        device_id,
        output_layout,
        gpu_output,
        batch_size,
        cvcuda_perf,
    ):
        # docs_tag: begin_init_postprocessorcvcuda
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device_id = device_id
        self.output_layout = output_layout
        self.gpu_output = gpu_output
        self.batch_size = batch_size
        self.cvcuda_perf = cvcuda_perf

        # The Peoplenet model uses Gridbox system which divides an input image into a grid and
        # predicts four normalized bounding-box parameters for each grid.
        # The number of grid boxes is determined by the model architecture.
        # For peoplenet model, the 960x544 input image is divided into 60x34 grids.
        self.stride = 16
        self.bbox_norm = 35
        self.offset = 0.5
        self.network_width = 960
        self.network_height = 544
        self.num_rows = int(self.network_height / self.stride)
        self.num_cols = int(self.network_width / self.stride)
        self.num_classes = 3  # Number of classes the model is trained on
        self.bboxutil = BoundingBoxUtilsCvcuda(
            self.cvcuda_perf
        )  # Initializes the Bounding Box utils
        # Center of grids
        self.center_x = None
        self.center_y = None
        self.x_values = None
        self.y_values = None

        self.logger.info("Using CVCUDA as post-processor.")

    # docs_tag: end_init_postprocessorcvcuda

    def interpolate(self, boxes_pyt, image_scale_x, image_scale_y, batch_size):
        """
        Translates the bounding boxes from the grid layout to the image layout.
        """
        # docs_tag: begin_interpolate

        # Buffer batch size needs to be updated if batch size is modified
        if (
            self.center_x is None
            or self.center_y is None
            or self.batch_size != batch_size
        ):
            self.center_x = torch.zeros(
                [batch_size, self.num_rows, self.num_cols]
            ).cuda(device=self.device_id)
            self.center_y = torch.zeros(
                [batch_size, self.num_rows, self.num_cols]
            ).cuda(device=self.device_id)
            self.y_values = torch.full([self.num_cols], 1).cuda(device=self.device_id)
            self.x_values = torch.arange(0, self.num_cols).cuda(device=self.device_id)

            # Denormalize the bounding boxes
            # Compute the center of each grid
            self.cvcuda_perf.push_range("forloop1")
            for r in range(0, self.num_rows):
                self.center_y[:, r, :] = (
                    self.y_values * r * self.stride + self.offset
                ) / self.bbox_norm
                self.center_x[:, r, :] = (
                    self.x_values * self.stride + self.offset
                ) / self.bbox_norm

            self.cvcuda_perf.pop_range()

        # The raw bounding boxes shape is [N, C*4, X, Y]
        # Where N is batch size, C is number of classes, 4 is the bounding box coordinates,
        # X is the row index of the grid, Y is the column index of the grid
        # The order of the coordinates is left, bottom, right, top
        self.cvcuda_perf.push_range("forloop2")
        boxes_pyt = boxes_pyt.permute(0, 2, 3, 1)

        # The raw bounding boxes shape is [N, C*4, X, Y]
        # Where N is batch size, C is number of classes, 4 is the bounding box coordinates,
        # X is the row index of the grid, Y is the column index of the grid
        # The order of the coordinates is left, bottom, right, top
        # Shift the grid center
        for c in range(self.num_classes):
            boxes_pyt[:, :, :, 4 * c + 0] -= self.center_x
            boxes_pyt[:, :, :, 4 * c + 1] += self.center_y
            boxes_pyt[:, :, :, 4 * c + 2] += self.center_x
            boxes_pyt[:, :, :, 4 * c + 3] -= self.center_y
            # Apply the bounding box scale of the model
            boxes_pyt[:, :, :, 4 * c + 0] *= -self.bbox_norm * image_scale_x
            boxes_pyt[:, :, :, 4 * c + 1] *= self.bbox_norm * image_scale_y
            boxes_pyt[:, :, :, 4 * c + 2] *= self.bbox_norm * image_scale_x
            boxes_pyt[:, :, :, 4 * c + 3] *= -self.bbox_norm * image_scale_y

        self.cvcuda_perf.pop_range()
        return boxes_pyt

    # docs_tag: end_interpolate

    def __call__(self, raw_boxes_pyt, raw_scores_pyt, frame_nhwc):

        self.cvcuda_perf.push_range("postprocess.cvcuda")

        # docs_tag: begin_call_filterbboxcvcuda
        self.cvcuda_perf.push_range("interpolate")
        batch_size = raw_boxes_pyt.shape[0]
        image_scale_x = frame_nhwc.shape[2] / self.network_width
        image_scale_y = frame_nhwc.shape[1] / self.network_height
        # Interpolate bounding boxes to original image resolution
        interpolated_boxes_pyt = self.interpolate(
            raw_boxes_pyt, image_scale_x, image_scale_y, batch_size
        )
        self.cvcuda_perf.pop_range()

        self.cvcuda_perf.push_range("pre-nms")
        raw_scores_pyt = raw_scores_pyt.permute(0, 2, 3, 1)
        batch_scores_pyt = torch.flatten(raw_scores_pyt, start_dim=1, end_dim=3)

        # Apply NMS to filter the bounding boxes based on the confidence threshold
        # and the IOU threshold.
        batch_bboxes_pyt = torch.flatten(interpolated_boxes_pyt, start_dim=1, end_dim=2)
        batch_bboxes_pyt = batch_bboxes_pyt.reshape(batch_size, -1, 4)
        # Convert from left, bottom, right, top format to x, y, w, h format
        batch_bboxes_pyt[:, :, [1, 3]] = batch_bboxes_pyt[:, :, [3, 1]]
        batch_bboxes_pyt[:, :, 2] = (
            batch_bboxes_pyt[:, :, 2] - batch_bboxes_pyt[:, :, 0]
        )
        batch_bboxes_pyt[:, :, 3] = (
            batch_bboxes_pyt[:, :, 3] - batch_bboxes_pyt[:, :, 1]
        )
        # Convert to int16 - the data type required by the CV-CUDA NMS.
        batch_bboxes_pyt = batch_bboxes_pyt.to(
            torch.int16, memory_format=torch.contiguous_format
        )

        # Wrap torch tensor as cvcuda array
        cvcuda_boxes = cvcuda.as_tensor(batch_bboxes_pyt)
        cvcuda_scores = cvcuda.as_tensor(batch_scores_pyt.contiguous().cuda())
        self.cvcuda_perf.pop_range()

        # Apply non-maximum suppression on the bounding boxes. CV-CUDA NMS will not change
        # the shape of the resulting tensor. It will still have the same shape as the
        # input tensor. It will simply return an output boolean mask with suppressed bboxes
        # as zeros and selected bboxes as ones. Later we will filter those ones out.
        self.cvcuda_perf.push_range("nms")
        nms_masks = cvcuda.nms(
            cvcuda_boxes, cvcuda_scores, self.confidence_threshold, self.iou_threshold
        )
        nms_masks_pyt = torch.as_tensor(
            nms_masks.cuda(), device="cuda:%d" % self.device_id, dtype=torch.bool
        )
        self.cvcuda_perf.pop_range()

        # Give these NMS bounding boxes to our helper class which will filter the zeros
        # out and render bounding boxes with blur in them on the input frame.
        # docs_tag: start_outbuffer
        self.cvcuda_perf.push_range("bboxutil")
        frame_nhwc = self.bboxutil(batch_bboxes_pyt, nms_masks_pyt, frame_nhwc)
        self.cvcuda_perf.pop_range()
        if self.output_layout == "NCHW":
            render_output = cvcuda.reformat(frame_nhwc, "NCHW")
        else:
            assert self.output_layout == "NHWC"
            render_output = frame_nhwc

        if self.gpu_output:
            render_output = torch.as_tensor(
                render_output.cuda(), device="cuda:%d" % self.device_id
            )
        else:
            render_output = torch.as_tensor(render_output.cuda()).cpu().numpy()

        self.cvcuda_perf.pop_range()  # postprocess

        # Return the original nhwc frame with bboxes rendered and ROI's blurred
        return render_output
        # docs_tag: end_outbuffer


class BoundingBoxUtilsCvcuda:
    def __init__(self, cvcuda_perf):
        # docs_tag: begin_init_cuosd_bboxes
        # Settings for the bounding boxes to be rendered
        self.border_color = (0, 255, 0, 255)
        self.fill_color = (0, 0, 255, 0)
        self.thickness = 5
        self.kernel_size = 7  # kernel size for the blur ROI
        self.cvcuda_perf = cvcuda_perf
        # docs_tag: end_init_cuosd_bboxes

    def __call__(self, batch_bboxes_pyt, nms_masks_pyt, frame_nhwc):
        # docs_tag: begin_call_cuosd_bboxes
        # We will use CV-CUDA's box_blur and bndbox operators to blur out
        # the contents of the bounding boxes and draw them with color on the
        # input frame. For that to work, we must first filter out the boxes
        # which are all zeros. After doing that, we will create 3 lists:
        #   1) A list maintaining the count of valid bounding boxes per image in the batch.
        #   2) A list of all bounding box objects.
        #   3) A list of all bounding boxes stored as blur box objects.
        #
        # Once this is done, we can convert these lists to two CV-CUDA
        # structures that can be given to the blur and bndbox operators:
        #   1) cvcuda.BndBoxesI : To store the bounding boxes for the batch
        #   2) cvcuda.BlurBoxesI : To store the bounding boxes as blur boxes for the batch.
        #
        self.cvcuda_perf.push_range("forloop")
        num_boxes = []
        bounding_boxes = []
        blur_boxes = []

        # Create an array of bounding boxes with render settings.
        for current_boxes, current_masks in zip(batch_bboxes_pyt, nms_masks_pyt):
            filtered_boxes = current_boxes[current_masks]
            # Save the count of non-zero bounding boxes of this image.
            num_boxes.append(filtered_boxes.shape[0])

            for box in filtered_boxes:
                bounding_boxes.append(
                    cvcuda.BndBoxI(
                        box=tuple(box),
                        thickness=self.thickness,
                        borderColor=self.border_color,
                        fillColor=self.fill_color,
                    )
                )
                blur_boxes.append(
                    cvcuda.BlurBoxI(box=tuple(box), kernelSize=self.kernel_size)
                )

        batch_bounding_boxes = cvcuda.BndBoxesI(
            numBoxes=num_boxes, boxes=tuple(bounding_boxes)
        )
        batch_blur_boxes = cvcuda.BlurBoxesI(
            numBoxes=num_boxes, boxes=tuple(blur_boxes)
        )
        self.cvcuda_perf.pop_range()  # for loop

        # Apply blur first.
        self.cvcuda_perf.push_range("boxblur_into")
        cvcuda.boxblur_into(frame_nhwc, frame_nhwc, batch_blur_boxes)
        self.cvcuda_perf.pop_range()

        # Render the bounding boxes.
        self.cvcuda_perf.push_range("bndbox_into")
        cvcuda.bndbox_into(frame_nhwc, frame_nhwc, batch_bounding_boxes)
        self.cvcuda_perf.pop_range()

        # docs_tag: end_call_cuosd_bboxes
        return frame_nhwc
