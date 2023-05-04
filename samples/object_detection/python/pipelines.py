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
        self.scale = 1 / 255
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id

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
        self, threshold, iou_threshold, device_id, output_layout, gpu_output, batch_size
    ):
        # docs_tag: begin_init_postprocessorcvcuda
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id

        self.gpu_output = gpu_output
        self.output_layout = output_layout

        # confidence threshold of the detections.
        self.confidence_threshold = threshold
        # bboxes with iou more than iou_threshold will be discarded
        self.iou_threshold = iou_threshold

        # Peoplenet model uses Gridbox system which divides an input image into a grid and
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
        # Number of classes the mode is trained on
        self.num_classes = 3
        self.batch_size = batch_size
        # Define the Bounding Box utils
        self.bboxutil = BoundingBoxUtilsCvcuda()

        # Center of grids
        self.center_x = None
        self.center_y = None
        self.x_values = None
        self.y_values = None

        self.logger.info("Using CVCUDA as post-processor.")

    # docs_tag: end_init_postprocessorcvcuda

    def interpolate(self, boxes, image_scale_x, image_scale_y, batch_size):
        # docs_tag: begin_interpolate
        boxes = torch.as_tensor(boxes)

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
        for b in range(batch_size):
            for r in range(0, self.num_rows):
                self.center_y[b, r, :] = (
                    self.y_values * r * self.stride + self.offset
                ) / self.bbox_norm
                self.center_x[b, r, :] = (
                    self.x_values * self.stride + self.offset
                ) / self.bbox_norm

        """
        The raw bounding boxes shape is [N, C*4, X, Y]
        Where N is batch size, C is number of classes, 4 is the bounding box coordinates,
        X is the row index of the grid, Y is the column index of the grid
        The order of the coordinates is left, bottom, right, top
        """
        for c in range(self.num_classes):
            # Shift the grid centers
            boxes[:, 4 * c + 0, :, :] -= self.center_x
            boxes[:, 4 * c + 1, :, :] += self.center_y
            boxes[:, 4 * c + 2, :, :] += self.center_x
            boxes[:, 4 * c + 3, :, :] -= self.center_y
            # Apply the bounding box scale of the model
            boxes[:, 4 * c + 0, :, :] *= -self.bbox_norm * image_scale_x
            boxes[:, 4 * c + 1, :, :] *= self.bbox_norm * image_scale_y
            boxes[:, 4 * c + 2, :, :] *= self.bbox_norm * image_scale_x
            boxes[:, 4 * c + 3, :, :] *= -self.bbox_norm * image_scale_y
        return boxes

    # docs_tag: end_interpolate

    def __call__(self, raw_boxes, raw_scores, frame_nhwc):

        nvtx.push_range("postprocess.cvcuda")

        # docs_tag: begin_call_filterbboxcvcuda

        batch_size = raw_boxes.shape[0]
        image_scale_x = frame_nhwc.shape[2] / self.network_width
        image_scale_y = frame_nhwc.shape[1] / self.network_height
        # Interpolate bounding boxes to original image resolution
        interpolated_boxes = self.interpolate(
            raw_boxes, image_scale_x, image_scale_y, batch_size
        )

        # NMS to filter bounding boxes based on confidence threshold and iou threshold
        batch_boxes = None
        batch_scores = None
        for b in range(batch_size):
            per_batch_bboxes = None
            per_batch_scores = None
            # Convert N, C*4, X, Y format into N, B, 4 format
            # where N is batch size, C is number of classes,X,Y is the number of grids,
            # B is the number of bounding boxes
            for c in range(self.num_classes):
                # Address formatting issue when indexed directly
                class_idx = c * 4
                class_idx_4 = class_idx + 4
                # Extract the bounding boxes across all grids
                per_class_bboxes = interpolated_boxes[b, class_idx:class_idx_4, :, :]
                # Flatten the bounding boxes to 4, B
                per_class_bboxes = per_class_bboxes.reshape(4, -1)
                # Reshape 4, B to B, 4
                per_class_bboxes = per_class_bboxes.T
                # Convert to int32 data type required by cvcuda NMS
                per_class_bboxes = per_class_bboxes.type(torch.int32)
                # Convert from left, bottom, right, top format to x, y, w, h format
                per_class_bboxes[:, [1, 3]] = per_class_bboxes[:, [3, 1]]
                per_class_bboxes[:, 2] = per_class_bboxes[:, 2] - per_class_bboxes[:, 0]
                per_class_bboxes[:, 3] = per_class_bboxes[:, 3] - per_class_bboxes[:, 1]
                # Convert score to [B] format of float32 datatype
                per_class_scores = raw_scores[b, c, :, :].flatten()
                if per_batch_bboxes is None:
                    per_batch_bboxes = per_class_bboxes
                    per_batch_scores = per_class_scores
                else:
                    per_batch_bboxes = torch.cat(
                        (per_batch_bboxes, per_class_bboxes), 0
                    )
                    per_batch_scores = torch.cat(
                        (per_batch_scores, per_class_scores), 0
                    )
            if batch_boxes is None:
                batch_boxes = per_batch_bboxes.unsqueeze(0)
                batch_scores = per_batch_scores.unsqueeze(0)
            else:
                batch_boxes = torch.cat((batch_boxes, per_batch_bboxes.unsqueeze(0)), 0)
                batch_scores = torch.cat(
                    (batch_scores, per_batch_scores.unsqueeze(0)), 0
                )

        # Wrap torch tensor as cvcuda array
        cvcuda_boxes = cvcuda.as_tensor(batch_boxes.contiguous().cuda())
        cvcuda_scores = cvcuda.as_tensor(batch_scores.contiguous().cuda())

        # Filter bounding boxes using NMS
        nms_boxes = cvcuda.nms(
            cvcuda_boxes, cvcuda_scores, self.confidence_threshold, self.iou_threshold
        )

        # Wrap output of NMS as torch tensor. CVCUDA NMS zeros out the invalid bboxes
        filtered_bboxes = torch.as_tensor(nms_boxes.cuda()).contiguous()

        # Get the indices of the non zero bounding boxes
        torch_indices = torch.nonzero(filtered_bboxes)
        torch_unique_indices = torch.unique(torch_indices.T[1])
        filtered_bboxes = torch.index_select(filtered_bboxes, 1, torch_unique_indices)
        # docs_tag: end_call_filterbboxcvcuda

        # render bounding boxes and Blur ROI's
        # docs_tag: start_outbuffer
        frame_nhwc = self.bboxutil(filtered_bboxes, frame_nhwc)
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

        nvtx.pop_range()  # postprocess

        # Return 2 pieces of information:
        #   1. The original nhwc frame with bboxes rendered and ROI's blurred
        #   2. The bounding boxes predicted
        return (render_output, filtered_bboxes)
        # docs_tag: end_outbuffer


class BoundingBoxUtilsCvcuda:
    def __init__(self):
        # docs_tag: begin_init_cuosd_bboxes
        # Settings for the bounding boxes to be rendered
        self.border_color = (0, 255, 0, 255)
        self.fill_color = (0, 0, 255, 0)
        self.thickness = 5
        # kernel size for the blur ROI
        self.kernel_size = 7
        # docs_tag: end_init_cuosd_bboxes

    def __call__(self, bboxes, frame_nhwc):
        # docs_tag: begin_call_cuosd_bboxes
        batch_size = frame_nhwc.shape[0]
        num_boxes = []
        for b in range(len(bboxes)):
            num_boxes.append(len(bboxes[b]))
        boxes = []
        blur_boxes = []
        # Create an array of bounding boxes with render settings.
        for b in range(batch_size):
            for i in range(num_boxes[b]):
                box = [
                    bboxes[b][i][0],
                    bboxes[b][i][1],
                    bboxes[b][i][2],
                    bboxes[b][i][3],
                ]
                boxes.append(
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
            cusod_boxes = cvcuda.BndBoxesI(numBoxes=num_boxes, boxes=tuple(boxes))
            cuosd_blur_boxes = cvcuda.BlurBoxesI(
                numBoxes=num_boxes, boxes=tuple(blur_boxes)
            )

        cvcuda.boxblur_into(frame_nhwc, frame_nhwc, cuosd_blur_boxes)

        # Render bounding boxes and blur the ROI inside the bounding box
        cvcuda.bndbox_into(frame_nhwc, frame_nhwc, cusod_boxes)

        # docs_tag: end_call_cuosd_bboxes
        return frame_nhwc
