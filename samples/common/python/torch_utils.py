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

from batch import Batch

import os
import glob
import logging
import nvtx
import torch
import torchnvjpeg
import torchvision.transforms.functional as F


# docs_tag: begin_init_imagebatchdecoder_pytorch
class ImageBatchDecoderPyTorch:
    def __init__(
        self,
        input_path,
        batch_size,
        device_id,
        cuda_ctx,
    ):
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.input_path = input_path
        self.device_id = device_id
        self.total_decoded = 0
        self.batch_idx = 0
        self.cuda_ctx = cuda_ctx

        if os.path.isfile(self.input_path):
            if os.path.splitext(self.input_path)[1] == ".jpg":
                # Read the input image file.
                self.file_names = [self.input_path] * self.batch_size
                # We will use the torchnvjpeg based decoder on the GPU in case of images.
                # This will be allocated once during the first run or whenever a batch
                # size change happens.
                self.decoder = None
            else:
                raise ValueError("Unable to read file %s as image." % self.input_path)

        elif os.path.isdir(self.input_path):
            # It is a directory. Grab file names of all JPG images.
            self.decoder = None
            self.file_names = glob.glob(os.path.join(self.input_path, "*.jpg"))
            self.logger.info("Found a total of %d JPEG images." % len(self.file_names))

        else:
            raise ValueError(
                "Unknown expression given as input_path: %s." % self.input_path
            )

        # docs_tag: end_parse_imagebatchdecoder_pytorch

        # docs_tag: begin_batch_imagebatchdecoder_pytorch
        self.file_name_batches = [
            self.file_names[i : i + self.batch_size]  # noqa: E203
            for i in range(0, len(self.file_names), self.batch_size)
        ]

        self.max_image_size = 1024 * 1024 * 3  # Maximum possible image size.

        self.logger.info("Using torchnvjpeg as decoder.")

        # docs_tag: end_init_imagebatchdecoder_pytorch

    def __call__(self):
        # docs_tag: begin_call_imagebatchdecoder_pytorch
        nvtx.push_range("decoder.torch.%d" % self.batch_idx)

        if self.total_decoded == len(self.file_names):
            return None

        file_name_batch = self.file_name_batches[self.batch_idx]
        effective_batch_size = len(file_name_batch)
        data_batch = [open(path, "rb").read() for path in file_name_batch]

        # docs_tag: end_read_imagebatchdecoder_pytorch

        # docs_tag: begin_decode_imagebatchdecoder_pytorch
        if not self.decoder or effective_batch_size != self.batch_size:
            decoder = torchnvjpeg.Decoder(
                device_padding=0,
                host_padding=0,
                gpu_huffman=True,
                device_id=self.device_id,
                bath_size=effective_batch_size,
                max_cpu_threads=8,  # this is max_cpu_threads parameter. Not used internally.
                max_image_size=self.max_image_size,
                stream=None,
            )

        image_tensor_list = decoder.batch_decode(data_batch)

        # Convert the list of tensors to a tensor itself.
        image_tensors_nhwc = torch.stack(image_tensor_list)

        self.total_decoded += len(image_tensor_list)
        # docs_tag: end_decode_imagebatchdecoder_pytorch

        # docs_tag: begin_return_imagebatchdecoder_pytorch
        batch = Batch(
            batch_idx=self.batch_idx, data=image_tensors_nhwc, fileinfo=file_name_batch
        )
        self.batch_idx += 1

        nvtx.pop_range()

        return batch
        # docs_tag: end_return_imagebatchdecoder_pytorch

    def start(self):
        pass

    def join(self):
        pass


# docs_tag: begin_init_imagebatchencoder_pytorch
class ImageBatchEncoderPyTorch:
    def __init__(
        self,
        output_path,
        fps,
        device_id,
        cuda_ctx,
    ):
        self.logger = logging.getLogger(__name__)
        self._encoder = None
        self.input_layout = "NCHW"
        self.gpu_input = True
        self.output_path = output_path
        self.device_id = device_id

        self.logger.info("Using PyTorch/PIL as encoder.")
        # docs_tag: end_init_imagebatchencoder_pytorch

    # docs_tag: begin_call_imagebatchencoder_pytorch
    def __call__(self, batch):
        nvtx.push_range("encoder.torch.%d" % batch.batch_idx)

        image_tensors_nchw = batch.data

        # Bring the image_tensors_nchw to CPU and convert it to a PIL
        # image and save those.
        for img_idx in range(image_tensors_nchw.shape[0]):
            img_name = os.path.splitext(os.path.basename(batch.fileinfo[img_idx]))[0]
            results_path = os.path.join(self.output_path, "out_%s.jpg" % img_name)
            self.logger.info("Saving the overlay result to: %s" % results_path)
            overlay_cpu = image_tensors_nchw[img_idx].detach().cpu()
            overlay_pil = F.to_pil_image(overlay_cpu)
            overlay_pil.save(results_path)

        nvtx.pop_range()

        # docs_tag: end_call_imagebatchencoder_pytorch

    def start(self):
        pass

    def join(self):
        pass
