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

# Import python modules

import os
import glob
import torch
from torchvision import models
import torchnvjpeg
import numpy as np
import argparse

# tag: Import CVCUDA module
import cvcuda

# tag: Classification Sample

"""
Image Classification python sample

The image classification sample uses Resnet50 based model trained on Imagenet
The sample app pipeline includes preprocessing, inference and post process stages
which takes as input a batch of images and returns the TopN classification results
of each image.

This sample gives an overview of the interoperability of pytorch with CVCUDA
tensors and operators
"""


class ClassificationSample:
    def __init__(
        self,
        input_path,
        labels_file,
        batch_size,
        target_img_height,
        target_img_width,
        device_id,
    ):
        self.input_path = input_path
        self.batch_size = batch_size
        self.target_img_height = target_img_height
        self.target_img_width = target_img_width
        self.device_id = device_id
        self.labels_file = labels_file

        # tag: Image Loading
        # Start by parsing the input_path expression first.
        if os.path.isfile(self.input_path):
            # Read the input image file.
            self.file_names = [self.input_path] * self.batch_size
            # Then create a dummy list with the data from the same file to simulate a
            # batch.
            self.data = [open(path, "rb").read() for path in self.file_names]

        elif os.path.isdir(self.input_path):
            # It is a directory. Grab all the images from it.
            self.file_names = glob.glob(os.path.join(self.input_path, "*.jpg"))
            self.data = [open(path, "rb").read() for path in self.file_names]
            print("Read a total of %d JPEG images." % len(self.data))

        else:
            print(
                "Input path not found. "
                "It is neither a valid JPEG file nor a directory: %s" % self.input_path
            )
            exit(1)

        # tag: Validate other inputs
        if not os.path.isfile(self.labels_file):
            print("Labels file not found: %s" % self.labels_file)
            exit(1)

        if self.batch_size <= 0:
            print("batch_size must be a value >=1.")
            exit(1)

        if self.target_img_height < 10:
            print("target_img_height must be a value >=10.")
            exit(1)

        if self.target_img_width < 10:
            print("target_img_width must be a value >=10.")
            exit(1)

    def run(self):
        # Runs the complete sample end-to-end.
        max_image_size = 1024 * 1024 * 3  # Maximum possible image size.

        # tag: NvJpeg Decoder
        # We will batchify the file_list and data_list based on the
        # batch size and start processing these batches one by one.
        file_name_batches = [
            self.file_names[i : i + self.batch_size]  # noqa: E203
            for i in range(0, len(self.file_names), self.batch_size)
        ]
        data_batches = [
            self.data[i : i + self.batch_size]  # noqa: E203
            for i in range(0, len(self.data), self.batch_size)
        ]
        batch_idx = 0

        for file_name_batch, data_batch in zip(file_name_batches, data_batches):
            effective_batch_size = len(file_name_batch)
            print("Processing batch %d of %d" % (batch_idx + 1, len(file_name_batches)))

            # Decode in batch using torchnvjpeg decoder on the GPU.
            decoder = torchnvjpeg.Decoder(
                0,
                0,
                True,
                self.device_id,
                effective_batch_size,
                8,  # this is max_cpu_threads parameter. Not used internally.
                max_image_size,
                torch.cuda.current_stream(self.device_id),
            )
            image_tensor_list = decoder.batch_decode(data_batch)

            # Convert the list of tensors to a tensor itself.
            image_tensors = torch.stack(image_tensor_list)

            # tag: Wrapping into Tensor
            # A torch tensor can be wrapped into a CVCUDA Object using the "as_tensor"
            # function in the specified layout. The datatype and dimensions are derived
            # directly from the torch tensor.
            cvcuda_input_tensor = cvcuda.as_tensor(image_tensors, "NHWC")

            # tag: Preprocess
            """
            Preprocessing includes the following sequence of operations.
            Resize -> DataType Convert(U8->F32) -> Normalize
            (Apply mean and std deviation) -> Interleaved to Planar
            """

            # Resize
            # Resize to the input network dimensions
            cvcuda_resize_tensor = cvcuda.resize(
                cvcuda_input_tensor,
                (
                    effective_batch_size,
                    self.target_img_height,
                    self.target_img_width,
                    3,
                ),
                cvcuda.Interp.LINEAR,
            )

            # Convert to the data type and range of values needed by the input layer
            # i.e uint8->float. A Scale is applied to normalize the values in the
            # range 0-1
            cvcuda_convert_tensor = cvcuda.convertto(
                cvcuda_resize_tensor, np.float32, scale=1 / 255
            )

            """
            The input to the network needs to be normalized based on the mean and
            std deviation value to standardize the input data.
            """

            # Create a torch tensor to store the mean and standard deviation
            # values for R,G,B
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            base_tensor = torch.Tensor(mean)
            stddev_tensor = torch.Tensor(std)

            # Reshape the the number of channels. The R,G,B values scale and offset
            # will be applied to every color plane respectively across the batch
            base_tensor = torch.reshape(base_tensor, (1, 1, 1, 3)).cuda()
            stddev_tensor = torch.reshape(stddev_tensor, (1, 1, 1, 3)).cuda()

            # Wrap the torch tensor in a CVCUDA Tensor
            cvcuda_base_tensor = cvcuda.as_tensor(base_tensor, "NHWC")
            cvcuda_scale_tensor = cvcuda.as_tensor(stddev_tensor, "NHWC")

            # Apply the normalize operator and indicate the scale values are
            # std deviation i.e scale = 1/stddev
            cvcuda_norm_tensor = cvcuda.normalize(
                cvcuda_convert_tensor,
                base=cvcuda_base_tensor,
                scale=cvcuda_scale_tensor,
                flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
            )

            # The final stage in the preprocess pipeline includes converting the RGB
            # buffer into a planar buffer
            cvcuda_preprocessed_tensor = cvcuda.reformat(cvcuda_norm_tensor, "NCHW")

            # tag: Inference
            # Inference uses pytorch to run a resnet50 model on the preprocessed
            # input and outputs the classification scores for 1000 classes
            # Load Resnet model pretrained on Imagenet
            resnet50 = models.resnet50(pretrained=True)
            resnet50.to("cuda")
            resnet50.eval()

            # Run inference on the preprocessed input
            torch_preprocessed_tensor = torch.as_tensor(
                cvcuda_preprocessed_tensor.cuda(), device="cuda"
            )

            with torch.no_grad():
                infer_output = resnet50(torch_preprocessed_tensor)

            # tag: Postprocess
            """
            Postprocessing function normalizes the classification score from the network
            and sorts the scores to get the TopN classification scores.
            """
            # Apply softmax to Normalize scores between 0-1
            scores = torch.nn.functional.softmax(infer_output, dim=1)

            # Sort output scores in descending order
            _, indices = torch.sort(infer_output, descending=True)

            # tag: Display Top N Results
            # Read and parse the classes
            with open(self.labels_file, "r") as f:
                classes = [line.strip() for line in f.readlines()]

            # top results to print out
            topN = 5
            for img_idx in range(effective_batch_size):
                print(
                    "Result for the image: %d of %d"
                    % (img_idx + 1, effective_batch_size)
                )

                # Display Top N Results
                for idx in indices[img_idx][:topN]:
                    idx = idx.item()
                    print(
                        "\tClass : ",
                        classes[idx],
                        " Score : ",
                        scores[img_idx][idx].item(),
                    )


def main():
    parser = argparse.ArgumentParser(
        "Classification sample using CV-CUDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input_path",
        default="tabby_tiger_cat.jpg",
        type=str,
        help="Either a path to a JPEG image or a directory containing JPEG "
        "images to use as input.",
    )

    parser.add_argument(
        "-l",
        "--labels_file",
        default="imagenet-classes.txt",
        type=str,
        help="The labels file to read and parse.",
    )

    parser.add_argument(
        "-th",
        "--target_img_height",
        default=224,
        type=int,
        help="The height to which you want to resize the input_image before "
        "running inference.",
    )

    parser.add_argument(
        "-tw",
        "--target_img_width",
        default=224,
        type=int,
        help="The width to which you want to resize the input_image before "
        "running inference.",
    )

    parser.add_argument(
        "-b", "--batch_size", default=1, type=int, help="Input Batch size"
    )

    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="The GPU device to use for this sample.",
    )

    # Parse the command line arguments.
    args = parser.parse_args()

    # Run the sample.
    sample = ClassificationSample(
        args.input_path,
        args.labels_file,
        args.batch_size,
        args.target_img_height,
        args.target_img_width,
        args.device_id,
    )

    sample.run()


if __name__ == "__main__":
    main()
