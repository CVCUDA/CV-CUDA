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

"""
Semantic Segmentation Python sample

The semantic segmentation sample uses DeepLabv3 model from the torchvision
repository and shows the usage of CVCUDA by implementing a complete end-to-end
pipeline which can read images from the disk, pre-process them, run the inference
on them and save the overlay outputs back to the disk. This sample also gives an
overview of the interoperability of PyTorch and TensorRT with CVCUDA tensors and
operators.
"""

# docs_tag: begin_python_imports
import os
import sys
import glob
import argparse
import torch
import torchnvjpeg
import torchvision.transforms.functional as F
from torchvision.models import segmentation as segmentation_models
import numpy as np
import cvcuda
import tensorrt as trt

# Bring the commons folder from the samples directory into our path so that
# we can import modules from it.
common_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "common",
    "python",
)
sys.path.insert(0, common_dir)
from trt_utils import convert_onnx_to_tensorrt, setup_tensort_bindings  # noqa: E402

# docs_tag: end_python_imports


# docs_tag: class_def
class SemanticSegmentationSample:
    def __init__(
        self,
        input_path,
        results_dir,
        visualization_class_name,
        batch_size,
        target_img_height,
        target_img_width,
        device_id,
        backend,
    ):
        # docs_tag: begin_class_init
        self.input_path = input_path
        self.results_dir = results_dir
        self.visualization_class_name = visualization_class_name
        self.batch_size = batch_size
        self.target_img_height = target_img_height
        self.target_img_width = target_img_width
        self.device_id = device_id
        self.backend = backend
        self.class_to_idx_dict = None

        if self.backend not in ["pytorch", "tensorrt"]:
            print(
                "Invalid backend option specified: %s. "
                "Currently supports: pytorch, tensorrt" % self.backend
            )
            exit(1)
        # docs_tag: end_class_init

        # docs_tag: begin_data_read
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
        # docs_tag: end_data_read

        # docs_tag: begin_input_validation
        if not os.path.isdir(self.results_dir):
            print("Output directory not found: %s" % self.results_dir)
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
        # docs_tag: end_input_validation

    # docs_tag: setup_model_def
    def setup_model(self):
        # Setup the model and a few more things depending on the type of backend.
        # docs_tag: begin_setup_pytorch
        if self.backend == "pytorch":
            # Fetch the segmentation index to class name information from the weights
            # meta properties.
            torch_model = segmentation_models.deeplabv3_resnet101
            weights_info = segmentation_models.DeepLabV3_ResNet101_Weights

            weights = weights_info.DEFAULT
            self.class_to_idx_dict = {
                cls: idx for (idx, cls) in enumerate(weights.meta["categories"])
            }

            if self.visualization_class_name not in self.class_to_idx_dict:
                print(
                    "Requested segmentation class '%s' is not supported by the "
                    "DeepLabV3 model." % self.visualization_class_name
                )
                print(
                    "All supported class names are: %s"
                    % ", ".join(self.class_to_idx_dict.keys())
                )
                exit(1)

            # Inference uses PyTorch to run a segmentation model on the pre-processed
            # input and outputs the segmentation masks.
            model = torch_model(weights=weights)
            model.cuda(self.device_id)
            model.eval()

            return model
            # docs_tag: end_setup_pytorch

        # docs_tag: begin_setup_tensorrt
        elif self.backend == "tensorrt":
            # For TensorRT, the process is the following:
            # We check if there already exists a TensorRT engine generated
            # previously. If not, we check if there exists an ONNX model generated
            # previously. If not, we will generate both of the one by one
            # and then use those.
            # The underlying pytorch model that we use in case of TensorRT
            # inference is the FCN model from torchvision. It is only used during
            # the conversion process and not during the inference.
            onnx_file_path = os.path.join(
                self.results_dir,
                "model.%d.%d.%d.onnx"
                % (self.batch_size, self.target_img_height, self.target_img_height),
            )
            trt_engine_file_path = os.path.join(
                self.results_dir,
                "model.%d.%d.%d.trtmodel"
                % (self.batch_size, self.target_img_height, self.target_img_height),
            )

            torch_model = segmentation_models.fcn_resnet101
            weights_info = segmentation_models.FCN_ResNet101_Weights

            weights = weights_info.DEFAULT
            self.class_to_idx_dict = {
                cls: idx for (idx, cls) in enumerate(weights.meta["categories"])
            }

            if self.visualization_class_name not in self.class_to_idx_dict:
                print(
                    "Requested segmentation class '%s' is not supported by the "
                    "FCN model." % self.visualization_class_name
                )
                print(
                    "All supported class names are: %s"
                    % ", ".join(self.class_to_idx_dict.keys())
                )
                exit(1)

            # Check if we have a previously generated model.
            if not os.path.isfile(trt_engine_file_path):
                if not os.path.isfile(onnx_file_path):
                    # First we use PyTorch to create a segmentation model.
                    with torch.no_grad():
                        pyt_model = torch_model(weights=weights)
                        pyt_model.to("cuda")
                        pyt_model.eval()

                        # Allocate a dummy input to help generate an ONNX model.
                        dummy_x_in = torch.randn(
                            self.batch_size,
                            3,
                            self.target_img_height,
                            self.target_img_width,
                            requires_grad=False,
                        ).cuda()

                        # Generate an ONNX model using the PyTorch's onnx export.
                        torch.onnx.export(
                            pyt_model,
                            args=dummy_x_in,
                            f=onnx_file_path,
                            export_params=True,
                            opset_version=11,
                            do_constant_folding=True,
                            input_names=["input"],
                            output_names=["output"],
                            dynamic_axes={
                                "input": {0: "batch_size"},
                                "output": {0: "batch_size"},
                            },
                        )

                        # Remove the tensors and model after this.
                        del pyt_model
                        del dummy_x_in
                        torch.cuda.empty_cache()

                    print("Generated an ONNX model and saved at: %s" % onnx_file_path)
                else:
                    print("Using a pre-built ONNX model from: %s" % onnx_file_path)

                # Now that we have an ONNX model, we will continue generating a
                # serialized TensorRT engine from it.
                success = convert_onnx_to_tensorrt(
                    onnx_file_path,
                    trt_engine_file_path,
                    max_batch_size=self.batch_size,
                    max_workspace_size=1,
                )
                if success:
                    print("Generated TensorRT engine in: %s" % trt_engine_file_path)
                else:
                    print("Failed to generate the TensorRT engine.")
                    exit(1)

            else:
                print(
                    "Using a pre-built TensorRT engine from: %s" % trt_engine_file_path
                )

            # docs_tag: begin_load_tensorrt
            # Once the TensorRT engine generation is all done, we load it.
            trt_logger = trt.Logger(trt.Logger.INFO)
            with open(trt_engine_file_path, "rb") as f, trt.Runtime(
                trt_logger
            ) as runtime:
                trt_model = runtime.deserialize_cuda_engine(f.read())

            # Create execution context.
            context = trt_model.create_execution_context()

            # Allocate the output bindings.
            output_tensors, output_idx = setup_tensort_bindings(
                trt_model, self.device_id
            )

            return context, output_tensors, output_idx
            # docs_tag: end_setup_tensorrt

        else:
            print(
                "Invalid backend option specified: %s. "
                "Currently supports: pytorch, tensorrt" % self.backend
            )
            exit(1)

    # docs_tag: begin_execute_inference
    def execute_inference(self, model_info, torch_preprocessed_tensor):
        # Executes inference depending on the type of the backend.
        # docs_tag: begin_infer_pytorch
        if self.backend == "pytorch":
            with torch.no_grad():
                infer_output = model_info(torch_preprocessed_tensor)["out"]

            return infer_output
            # docs_tag: end_infer_pytorch

        # docs_tag: begin_infer_tensorrt
        elif self.backend == "tensorrt":
            # Setup TensorRT IO binding pointers.
            # docs_tag: begin_tensorrt_unpack
            context, output_tensors, output_idx = model_info  # Un-pack this.
            # docs_tag: end_tensorrt_unpack

            # We need to check the allocated batch size and the required batch
            # size. Sometimes, during to batching, the last batch may be of
            # less size than the batch size. In those cases, we would simply
            # pad that with zero inputs and discard its output later on.
            # docs_tag: begin_check_last_batch
            allocated_batch_size = output_tensors[output_idx].shape[0]
            required_batch_size = torch_preprocessed_tensor.shape[0]

            if allocated_batch_size != required_batch_size:
                # Need to pad the input with extra zeros tensors.
                new_input_shape = [allocated_batch_size - required_batch_size] + list(
                    torch_preprocessed_tensor.shape[1:]
                )

                # Allocate just the extra input required.
                extra_input = torch.zeros(
                    size=new_input_shape,
                    dtype=torch_preprocessed_tensor.dtype,
                    device=self.device_id,
                )

                # Update the existing input tensor by joining it with the newly
                # created input.
                torch_preprocessed_tensor = torch.cat(
                    (torch_preprocessed_tensor, extra_input)
                )
                # docs_tag: end_check_last_batch

            # docs_tag: begin_tensorrt_run
            # Prepare the TensorRT I/O bindings.
            input_bindings = [torch_preprocessed_tensor.data_ptr()]
            output_bindings = []
            for t in output_tensors:
                output_bindings.append(t.data_ptr())
            io_bindings = input_bindings + output_bindings

            # Execute synchronously.
            context.execute_v2(bindings=io_bindings)
            infer_output = output_tensors[output_idx]
            # docs_tag: end_tensorrt_run

            # docs_tag: begin_discard_last_batch
            # Finally, check if we had padded the input. If so, we need to
            # discard the extra output.
            if allocated_batch_size != required_batch_size:
                # We need remove the padded output.
                infer_output = torch.split(
                    infer_output,
                    [required_batch_size, allocated_batch_size - required_batch_size],
                )[0]

            return infer_output
            # docs_tag: end_discard_last_batch
            # docs_tag: end_infer_tensorrt

        else:
            print(
                "Invalid backend option specified: %s. "
                "Currently supports: pytorch, tensorrt" % self.backend
            )
            exit(1)

    # docs_tag: begin_run
    def run(self):
        # docs_tag: begin_run_basics
        # Runs the complete sample end-to-end.
        max_image_size = 1024 * 1024 * 3  # Maximum possible image size.

        # First setup the model.
        model_info = self.setup_model()

        # Next, we would batchify the file_list and data_list based on the
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
        # docs_tag: end_run_basics

        # docs_tag: begin_batch_loop
        # We will use the torchnvjpeg based decoder on the GPU. This will be
        # allocated once during the first run or whenever a batch size change
        # happens.
        decoder = None

        for file_name_batch, data_batch in zip(file_name_batches, data_batches):
            print("Processing batch %d of %d" % (batch_idx + 1, len(file_name_batches)))
            effective_batch_size = len(file_name_batch)

            # docs_tag: begin_decode
            # Decode in batch using torchnvjpeg decoder on the GPU.
            if not decoder or effective_batch_size != self.batch_size:
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

            # Also save an NCHW version of the image tensors.
            image_tensors_nchw = image_tensors.permute(0, 3, 1, 2)  # from NHWC to NCHW

            input_image_height, input_image_width = (
                image_tensors.shape[1],
                image_tensors.shape[2],
            )
            # docs_tag: end_decode

            # docs_tag: begin_torch_to_cvcuda
            # A torch tensor can be wrapped into a CVCUDA Object using the "as_tensor"
            # function in the specified layout. The datatype and dimensions are derived
            # directly from the torch tensor.
            cvcuda_input_tensor = cvcuda.as_tensor(image_tensors, "NHWC")
            # docs_tag: end_torch_to_cvcuda

            # docs_tag: begin_preproc
            # Start the pre-processing now. For segmentation, pre-processing includes
            # the following sequence of operations.
            # Resize -> DataType Convert(U8->F32) -> Normalize -> Interleaved to Planar

            # Resize to the input network dimensions.
            cvcuda_resized_tensor = cvcuda.resize(
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
            # i.e uint8->float. The values are first scaled to the 0-1 range.
            cvcuda_float_tensor = cvcuda.convertto(
                cvcuda_resized_tensor, np.float32, scale=1 / 255
            )

            # Normalize using mean and std-dev
            mean_tensor = torch.Tensor([0.485, 0.456, 0.406])
            stddev_tensor = torch.Tensor([0.229, 0.224, 0.225])
            mean_tensor = mean_tensor.reshape(1, 1, 1, 3).cuda(self.device_id)
            stddev_tensor = stddev_tensor.reshape(1, 1, 1, 3).cuda(self.device_id)
            cvcuda_mean_tensor = cvcuda.as_tensor(mean_tensor, "NHWC")
            cvcuda_stddev_tensor = cvcuda.as_tensor(stddev_tensor, "NHWC")
            cvcuda_normalized_tensor = cvcuda.normalize(
                cvcuda_float_tensor,
                base=cvcuda_mean_tensor,
                scale=cvcuda_stddev_tensor,
                flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
            )

            # The final stage in the pre-process pipeline includes converting the NHWC
            # buffer into a NCHW buffer.
            cvcuda_preprocessed_tensor = cvcuda.reformat(
                cvcuda_normalized_tensor, "NCHW"
            )
            # docs_tag: end_preproc

            # docs_tag: begin_run_infer
            # Execute the inference after converting the tensor back to Torch.
            torch_preprocessed_tensor = torch.as_tensor(
                cvcuda_preprocessed_tensor.cuda(),
                device=torch.device("cuda", self.device_id),
            )
            infer_output = self.execute_inference(model_info, torch_preprocessed_tensor)
            # docs_tag: end_run_infer

            # Once the inference is over we would start the post-processing steps.
            # First, we normalize the probability scores from the network.
            # docs_tag: begin_normalize
            normalized_masks = torch.nn.functional.softmax(infer_output, dim=1)
            # docs_tag: end_normalize

            # Then filter based on the scores corresponding only to the class of
            # interest
            # docs_tag: begin_argmax
            class_masks = (
                normalized_masks.argmax(dim=1)
                == self.class_to_idx_dict[self.visualization_class_name]
            )
            class_masks = torch.unsqueeze(class_masks, dim=-1)  # Makes it NHWC
            class_masks = class_masks.type(torch.uint8)  # Make it uint8 from bool
            # docs_tag: end_argmax

            # Then convert the masks back to CV-CUDA tensor for rest of the
            # post-processing:
            # 1) Up-scaling back to the original image dimensions
            # 2) Apply blur on the original images and overlay on the original image.

            # Convert back to CV-CUDA tensor
            # docs_tag: begin_mask_upscale
            cvcuda_class_masks = cvcuda.as_tensor(class_masks.cuda(), "NHWC")
            # Upscale it.
            cvcuda_class_masks_upscaled = cvcuda.resize(
                cvcuda_class_masks,
                (effective_batch_size, input_image_height, input_image_width, 1),
                cvcuda.Interp.LINEAR,
            )
            # Convert back to PyTorch.
            class_masks_upscaled = torch.as_tensor(
                cvcuda_class_masks_upscaled.cuda(),
                device=torch.device("cuda", self.device_id),
            )
            # Repeat in last dimension to make the mask 3 channel
            class_masks_upscaled = class_masks_upscaled.repeat(1, 1, 1, 3)
            class_masks_upscaled_nchw = class_masks_upscaled.permute(
                0, 3, 1, 2
            )  # from NHWC to NCHW
            # docs_tag: end_mask_upscale

            # Blur the input images using the median blur op and convert to PyTorch.
            # docs_tag: begin_input_blur
            cvcuda_blurred_input_imgs = cvcuda.median_blur(
                cvcuda_input_tensor, ksize=(27, 27)
            )
            cvcuda_blurred_input_imgs = cvcuda.reformat(
                cvcuda_blurred_input_imgs, "NCHW"
            )
            blurred_input_imgs = torch.as_tensor(
                cvcuda_blurred_input_imgs.cuda(),
                device=torch.device("cuda", self.device_id),
            )
            # docs_tag: end_input_blur

            # Create an overlay image. We do this by selectively blurring out pixels
            # in the input image where the class mask prediction was absent (i.e. False)
            # We already have all the things required for this: The input images,
            # the blurred version of the input images and the upscale version
            # of the mask
            # docs_tag: begin_overlay
            mask_absent = class_masks_upscaled_nchw == 0
            image_tensors_nchw[mask_absent] = blurred_input_imgs[
                mask_absent
            ]  # In-place
            # docs_tag: end_overlay

            # Loop over all the images in the current batch and save the
            # inference results.
            # docs_tag: begin_visualization_loop
            for img_idx in range(effective_batch_size):
                img_name = os.path.splitext(os.path.basename(file_name_batch[img_idx]))[
                    0
                ]
                results_path = os.path.join(self.results_dir, "out_%s.jpg" % img_name)
                print(
                    "\tSaving the overlay result for %s class for to: %s"
                    % (self.visualization_class_name, results_path)
                )

                # Convert the overlay which was in-place saved in
                # image_tensors_nchw to a PIL image on the CPU and save it.
                overlay_cpu = image_tensors_nchw[img_idx].detach().cpu()
                overlay_pil = F.to_pil_image(overlay_cpu)
                overlay_pil.save(results_path)

            # Increment the batch counter.
            batch_idx += 1
            # docs_tag: end_visualization_loop


# docs_tag: main_func
def main():
    parser = argparse.ArgumentParser(
        "Semantic segmentation sample using CV-CUDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_path",
        default="./assets/Weimaraner.jpg",
        type=str,
        help="Either a path to a JPEG image or a directory containing JPEG "
        "images to use as input.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default="/tmp",
        type=str,
        help="The folder where the output segmentation overlay should be stored.",
    )

    parser.add_argument(
        "-c",
        "--class_name",
        default="dog",
        type=str,
        help="The segmentation class to visualize the results for.",
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
        "-b",
        "--batch_size",
        default=1,
        type=int,
        help="Artificially simulated batch size. The same input image will be read and "
        "used this many times. Useful for performance bench-marking.",
    )

    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="The GPU device to use for this sample.",
    )

    parser.add_argument(
        "-bk",
        "--backend",
        default="pytorch",
        type=str,
        help="The inference backend to use. Currently supports pytorch, tensorrt.",
    )

    # Parse the command line arguments.
    args = parser.parse_args()

    # Run the sample.
    sample = SemanticSegmentationSample(
        args.input_path,
        args.output_dir,
        args.class_name,
        args.batch_size,
        args.target_img_height,
        args.target_img_width,
        args.device_id,
        args.backend,
    )

    sample.run()


if __name__ == "__main__":
    main()
