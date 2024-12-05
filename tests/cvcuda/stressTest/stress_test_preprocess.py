# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cvcuda
import torch
import random
import time

time_of_test_in_min = 0.1


def preprocess(input, out_size):
    frame_nhwc = cvcuda.as_tensor(
        torch.as_tensor(input).to(device="cuda:0", non_blocking=True),
        "NHWC",
    )
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
    return normalized


def preprocess_into(input, out_size):
    cvcuda_RGBtensor = cvcuda.as_tensor(input.cuda(), "NHWC")

    torch_RGBtensor_resized = torch.empty(
        (
            cvcuda_RGBtensor.shape[0],
            out_size[1],
            out_size[0],
            cvcuda_RGBtensor.shape[3],
        ),
        dtype=torch.uint8,
        device="cuda:0",
    )
    cvcuda_RGBtensor_resized = cvcuda.as_tensor(
        torch_RGBtensor_resized.cuda(),
        "NHWC",
    )
    cvcuda.resize_into(
        cvcuda_RGBtensor_resized,
        cvcuda_RGBtensor,
        cvcuda.Interp.LINEAR,
    )

    torch_nchw = torch.empty(
        (input.shape[0], 3, out_size[1], out_size[0]),
        dtype=torch.uint8,
        device="cuda:0",
    )
    cvcuda_nchw = cvcuda.as_tensor(torch_nchw.cuda(0), "NCHW")
    # normalized = cvcuda.convertto(cvcuda_nchw, np.float32, scale=1 / 255)
    cvcuda.reformat_into(cvcuda_nchw, cvcuda_RGBtensor_resized)
    return torch_nchw


def generate_images(N, width=None, height=None, random_size=False):
    if random_size:
        w = random.randint(100, 500)
        h = random.randint(100, 500)
    else:
        w = width
        h = height
    return torch.as_tensor(torch.rand(N, h, w, 3), dtype=torch.uint8)


def test_random_image_size():
    target_img_width = 224
    target_img_height = 224
    batch_size = 20

    duration = time_of_test_in_min * 60  # 5 minutes
    start_time = time.time()

    while time.time() - start_time < duration:
        gradient_img_batch = generate_images(N=batch_size, random_size=True)
        image_size = (target_img_width, target_img_height)
        result = preprocess(gradient_img_batch, image_size)  # noqa: F841
    print("Random Image Size Test Complete")


def test_increasing_batch_size():
    target_img_width = 224
    target_img_height = 224
    batch_size = 1

    duration = time_of_test_in_min * 60  # 5 minutes
    start_time = time.time()

    while time.time() - start_time < duration:
        gradient_img_batch = generate_images(N=batch_size, random_size=True)
        image_size = (target_img_width, target_img_height)
        result = preprocess(gradient_img_batch, image_size)  # noqa: F841
        batch_size += 1
    print("Increasing Batch Size Test Complete")


def test_random_batch_size():
    target_img_width = 224
    target_img_height = 224
    batch_size = 1

    duration = time_of_test_in_min * 60  # 5 minutes
    start_time = time.time()

    while time.time() - start_time < duration:
        gradient_img_batch = generate_images(width=1080, height=1920, N=batch_size)
        image_size = (target_img_width, target_img_height)
        result = preprocess(gradient_img_batch, image_size)  # noqa: F841
        batch_size = random.randint(1, 80)
    print("Random Batch Size Test Complete")


def test_random_image_size_into():
    target_img_width = 224
    target_img_height = 224
    batch_size = 20

    duration = time_of_test_in_min * 60  # 5 minutes
    start_time = time.time()

    while time.time() - start_time < duration:
        gradient_img_batch = generate_images(N=batch_size, random_size=True)
        image_size = (target_img_width, target_img_height)
        result = preprocess_into(gradient_img_batch, image_size)  # noqa: F841
    print("Into operator Random Image Size Test Complete")


def test_increasing_batch_size_into():
    target_img_width = 224
    target_img_height = 224
    batch_size = 1

    duration = time_of_test_in_min * 60  # 5 minutes
    start_time = time.time()

    while time.time() - start_time < duration:
        gradient_img_batch = generate_images(N=batch_size, random_size=True)
        image_size = (target_img_width, target_img_height)
        result = preprocess_into(gradient_img_batch, image_size)  # noqa: F841
        batch_size += 1
    print("Into operator Random Image Size Test Complete")


def test_random_batch_size_into():
    target_img_width = 224
    target_img_height = 224
    batch_size = 1

    duration = time_of_test_in_min * 60  # 5 minutes
    start_time = time.time()

    while time.time() - start_time < duration:
        gradient_img_batch = generate_images(N=batch_size, random_size=True)
        image_size = (target_img_width, target_img_height)
        result = preprocess_into(gradient_img_batch, image_size)  # noqa: F841
        batch_size = random.randint(1, 80)
    print("Into Operator Increasing Batch Size Test Complete")


def main():
    print(torch.cuda.get_device_properties(0))
    test_random_image_size()
    test_random_batch_size()
    test_random_image_size_into()
    test_random_batch_size_into()

    test_increasing_batch_size()
    test_increasing_batch_size_into()


if __name__ == "__main__":
    main()
