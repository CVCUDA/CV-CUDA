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
import threading
import queue
import time


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
    torch.cuda.synchronize()
    cvcuda_RGBtensor = cvcuda.as_tensor(input.cuda(), "NHWC")
    torch.cuda.synchronize()
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
    cvcuda.reformat_into(cvcuda_nchw, cvcuda_RGBtensor_resized)
    return torch_nchw


def generate_images(N, width=None, height=None, random_size=False):
    if random_size:
        w = random.randint(1, 10)
        h = random.randint(1, 10)
    else:
        w = width
        h = height
    return torch.as_tensor(torch.rand(N, h, w, 3), dtype=torch.uint8)


def worker(device_id, task_queue, result_queue):
    while True:
        task = task_queue.get()
        if task is None:
            break
        gradient_img_batch, image_size = task
        result = preprocess(gradient_img_batch, image_size)
        result_queue.put(result)
        task_queue.task_done()


def worker_into(device_id, task_queue, result_queue):
    while True:
        task = task_queue.get()
        if task is None:
            break
        gradient_img_batch, image_size = task
        result = preprocess_into(gradient_img_batch, image_size)
        result_queue.put(result)
        task_queue.task_done()


def test_random_image_size():
    device_id = 0
    num_threads = 15

    task_queue = queue.Queue()
    result_queue = queue.Queue()

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(device_id, task_queue, result_queue))
        t.start()
        threads.append(t)

    # Set the duration to run the function (in seconds)
    duration = 10
    start_time = time.time()

    while time.time() - start_time < duration:
        batch_size = 10
        target_img_width = random.randint(220, 230)
        target_img_height = random.randint(220, 230)
        gradient_img_batch = generate_images(N=batch_size, random_size=True)
        image_size = (target_img_width, target_img_height)
        task_queue.put((gradient_img_batch, image_size))

    # Signal the threads to stop
    for _ in range(num_threads):
        task_queue.put(None)

    for t in threads:
        t.join()

    print("Random Output Image Size test complete")


def test_random_batch_size():
    device_id = 0
    num_threads = 10

    task_queue = queue.Queue()
    result_queue = queue.Queue()

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(device_id, task_queue, result_queue))
        t.start()
        threads.append(t)

    # Set the duration to run the function (in seconds)
    duration = 10
    start_time = time.time()

    while time.time() - start_time < duration:
        batch_size = random.randint(5, 10)
        target_img_width = random.randint(110, 115)
        target_img_height = random.randint(220, 230)
        gradient_img_batch = generate_images(N=batch_size, random_size=True)
        image_size = (target_img_width, target_img_height)
        task_queue.put((gradient_img_batch, image_size))

    # Signal the threads to stop
    for _ in range(num_threads):
        task_queue.put(None)

    for t in threads:
        t.join()

    print("Random Batch Size test complete")


def test_random_image_size_into():
    device_id = 0
    num_threads = 15

    task_queue = queue.Queue()
    result_queue = queue.Queue()

    threads = []
    for i in range(num_threads):
        t = threading.Thread(
            target=worker_into, args=(device_id, task_queue, result_queue)
        )
        t.start()
        threads.append(t)

    # Set the duration to run the function (in seconds)
    duration = 10
    start_time = time.time()

    while time.time() - start_time < duration:
        batch_size = 10
        target_img_width = random.randint(220, 230)
        target_img_height = random.randint(220, 230)
        gradient_img_batch = generate_images(N=batch_size, random_size=True)
        image_size = (target_img_width, target_img_height)
        task_queue.put((gradient_img_batch, image_size))

    # Signal the threads to stop
    for _ in range(num_threads):
        task_queue.put(None)

    for t in threads:
        t.join()

    print("Into Random Output Image Size test complete")


def test_random_batch_size_into():
    device_id = 0
    num_threads = 10

    task_queue = queue.Queue()
    result_queue = queue.Queue()

    threads = []
    for i in range(num_threads):
        t = threading.Thread(
            target=worker_into, args=(device_id, task_queue, result_queue)
        )
        t.start()
        threads.append(t)

    # Set the duration to run the function (in seconds)
    duration = 10
    start_time = time.time()

    while time.time() - start_time < duration:
        batch_size = random.randint(5, 10)
        target_img_width = random.randint(110, 115)
        target_img_height = random.randint(220, 230)
        gradient_img_batch = generate_images(N=batch_size, random_size=True)
        # print(gradient_img_batch.size())
        image_size = (target_img_width, target_img_height)
        task_queue.put((gradient_img_batch, image_size))

    # Signal the threads to stop
    for _ in range(num_threads):
        task_queue.put(None)

    for t in threads:
        t.join()

    print("Into Random Batch Size test complete")


def main():
    # test_random_image_size()
    test_random_batch_size_into()
    time.sleep(10)
    test_random_batch_size()
    # test_random_image_size_into()


if __name__ == "__main__":
    main()
