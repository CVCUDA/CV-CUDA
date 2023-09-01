# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# docs_tag: begin_python_imports
# NOTE: One must import PyCuda driver first, before CVCUDA or VPF otherwise
# things may throw unexpected errors.
import pycuda.driver as cuda
import os
import sys
import time
import json
from types import SimpleNamespace
import torch
import numpy as np
import threading
from collections import deque
from pathlib import Path

# Import Triton modules
import triton_python_backend_utils as pb_utils

import cvcuda

# Bring module folders from the samples directory into our path so that
# we can import modules from it.
samples_dir = Path(os.path.abspath(__file__)).parents[5]  # samples/
segmentation_dir = Path(os.path.abspath(__file__)).parents[
    3
]  # samples/segmentation/python
sys.path.insert(0, os.path.join(samples_dir, ""))
sys.path.insert(0, os.path.join(segmentation_dir, ""))

from model_inference import SegmentationPyTorch, SegmentationTensorRT  # noqa: E402
from pipelines import PreprocessorCvcuda, PostprocessorCvcuda  # noqa: E402

from common.python.vpf_utils import (  # noqa:E402
    VideoBatchStreamingEncoderVPF,
    VideoBatchStreamingDecoderVPF,
)

from common.python.perf_utils import CvCudaPerf  # noqa: E402

# docs_tag: end_python_imports


# Triton Python Model
class TritonPythonModel:
    def initialize(self, args):
        # docs_tag: begin_init_model
        self.model_config = json.loads(args["model_config"])

        # Verify decoupled policy
        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config
        )
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(
                    args["model_name"]
                )
            )

        # Get output configuration
        out_config = pb_utils.get_output_config_by_name(self.model_config, "PACKET_OUT")
        # Convert Triton types to Numpy types
        self.out_dtype = pb_utils.triton_string_to_numpy(out_config["data_type"])

        params = self.model_config["parameters"]

        self.device_id = int(params["device_id"]["string_value"])
        self.network_width = int(params["network_width"]["string_value"])
        self.network_height = int(params["network_height"]["string_value"])
        self.visualization_class_name = params["visualization_class_name"][
            "string_value"
        ]
        self.inference_backend = params["inference_backend"]["string_value"]
        self.max_batch_size = int(params["max_batch_size_trt_engine"]["string_value"])
        # in streaming mode, Triton max batch size is always 1,
        # which is different from TRT engine profile needs a max batch size as well

        cuda_device = cuda.Device(self.device_id)
        self.cuda_ctx = cuda_device.retain_primary_context()
        self.cuda_ctx.push()
        self.cvcuda_stream = cvcuda.Stream()
        self.torch_stream = torch.cuda.ExternalStream(self.cvcuda_stream.handle)

        # Use CvCudaPerf class to record performance of various portions of code
        # It reports the data back to nvtx internally.
        # Since it requires a minimal object with certain properties passed in it
        # we will create it here. SimpleNamespace is used to create an object
        # with arbitrary attributes.
        args = SimpleNamespace()
        args.output_dir = "/tmp"
        args.device_id = self.device_id
        self.cvcuda_perf = CvCudaPerf("fcn_resnet101_streaming", default_args=args)

        if self.inference_backend == "tensorrt":
            self.inference = SegmentationTensorRT(
                output_dir="/tmp",
                seg_class_name=self.visualization_class_name,
                batch_size=self.max_batch_size,
                image_size=(self.network_width, self.network_height),
                device_id=self.device_id,
                cvcuda_perf=self.cvcuda_perf,
            )
        else:
            self.inference = SegmentationPyTorch(
                output_dir="/tmp",
                seg_class_name=self.visualization_class_name,
                batch_size=1,  # not used, pytorch can run any batch size
                image_size=(self.network_width, self.network_height),
                device_id=self.device_id,
                cvcuda_perf=self.cvcuda_perf,
            )

        self.preprocess = PreprocessorCvcuda(self.device_id, self.cvcuda_perf)
        self.postprocess = PostprocessorCvcuda(
            "NHWC",  # NHWC works better for CVCUDA-->VPF
            gpu_output=True,
            device_id=self.device_id,
            cvcuda_perf=self.cvcuda_perf,
            torch_output=False,  # let postprocess directly return cvcuda tensor
        )

        # To keep track of response threads so that we can delay
        # finalizing the model until all response threads
        # have completed.
        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()
        # docs_tag: end_init_model

        self.batch_count = 0

        # map of active [video id] to response handle
        self.video_response_handles = dict()
        # VPF encoder & decoder instances
        self.video_encoders = dict()
        self.video_decoders = dict()
        self.output_queues = dict()

        self.logger = pb_utils.Logger

    # docs_tag: begin_execute_model
    def execute(self, requests):
        """
        For streaming model, must use decoupled mode since there is no 1:1 mapping of
        input packet and output packet:
        https://github.com/triton-inference-server/python_backend/blob/main/README.md#decoupled-mode.
        Change 'model_transaction_policy' in `config.pbtxt` accordingly.
        Hold InferenceResponseSender object in a separate thread to unblock the main caller thread
        from execute() call, while keep generating responses at any time.

        The request.get_response_sender() must be used to
        get an InferenceResponseSender object associated with the request.
        Use the InferenceResponseSender.send(response=<infer response object>,
        flags=<flags>) to send responses.

        In the final response sent using the response sender object, must
        set the flags argument to TRITONSERVER_RESPONSE_COMPLETE_FINAL to
        indicate no responses will be sent for the corresponding request. When the flags argument is set to
        TRITONSERVER_RESPONSE_COMPLETE_FINAL, providing the response argument is
        optional.
        """
        # Streamed model does not support batching for packets, so 'request_count' should always
        # be 1.
        if len(requests) != 1:
            raise pb_utils.TritonModelException(
                "Unsupported batch size %d" % len(requests)
            )

        # Only use the first request handle to send all responses
        # terminate all other request handles (otherwise client will hang)
        request_id = requests[0].request_id()
        video_id, packet_id = request_id.split("_")
        packet_id = int(packet_id)
        if packet_id == 0:
            # Record response handle
            self.first_response_handle = requests[0].get_response_sender()
            self.video_response_handles[video_id] = self.first_response_handle
            self.logger.log_info(
                f"[Video Stream] Start stream processing video ID {video_id}"
            )
        else:
            # Terminate response handle right away
            requests[0].get_response_sender().send(
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )

        packet = pb_utils.get_input_tensor_by_name(requests[0], "PACKET_IN").as_numpy()
        meta_part1 = pb_utils.get_input_tensor_by_name(
            requests[0], "META1"
        ).as_numpy()  # key,pts,dts
        meta_part2 = pb_utils.get_input_tensor_by_name(
            requests[0], "META2"
        ).as_numpy()  # pos,bsl,duration
        is_first_packet = pb_utils.get_input_tensor_by_name(
            requests[0], "FIRST_PACKET"
        ).as_numpy()[0]
        is_last_packet = pb_utils.get_input_tensor_by_name(
            requests[0], "LAST_PACKET"
        ).as_numpy()[0]
        if is_first_packet:  # optional, only sent along with first packet
            # bs,w,h,fps,total_frames, pix_fmt, codec, cspace, crange
            self.meta_part0 = pb_utils.get_input_tensor_by_name(
                requests[0], "META0"
            ).as_numpy()

        # upon recv first packet, do initialization
        if is_first_packet:
            # init decoder with metadata
            self.video_decoders[video_id] = VideoBatchStreamingDecoderVPF(
                "server",
                self.logger,
                self.cvcuda_perf,
                self.device_id,
                self.cuda_ctx,
                self.cvcuda_stream,
                self.meta_part0,
            )

            # init encoder with metadata
            self.video_encoders[video_id] = VideoBatchStreamingEncoderVPF(
                "server",
                self.logger,
                self.cvcuda_perf,
                self.device_id,
                self.cuda_ctx,
                self.cvcuda_stream,
                self.meta_part0[3],
            )  # fps from metadata

            self.output_queues[video_id] = deque([])

            self.spawn_thread(self.first_response_handle, video_id)

        curr_encoder = self.video_encoders[video_id]
        curr_decoder = self.video_decoders[video_id]
        curr_queue = self.output_queues[video_id]

        # nvtx annotation to record video id (distinguish multi-stream) and batch id
        marker_suffix = (
            f"video{video_id[:4]}.batch{curr_decoder.decoder.frame_batch_idx}"
        )
        self.cvcuda_perf.push_range(marker_suffix)
        self.cvcuda_perf.push_range("decoder.vpf")

        # decoding into CVCUDA tensor. Flush when last packet is received
        frame_tensor = curr_decoder(packet, meta_part1, meta_part2, is_last_packet)

        self.cvcuda_perf.pop_range()

        # async decoding doesn't guarantee to always return frames for every request
        # at decode success, run decode-preprocess-infer-postprocess-encode pipeline
        if frame_tensor:

            # Pre-process (input is CV-CUDA tensor)
            orig_tensor, resized_tensor, normalized_tensor = self.preprocess(
                frame_tensor,
                out_size=(self.network_width, self.network_height),
            )
            self.cvcuda_perf.push_range("inference")

            # Model inference
            probabilities = self.inference(normalized_tensor)

            self.cvcuda_perf.pop_range()

            # Post-process (output is made to be cvcuda tensor rather than torch tensor)
            blurred_frame = self.postprocess(
                probabilities,
                orig_tensor,
                resized_tensor,
                self.inference.class_index,
            )

            self.width, self.height = blurred_frame.shape[2], blurred_frame.shape[1]

            self.cvcuda_perf.push_range("encoder.vpf")

            # Encode cvcuda NHWC tensor, batched
            encoded_frame = curr_encoder(blurred_frame)

            self.cvcuda_perf.pop_range()

            self.batch_count += 1

            curr_queue.extend(encoded_frame)

        # Upon receiving the last packet, flush the encoder buffer.
        # Before that, decoder is guaranteed flushed as well.
        # Note: last packet is a dummy packet without real packet data
        if is_last_packet:
            flushed_packets = curr_encoder.encoder.flush()
            curr_queue.extend(flushed_packets)
            curr_queue.append(None)  # enqueue terminal flag
            self.logger.log_info(
                f"[Encoder] Last batch of {len(flushed_packets)} packets are flushed"
            )

            self.cvcuda_perf.pop_range()

        # Unlike in non-decoupled model transaction policy, execute function
        # here returns no response. A return from this function only notifies
        # Triton that the model instance is ready to receive another request. As
        # we are not waiting for the response thread to complete here, it is
        # possible that at any give time the model may be processing multiple
        # requests.
        return None

        # docs_tag: end_execute_model

    def spawn_thread(self, response_sender, video_id):
        """
        Start a separate, persistent thread to send the responses for the request. The
        sending back of the responses is delegated to this thread.
        NOTE: only one SINGLE thread is spawned and maintained per video id instead of one thread per request.
        Why? Because order of multi-threading is not guaranteed
        -- when N response packets are sent by N threads while each packet size is random,
        the response packets may arrive out-of-order on client side.
        A packet queue is maintained for the thread to pull response packet.
        Since all this work is on a separate thread,
        the latency is hidden behind the model inference pipeline.
        """

        self.thread = threading.Thread(
            target=self.response_thread, args=(response_sender, video_id)
        )

        # making the thread persistent and completely independent of main thread
        self.thread.daemon = True

        with self.inflight_thread_count_lck:
            self.inflight_thread_count += 1

        self.thread.start()

    def response_thread(self, response_sender, video_id):
        # The response_sender is used to send response(s) associated with the
        # corresponding request.

        # all python backend utils (pb_utils) call should stay in server code
        # rather than vpf_utils.py, because client code depends on it too thus
        # cannot import pb_utils
        while True:
            # skip if queue is empty, exit if terminal flag is in queue
            if self.output_queues[video_id]:
                packet = self.output_queues[video_id].popleft()
                if packet is not None:
                    # regular packet item
                    response = pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("PACKET_OUT", packet),
                            pb_utils.Tensor(
                                "FRAME_SIZE",
                                np.array([self.height, self.width], dtype=np.uint64),
                            ),
                            pb_utils.Tensor(
                                "LAST_PACKET", np.array([False], dtype=bool)
                            ),
                        ]
                    )
                    response_sender.send(response)
                else:
                    # terminal flag as a None item in the queue
                    # unfortunately, Triton doesn't support signaling client of the terminal response, so
                    # we have to manually send a dummy packet:
                    # https://github.com/triton-inference-server/server/issues/4999
                    response = pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(
                                "PACKET_OUT", np.ndarray(shape=(0), dtype=np.uint8)
                            ),
                            pb_utils.Tensor(
                                "FRAME_SIZE", np.array([0, 0], dtype=np.uint64)
                            ),
                            pb_utils.Tensor(
                                "LAST_PACKET", np.array([True], dtype=bool)
                            ),
                        ]
                    )
                    response_sender.send(response)
                    # We must also close the response sender to indicate to Triton that we are
                    # done sending responses for the corresponding request. We can't use the
                    # response sender after closing it. The response sender is closed by
                    # setting the TRITONSERVER_RESPONSE_COMPLETE_FINAL.
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                    # remove finished video from active map
                    self.video_response_handles.pop(video_id)
                    self.logger.log_info(
                        f"[Video Stream] Finish stream processing video ID {video_id}"
                    )
                    break

            # this is very important! if thread is just busy waiting without sleep, the main thread
            # could be blocked due to Python's GIL global lock. Add minimal sleep can mitigate this
            time.sleep(0.001)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1

    # docs_tag: begin_finalize_model
    def finalize(self):
        self.cuda_ctx.pop()

    # docs_tag: end_finalize_model
