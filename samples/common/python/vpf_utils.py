# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
vpf_utils

This file hosts various Video Processing Framework(VPF) related utilities.
"""


import os
import sys
import av
import logging
import numpy as np
import torch
import nvcv
import cvcuda
from fractions import Fraction
import PyNvCodec as nvc
import PytorchNvCodec as pnvc

from pathlib import Path

# Bring module folders from the samples directory into our path so that
# we can import modules from it.
samples_dir = Path(os.path.abspath(__file__)).parents[2]  # samples/
sys.path.insert(0, os.path.join(samples_dir, ""))

"""
Streaming video version of the Video Batch Decoder using VPF.
"""


class VideoBatchStreamingDecoderVPF:
    def __init__(self, side, logger, cvcuda_perf, *args):
        self.logger = logging.getLogger(__name__)
        self.side = side
        self.cvcuda_perf = cvcuda_perf
        self.logger = logging.getLogger(__name__) if self.side == "client" else logger
        # client can use Python logger
        # server is managed by Triton so we can only use Triton Python backend logger
        # server logger is pb_utils.Logger object, which can only be passed as param
        # because vpf_utils.py is shared by both client and server code

        self.decoder = nvstreamingdecoder(self.side, self.logger)
        if self.side == "client":
            self.cvcuda_perf.push_range("decoder.vpf.client_setup")
            self.decoder.client_setup(*args)
            self.cvcuda_perf.pop_range()
        elif self.side == "server":
            self.cvcuda_perf.push_range("decoder.vpf.server_setup")
            self.decoder.server_setup(*args)
            self.cvcuda_perf.pop_range()

    # docs_tag: begin_call_videobatchdecoder_vpf
    def __call__(self, *args):
        if self.side == "client":
            self.cvcuda_perf.push_range("decoder.vpf.client_send")
            self.decoder.client_send(*args)
            self.cvcuda_perf.pop_range()
        elif self.side == "server":
            self.cvcuda_perf.push_range("decoder.vpf.server_receive")
            ret = self.decoder.server_recv(*args)
            self.cvcuda_perf.pop_range()
            return ret
        # docs_tag: end_batch_videobatchdecoder_vpf

    def start(self):
        pass

    def join(self):
        pass


class VideoBatchStreamingEncoderVPF:
    def __init__(self, side, logger, cvcuda_perf, *args):
        self.side = side
        self.cvcuda_perf = cvcuda_perf
        self.logger = logging.getLogger(__name__) if self.side == "client" else logger
        # client can use Python logger
        # server is managed by Triton so we can only use Triton Python backend logger
        # server logger is pb_utils.Logger object, which can only be passed as param
        # because vpf_utils.py is shared by both client and server code

        self.encoder = nvstreamingencoder(self.side, self.logger)
        if self.side == "client":
            self.cvcuda_perf.push_range("encoder.vpf.client_setup")
            self.encoder.client_setup(*args)
            self.cvcuda_perf.pop_range()
        elif self.side == "server":
            self.cvcuda_perf.push_range("encoder.vpf.server_setup")
            self.encoder.server_setup(*args)
            self.cvcuda_perf.pop_range()

    # docs_tag: begin_call_videobatchencoder_vpf
    def __call__(self, *args):
        if self.side == "client":
            self.cvcuda_perf.push_range("encoder.vpf.client_receive")
            self.encoder.client_recv(*args)
            self.cvcuda_perf.pop_range()
        elif self.side == "server":
            self.cvcuda_perf.push_range("encoder.vpf.server_send")
            ret = self.encoder.server_send(*args)
            self.cvcuda_perf.pop_range()
            return ret
        # docs_tag: end_batch_videobatchencoder_vpf

    def start(self):
        pass

    def join(self):
        self.encoder.container.close()


class nvstreamingdecoder:
    def __init__(self, side, logger):
        """
        Initialize streaming decoder on client and server side.
        :param side: "client" or "server"
        """
        self.side = side
        self.logger = logger

    def client_setup(self, enc_file, model_name, model_version):
        """
        Client-side initialization of demuxer and streaming utilities.
        :param enc_file: Full path to the MP4 file that needs to be decoded.
        """
        if self.side != "client":
            raise ValueError("client_setup must be called from client side.")

        self.logger.info("Using VPF as streaming decoder.")

        # Demuxer to (1) get video file properties (2) later for demux() to packets.
        self.nvDemux = nvc.PyFFmpegDemuxer(enc_file)
        self.w, self.h = self.nvDemux.Width(), self.nvDemux.Height()
        self.fps = self.nvDemux.Framerate()
        self.total_frames = self.nvDemux.Numframes()

        # Format compatibility check
        self.pix_fmt = self.nvDemux.Format()
        self.is_yuv420 = (
            nvc.PixelFormat.YUV420 == self.pix_fmt
            or nvc.PixelFormat.NV12 == self.pix_fmt
        )
        self.is_yuv444 = nvc.PixelFormat.YUV444 == self.pix_fmt

        self.codec = self.nvDemux.Codec()
        is_hevc = nvc.CudaVideoCodec.HEVC == self.codec

        # Nvdec supports NV12 (resampled YUV420) and YUV444 formats
        # But YUV444 HW decode is supported for HEVC only
        self.is_hw_dec = self.is_yuv420 or self.is_yuv444
        if self.is_hw_dec and self.is_yuv444 and not is_hevc:
            self.is_hw_dec = False

        if not self.is_hw_dec:
            raise ValueError(
                "Current combination of hardware and the video file being read does not support "
                "hardware accelerated decoding."
            )

        # Determine colorspace conversion parameters.
        # Some video streams don't specify these parameters so default values
        # are most widespread bt601 and mpeg.
        self.cspace, self.crange = self.nvDemux.ColorSpace(), self.nvDemux.ColorRange()

        self.model_name = model_name
        self.model_version = model_version

    def client_send(self, client, grpc_class, video_id, batch_size):
        # raw bytes of packet to set over Triton network
        packet = np.ndarray(shape=(0), dtype=np.uint8)
        # metadata of packet (timestamp, pos, bsl, etc.)
        packet_data = nvc.PacketData()

        is_first_packet = True
        is_last_packet = False
        packet_count = 0
        packet_size = 0
        while True:
            # Demuxer has SYNC design, it guarantees to return a packet every time it's called.
            # If demuxer can't return packet it usually means EOF.
            if not self.nvDemux.DemuxSinglePacket(packet):
                is_last_packet = True

            # Get the most recent packet data to obtain frame timestamp
            self.nvDemux.LastPacketData(packet_data)

            # ASYNC send raw bytes and metadata to server
            # VPF/src/TC/inc/CodecsSupport.hpp: key/pts/dts - int64, pos/bsl/duration - uint64
            # key: keyframe boolean flag
            # pts/dts: presentation timestamp / decode timestamp,
            # during playback it's helpful but not needed for normal decoding
            # pos: byte offset (?)
            # bsl: bitstream length
            # duration: ?
            meta_part0 = np.array(
                [
                    batch_size,
                    self.w,
                    self.h,
                    self.fps,
                    self.total_frames,
                    self.pix_fmt.value,
                    self.codec.value,
                    self.cspace.value,
                    self.crange.value,
                ],
                dtype=np.int32,
            )  # get value for enums
            meta_part1 = np.array(
                [packet_data.key, packet_data.pts, packet_data.dts], dtype=np.int64
            )
            meta_part2 = np.array(
                [packet_data.pos, packet_data.bsl, packet_data.duration],
                dtype=np.uint64,
            )

            inputs = []
            inputs.append(
                grpc_class.InferInput("PACKET_IN", packet.shape, "UINT8")
            )  # variable shape
            inputs.append(grpc_class.InferInput("META1", meta_part1.shape, "INT64"))
            inputs.append(grpc_class.InferInput("META2", meta_part2.shape, "UINT64"))
            inputs.append(grpc_class.InferInput("FIRST_PACKET", [1], "BOOL"))
            inputs.append(grpc_class.InferInput("LAST_PACKET", [1], "BOOL"))
            inputs[0].set_data_from_numpy(packet)
            inputs[1].set_data_from_numpy(meta_part1)
            inputs[2].set_data_from_numpy(meta_part2)
            inputs[3].set_data_from_numpy(np.array([is_first_packet], dtype=bool))
            inputs[4].set_data_from_numpy(np.array([is_last_packet], dtype=bool))
            if is_first_packet:  # optional, only send once for first packet
                inputs.append(grpc_class.InferInput("META0", meta_part0.shape, "INT32"))
                inputs[5].set_data_from_numpy(meta_part0)

            # IMPORTANT: streaming mode must process one video at a time for now.
            # For advanced support of multi-video streaming, request_id may be used to represent the client ID
            # and use sequence batcher (w/ multi-threading support) on server side to handle the response.
            if is_first_packet:
                is_first_packet = False

            outputs = []
            outputs.append(grpc_class.InferRequestedOutput("PACKET_OUT"))

            # gRPC bi-directional streaming concept
            # https://grpc.io/docs/what-is-grpc/core-concepts/#rpc-life-cycle
            # why bi-directional? In streamed decoding/encoding, both the encoded and decoded data
            # are compressed in packets, so there is no 1:1 unary RPC mapping
            # of the client request & server response. This is also called decoupled mode.
            client.async_stream_infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs,
                request_id=str(video_id) + "_" + str(packet_count),
            )

            packet_count += 1
            packet_size += len(packet)
            self.logger.debug(
                f"packet No. {packet_count}, size {packet.size}, pos {packet_data.pos} async sent to server"
            ) if packet_count % 50 == 0 else None

            if is_last_packet:
                self.logger.debug(
                    "last sentinel packet sent to mark the end of the stream"
                )
                break

        self.logger.debug(
            f"video {str(video_id)} is async sent to server via packet bytestream."
            f"[width={self.w}, height={self.h}, total frames={self.total_frames}, "
            f"total packets={packet_count} (including one EOF packet), "
            f"total bytes={packet_size // 1024 // 1024}(MB)]"
        )

    def server_setup(self, device_id, cuda_ctx, cuda_str, metadata):
        """
        Server-side initialization of HW-accelerated video decoder and streaming utilities.
        :param device_id: id of video card which will be used for decoding & processing.
        :param cuda_ctx: A cuda context object.
        :param cuda_str: A cuda stream object.
        :param metadata: video demux info.
        Ref: https://github.com/NVIDIA/VideoProcessingFramework/blob/master/samples/SampleDemuxDecode.py
        """
        if self.side != "server":
            raise ValueError("server_setup must be called from server side")

        self.device_id, self.cuda_ctx, self.cuda_str = device_id, cuda_ctx, cuda_str
        (
            self.frame_batch_size,
            self.w,
            self.h,
            self.fps,
            self.total_frames,
            pix_fmt,
            codec,
            cspace,
            crange,
        ) = metadata
        self.pix_fmt, self.codec, self.cspace, self.crange = (
            nvc.PixelFormat(pix_fmt),
            nvc.CudaVideoCodec(codec),
            nvc.ColorSpace(cspace),
            nvc.ColorRange(crange),
        )

        self.nvDec = nvc.PyNvDecoder(
            self.w,
            self.h,
            self.pix_fmt,
            self.codec,
            self.cuda_ctx.handle,
            self.cuda_str.handle,
        )

        # Determine colorspace conversion parameters.
        # Some video streams don't specify these parameters so default values
        # are most widespread bt601 and mpeg.
        if nvc.ColorSpace.UNSPEC == self.cspace:
            self.cspace = nvc.ColorSpace.BT_601
        if nvc.ColorRange.UDEF == self.crange:
            self.crange = nvc.ColorRange.MPEG
        self.cc_ctx = nvc.ColorspaceConversionContext(self.cspace, self.crange)

        # frameSize = int(self.w * self.h * 3 / 2)
        # self.rawFrame = np.ndarray(shape=(frameSize), dtype=np.uint8)

        # Set CVCUDA color conversion code to do YUV->RGB
        is_yuv420 = (
            nvc.PixelFormat.YUV420 == self.pix_fmt
            or nvc.PixelFormat.NV12 == self.pix_fmt
        )
        is_yuv444 = nvc.PixelFormat.YUV444 == self.pix_fmt
        self.cvcuda_code = None
        if is_yuv420:
            self.cvcuda_code = cvcuda.ColorConversion.YUV2RGB_NV12
        elif is_yuv444:
            self.cvcuda_code = cvcuda.ColorConversion.YUV2RGB

        self.total_decoded = 0

        # We would use VPF for video encoding/decoding, and CVCUDA to do color conversions
        # to and from RGB to NV12 format. These formats are required by VPF to encode/decode
        # video streams. Since CVCUDA can do these conversions much faster on a batch level
        # and since VPF does not work on batches, we would perform these conversions here
        # in this class using CVCUDA. We would pre-allocate the memory required by these
        # conversions upon the first use or whenever the batch size changes. This would allow
        # us to use the 'into' versions of CVCUDA operators without allocating/de-allocating
        # memory on every batch. We need to be mindful of the following things when dealing
        # with NV12 format in CVCUDA:
        # NV12 is a complex format and it is not tensor friendly so libraries use a workaround
        # to put the NV12 in a "matrix" form. They put the YUV from NV12 as 3/2 height
        # 1 height is Y luma that is full resolution
        # 1/2 height is UV chroma that is 2x2 down-scaled
        # Hence you would see YUV or NV12's H dimension 1.5 times the RGB's H dimension.
        self.cvcuda_RGBtensor_batch = cvcuda.Tensor(
            (self.frame_batch_size, self.h, self.w, 3),
            nvcv.Type.U8,
            nvcv.TensorLayout.NHWC,
        )

        # client send packets to server where Triton cannot do any effective batching on frame, therefore
        # server has to handle batching explicitly. Meanwhile, server recv call can't be blocking because
        # it's the call is invoked per client request -- need to cache the frames and batch them
        self.frame_batch = []
        self.frame_batch_idx = 0

    def server_recv(self, packet, packet_data1, packet_data2, is_last_packet):
        """
        A good feature can be add here - single surface decoding + batched buffering. Or is it even possible
        directly batched surface decoding?
        Challenge: data over Triton is now streamed packets rather than frames,
        Triton's auto batching on packets can't guarantee the decoded surface is batched.
        But in the inference pipeline, batched processing can be very important,
        otherwise model can only run with BS=1...
        Therefore, we need to support this feature either by
        (1) extra server receive logic here to buffer the decoded surface and return the batch to pipeline
        (2) check if VPF can support accumulating packets and decoding into batched surface.
        On (2), at least the following feature requests to VPF:
        (i) batched decoded surface output from packets
        (ii) batched tensor copy interface with PyT/CVCUDA. Also ideally zero-copy interface with CVCUDA.
        (iii) better handled in Triton C++ backend
        """
        # recover packet metadata
        packet_data = nvc.PacketData()
        packet_data.key, packet_data.pts, packet_data.dts = packet_data1
        packet_data.pos, packet_data.bsl, packet_data.duration = packet_data2

        # Decoder is ASYNC by design.
        # As it consumes packets from demuxer one at a time it may NOT return
        # decoded surface every time the decoding function is called.
        # Usually, at the first few decoding steps, no surface is returned. After that
        # each step consistently returns surface, and the final flushing stage will return
        # a few more. From end-to-end, # of received packets = # of decoded frames w/ flushing.
        surface_nv12 = self.nvDec.DecodeSurfaceFromPacket(packet_data, packet)
        if not surface_nv12.Empty():
            # valid surface is decoded
            self.total_decoded += 1
            self.frame_batch.append(self.surface_to_tensor(surface_nv12))

            # edge case: valid surface is decoded, but it's also the at last packet
            # need to flush and use irregular batch instead of entering batch size check
            # otherwise full batch reached && is last packet coincidence can't trigger flush
            if is_last_packet:
                ret = self.flush()
                return ret

            if len(self.frame_batch) == self.frame_batch_size:
                self.logger.log_info(
                    f"[Decoder] Batch {self.frame_batch_idx} of size {self.frame_batch_size} is decoded"
                ) if self.frame_batch_idx % 50 == 0 else None

                # [N,H,W]
                image_tensor_nhwc = torch.stack(self.frame_batch)
                # [N,H,W] to [N,H,W,1] 4D torch tensor
                image_tensor_nhwc = torch.unsqueeze(image_tensor_nhwc, -1)
                # Make it a CVCUDA Tensor, C will be 1.
                cvcuda_NV12tensor = cvcuda.as_tensor(
                    image_tensor_nhwc, nvcv.TensorLayout.NHWC
                )
                # Convert from NV12 to RGB. This will be NHWC.
                cvcuda.cvtcolor_into(
                    self.cvcuda_RGBtensor_batch, cvcuda_NV12tensor, self.cvcuda_code
                )

                # reset
                self.frame_batch = []
                self.frame_batch_idx += 1

                return self.cvcuda_RGBtensor_batch
        else:
            # no valid surface available. Two cases:
            # (1) async, accumulated packets are not sufficient yet. Need to keep going
            # (2) reach the end of packet stream. Need to flush.
            # Note: there could be some unbatched frame remained
            # in frame_batch list that we need to concatenate. So the last batch has arbitrary length
            if is_last_packet:
                ret = self.flush()
                return ret
            else:
                return None

    def flush(self):
        self.logger.log_info("[Decoder] Received EOF packet. Start flushing surfaces")
        last_frame_batch_size = (
            len(self.frame_batch) + self.total_frames - self.total_decoded
        )
        self.cvcuda_RGBtensor_batch = cvcuda.Tensor(
            (last_frame_batch_size, self.h, self.w, 3),
            nvcv.Type.U8,
            nvcv.TensorLayout.NHWC,
        )

        # due to the async nature, when the last packet is received, there are surface data in the pipe
        # that needs to be flushed. Data size unknown, could be multiple surfaces so need while loop.
        while True:
            # Now we flush decoder until emtpy the decoded frames queue.
            surface_nv12 = self.nvDec.FlushSingleSurface()
            if surface_nv12.Empty():
                break
            self.total_decoded += 1
            self.frame_batch.append(self.surface_to_tensor(surface_nv12))

        self.logger.log_info(
            f"[Decoder] Batch {self.frame_batch_idx} of size {last_frame_batch_size} is flushed"
        )
        # [N,H,W]
        image_tensor_nhwc = torch.stack(self.frame_batch)
        # [N,H,W] to [N,H,W,1] 4D torch tensor
        image_tensor_nhwc = torch.unsqueeze(image_tensor_nhwc, -1)
        # Make it a CVCUDA Tensor, C will be 1.
        cvcuda_NV12tensor = cvcuda.as_tensor(image_tensor_nhwc, nvcv.TensorLayout.NHWC)
        # Convert from NV12 to RGB. This will be NHWC.
        cvcuda.cvtcolor_into(
            self.cvcuda_RGBtensor_batch, cvcuda_NV12tensor, self.cvcuda_code
        )

        # reset
        self.frame_batch = []
        self.frame_batch_idx += 1
        return self.cvcuda_RGBtensor_batch

    # docs_tag: begin_imp_nvstreamingdecoder
    def surface_to_tensor(self, surface):
        """
        Convert a decoded surface (on vRAM) to torch.cuda.FloatTensor.
        :param surface A NV12 surface.
        Note: must be NV12 rather than YUV420, CV-CUDA can do NV12-->YUV-->RGB directly,
        no need to use PySurfaceConverter for NV12-->YUV420 in VPF.
        CV-CUDA is also faster on batched input.
        :return torch NV12 tensor of shape [H, W], where H is 1.5 times the raw image height.
        """

        surf_plane = surface.PlanePtr()

        # VPF to PyTorch tensor [W,H]
        # Note: this VPF-->PyTorch for decoded frame is NOT zero-copy. It calls cudaMemcpy2D under the hood.
        # Optimization could be done such that VPF-->CVCUDA is zero copy without routing via PyT
        img_tensor = pnvc.makefromDevicePtrUint8(
            surf_plane.GpuMem(),
            surf_plane.Width(),
            surf_plane.Height(),  # NV12 Height is 1.5x RGB Height
            surf_plane.Pitch(),
            surf_plane.ElemSize(),
        )
        if img_tensor is None:
            raise RuntimeError("Cannot export to tensor.")

        return img_tensor

    # docs_tag: end_imp_nvstreamingdecoder


class nvstreamingencoder:
    def __init__(self, side, logger):
        """
        Initialize streaming encoder on client and server side.
        :param side: "client" or "server"
        """
        self.side = side
        self.logger = logger

    def client_setup(self, output_path, fps):
        if self.side != "client":
            raise ValueError("This must be called from client side")

        self.logger.info("Using VPF as streaming encoder.")

        self.output_path = output_path
        self.fps = fps

        self.pts_time = 0  # present timestamp
        self.delta_t = 1  # increment the packets' timestamp by this much.

        self.container = av.open(self.output_path, "w")
        self.avstream = self.container.add_stream("h264", rate=self.fps)
        # 1/fps would be our scale.
        self.avstream.time_base = 1 / Fraction(fps)

    def client_recv(self, packet, height, width):
        """
        Receive ffmpeg compressed packets and write to disk (remux).
        VPF doesn't have remuxing support, so we need to use PyAV ffmpeg.
        """
        self.avstream.height = height
        self.avstream.width = width
        self.write_frame(
            packet,
            self.pts_time,
            self.fps,
            self.avstream,
            self.container,
        )
        self.pts_time += self.delta_t

    def write_frame(self, encoded_frame, pts_time, fps, stream, container):
        encoded_bytes = bytearray(encoded_frame)
        pkt = av.packet.Packet(encoded_bytes)
        pkt.pts = pts_time
        pkt.dts = pts_time
        pkt.stream = stream
        pkt.time_base = 1 / Fraction(fps)
        container.mux(pkt)

    def server_setup(self, device_id, cuda_ctx, cuda_str, fps):
        """
        Create instance of HW-accelerated video encoder.
        :param device_id: id of video card which will be used for encoding & processing.
        :param fps: The FPS at which the encoding should happen.
        :param cuda_ctx: A cuda context object.
        :param cuda_str: A cuda stream object.
        """
        if self.side != "server":
            raise ValueError("This must be called from server side")

        self.device_id = device_id
        self.fps = round(Fraction(fps), 6)
        self.cuda_ctx = cuda_ctx
        self.cuda_str = cuda_str

        # defer init of encoder because the width/heigh is only know when we have inference results
        self.nvEnc = None

        # We would use VPF for video encoding/decoding, and CVCUDA to do color conversions
        # to and from RGB to NV12 format. These formats are required by VPF to encode/decode
        # video streams. Since CVCUDA can do these conversions much faster on a batch level
        # and since VPF does not work on batches, we would perform these conversions here
        # in this class using CVCUDA. We would pre-allocate the memory required by these
        # conversions upon the first use or whenever the batch size changes. This would allow
        # us to use the 'into' versions of CVCUDA operators without allocating/deallocating
        # memory on every batch. We need to be mindful of the following things when dealing
        # with NV12 format in CVCUDA:
        # NV12 is a complex format and it is not tensor friendly so libraries use a workaround
        # to put the NV12 in a "matrix" form. They put the YUV from NV12 as 3/2 height
        # 1 height is Y luma that is full resolution
        # 1/2 height is UV chroma that is 2x2 down-scaled
        # Hence you would see YUV's H dimension 1.5 times the RGB's H dimension.
        self.cvcuda_YUVtensor_batch = None
        self.surface = None
        self.surf_plane = None

    def server_send(self, frame_batch):
        """
        Encode cvcuda NHWC tensor to VPF YUV_NV12 surface.
        :param frame_batch: NHWC cvcuda RGB tensor
        """

        batch_size, width, height = (
            frame_batch.shape[0],
            frame_batch.shape[2],
            frame_batch.shape[1],
        )
        self.width, self.height = width, height

        # init encoder when width/height is known
        if self.nvEnc is None:
            opts = {
                "preset": "P5",
                "tuning_info": "high_quality",
                "codec": "h264",
                "fps": str(self.fps),
                "s": str(width) + "x" + str(height),
                "bitrate": "10M",
            }
            self.nvEnc = nvc.PyNvEncoder(
                opts,
                self.cuda_ctx.handle,
                self.cuda_str.handle,
            )

        # docs_tag: begin_alloc_cvcuda_videobatchdecoder_vpf
        # Allocate only for the first time or for the last batch.
        if (
            not self.cvcuda_YUVtensor_batch
            or batch_size != self.cvcuda_YUVtensor_batch.shape[0]
        ):
            self.cvcuda_YUVtensor_batch = cvcuda.Tensor(
                (batch_size, (height // 2) * 3, width, 1),
                nvcv.Type.U8,
                nvcv.TensorLayout.NHWC,
            )
        # docs_tag: end_alloc_cvcuda_videobatchdecoder_vpf

        # docs_tag: begin_convert_videobatchencoder_vpf
        # Color convert from RGB to YUV_NV12, in batch, before sending it over to VPF.
        cvcuda.cvtcolor_into(
            self.cvcuda_YUVtensor_batch,
            frame_batch,  # NHWC
            cvcuda.ColorConversion.RGB2YUV_NV12,
        )

        # Convert to torch tensor for VPF
        tensor = torch.as_tensor(self.cvcuda_YUVtensor_batch.cuda(), device="cuda")
        # docs_tag: end_convert_videobatchencoder_vpf

        # docs_tag: begin_encode_videobatchencoder_vpf
        # Encode frames from the batch one by one using VPF.
        ret = []
        for img_idx in range(tensor.shape[0]):
            encoded_packet = self.encode_from_tensor(tensor[img_idx])
            if encoded_packet is not None:
                ret.append(encoded_packet)
        # docs_tag: end_encode_videobatchencoder_vpf

        return ret

    def encode_from_tensor(self, tensor):
        """
        Encode single video frame from torch.cuda.FloatTensor YUV and send to client.
        Tensor must have planar RGB format and be normalized to range [0.0; 1.0].
        """
        assert tensor.dim() == 3
        assert tensor.device.index == self.device_id

        dst_surface = self.tensor_to_surface(tensor)

        if dst_surface.Empty():
            raise RuntimeError("Can not convert to yuv444.")

        # this buffer must be initialized everytime!
        # use a class variable self.encoded_frame here gives a subtle bug:
        # when append the self.encoded_frame in a list, it's basically just append the memory address
        # without copying the content. So the list ends up having all the same elements...
        # fix: re-init everytime to ensure deep copy
        packet = np.ndarray(shape=(0), dtype=np.uint8)
        success = self.nvEnc.EncodeSingleSurface(dst_surface, packet)

        return packet if success else None

    def tensor_to_surface(self, img_tensor):
        """
        Converts torch float tensor into a planar RGB surface.
        """
        if not self.surface:
            if self.cuda_ctx:
                self.surface = nvc.Surface.Make(
                    format=nvc.PixelFormat.NV12,
                    width=self.width,
                    height=self.height,
                    context=self.cuda_ctx.handle,
                )
            else:
                self.surface = nvc.Surface.Make(
                    format=nvc.PixelFormat.NV12,
                    width=self.width,
                    height=self.height,
                    gpu_id=self.device_id,
                )
            self.surf_plane = self.surface.PlanePtr()

        pnvc.TensorToDptr(
            img_tensor,
            self.surf_plane.GpuMem(),
            self.surf_plane.Width(),
            self.surf_plane.Height(),
            self.surf_plane.Pitch(),
            self.surf_plane.ElemSize(),
        )

        return self.surface

    def flush(self):
        ret = []
        while True:
            packet = np.ndarray(shape=(0), dtype=np.uint8)
            success = self.nvEnc.FlushSinglePacket(packet)
            if success:
                ret.append(packet)
            else:
                break
        return ret
