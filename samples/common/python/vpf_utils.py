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

"""
vpf_utils

This file hosts various Video Processing Framework(VPF) related utilities.
"""


from batch import Batch

import os
import av
import logging
import numpy as np
import torch
import nvtx
import nvcv
import cvcuda
from fractions import Fraction
import PyNvCodec as nvc
import PytorchNvCodec as pnvc


# docs_tag: begin_init_videobatchdecoder_vpf
class VideoBatchDecoderVPF:
    def __init__(
        self,
        input_path,
        batch_size,
        device_id,
        cuda_ctx,
    ):
        self.logger = logging.getLogger(__name__)
        self.input_path = input_path
        self.batch_size = batch_size
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx

        # Demuxer is instantiated only to collect required information about
        # certain video file properties.
        nvDemux = nvc.PyFFmpegDemuxer(self.input_path)
        self.fps = nvDemux.Framerate()
        self.total_frames = nvDemux.Numframes()
        self.total_decoded = 0
        self.batch_idx = 0

        # We use VPF to do video decoding. This instance will be allocated when the first
        # batch comes in.
        self.decoder = None

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
        # Hence you would see YUV's H dimension 1.5 times the RGB's H dimension.
        self.cvcuda_RGBtensor_batch = None

        self.logger.info("Using VPF as decoder.")
        # docs_tag: end_init_videobatchdecoder_vpf

    # docs_tag: begin_call_videobatchdecoder_vpf
    def __call__(self):
        nvtx.push_range("decoder.vpf.%d" % self.batch_idx)

        # Check if we have reached the end of the stream. If so, simply return None.
        if self.total_decoded == self.total_frames:
            return None

        # Check if we need to allocate the decoder for its first use.
        if self.decoder is None:
            self.decoder = nvdecoder(
                self.input_path,
                self.device_id,
                self.cuda_ctx,
            )

        # docs_tag: end_alloc_videobatchdecoder_vpf

        # docs_tag: begin_decode_videobatchdecoder_vpf
        # If we are in the last batch size, the total frames left to decode may be
        # less than equal to the batch size.
        if self.total_decoded + self.batch_size > self.total_frames:
            actual_batch_size = self.total_frames - self.total_decoded
        else:
            actual_batch_size = self.batch_size

        # Decode each frame one by one and put them in a list.
        frame_list = [self.decoder.decode_to_tensor() for x in range(actual_batch_size)]

        # Convert 3D list to 4D torch tensor.
        image_tensor_nhwc = torch.stack(frame_list)
        # docs_tag: end_decode_videobatchdecoder_vpf

        # docs_tag: begin_convert_videobatchdecoder_vpf
        # Create a CVCUDA tensor for color conversion YUV->RGB
        # Allocate only for the first time or for the last batch.
        if not self.cvcuda_RGBtensor_batch or actual_batch_size != self.batch_size:
            self.cvcuda_RGBtensor_batch = cvcuda.Tensor(
                (actual_batch_size, self.decoder.h, self.decoder.w, 3),
                nvcv.Type.U8,
                nvcv.TensorLayout.NHWC,
            )

        # Add the batch dim at the end to make it W,H,1 from W,H
        image_tensor_nhwc = torch.unsqueeze(image_tensor_nhwc, -1)
        # Make it a CVCUDA Tensor, C will be 1.
        cvcuda_YUVtensor = cvcuda.as_tensor(image_tensor_nhwc, nvcv.TensorLayout.NHWC)
        # Convert from YUV to RGB. This will be NHWC.
        cvcuda.cvtcolor_into(
            self.cvcuda_RGBtensor_batch, cvcuda_YUVtensor, self.decoder.cvcuda_code
        )
        self.total_decoded += len(frame_list)
        # docs_tag: end_convert_videobatchdecoder_vpf

        # docs_tag: begin_batch_videobatchdecoder_vpf
        # Create a batch instance and set its properties.
        batch = Batch(
            batch_idx=self.batch_idx,
            data=self.cvcuda_RGBtensor_batch,
            fileinfo=self.input_path,
        )
        self.batch_idx += 1

        nvtx.pop_range()

        return batch
        # docs_tag: end_batch_videobatchdecoder_vpf

    def start(self):
        pass

    def join(self):
        pass


# docs_tag: begin_init_videobatchencoder_vpf
class VideoBatchEncoderVPF:
    def __init__(
        self,
        output_path,
        fps,
        device_id,
        cuda_ctx,
    ):
        self.logger = logging.getLogger(__name__)
        self.output_path = output_path
        self.fps = fps
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx

        # We use VPF to do video encoding. This instance will be allocated when the first
        # batch comes in.
        self.encoder = None

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
        self.cvcuda_HWCtensor_batch = None
        self.cvcuda_YUVtensor_batch = None
        self.input_layout = "NCHW"
        self.gpu_input = True
        self.output_file_name = None

        self.logger.info("Using VPF as encoder.")
        # docs_tag: end_init_videobatchencoder_vpf

    # docs_tag: begin_call_videobatchencoder_vpf
    def __call__(self, batch):
        nvtx.push_range("encoder.vpf.%d" % batch.batch_idx)

        # Get the name of the original video file read by the decoder. We would use
        # the same filename to save the output video.
        file_name = os.path.splitext(os.path.basename(batch.fileinfo))[0]
        self.output_file_name = os.path.join(self.output_path, "out_%s.mp4" % file_name)

        # Check if we need to allocate the encoder for its first use.
        if self.encoder is None:
            self.encoder = nvencoder(
                self.device_id,
                batch.data.shape[3],
                batch.data.shape[2],
                self.fps,
                self.output_file_name,
                self.cuda_ctx,
            )

        # docs_tag: end_alloc_videobatchdecoder_vpf

        # docs_tag: begin_alloc_cvcuda_videobatchdecoder_vpf
        # Create 2 CVCUDA tensors: reformat NCHW->NHWC and color conversion RGB->YUV
        current_batch_size = batch.data.shape[0]
        height, width = batch.data.shape[2], batch.data.shape[3]
        # Allocate only for the first time or for the last batch.
        if (
            not self.cvcuda_HWCtensor_batch
            or current_batch_size != self.cvcuda_HWCtensor_batch.shape[0]
        ):
            self.cvcuda_HWCtensor_batch = cvcuda.Tensor(
                (current_batch_size, height, width, 3),
                nvcv.Type.U8,
                nvcv.TensorLayout.NHWC,
            )
            self.cvcuda_YUVtensor_batch = cvcuda.Tensor(
                (current_batch_size, (height // 2) * 3, width, 1),
                nvcv.Type.U8,
                nvcv.TensorLayout.NHWC,
            )
        # docs_tag: end_alloc_cvcuda_videobatchdecoder_vpf

        # docs_tag: begin_convert_videobatchencoder_vpf
        # Convert RGB to NV12, in batch, before sending it over to VPF.
        # Convert to CVCUDA tensor
        cvcuda_tensor = cvcuda.as_tensor(batch.data, nvcv.TensorLayout.NCHW)
        # Reformat
        cvcuda.reformat_into(self.cvcuda_HWCtensor_batch, cvcuda_tensor)
        # Color convert from RGB to YUV_NV12
        cvcuda.cvtcolor_into(
            self.cvcuda_YUVtensor_batch,
            self.cvcuda_HWCtensor_batch,
            cvcuda.ColorConversion.RGB2YUV_NV12,
        )

        # Convert back to torch tensor
        tensor = torch.as_tensor(self.cvcuda_YUVtensor_batch.cuda(), device="cuda")

        # docs_tag: end_convert_videobatchencoder_vpf

        # docs_tag: begin_encode_videobatchencoder_vpf
        # Encode frames from the batch one by one using VPF.
        for img_idx in range(tensor.shape[0]):
            img = tensor[img_idx]
            self.encoder.encode_from_tensor(img)

        nvtx.pop_range()

        # docs_tag: end_encode_videobatchencoder_vpf

    def start(self):
        pass

    def join(self):
        self.encoder.flush()
        self.logger.info("Wrote: %s" % self.output_file_name)
        pass


class nvdecoder:
    def __init__(
        self,
        enc_file,
        device_id,
        cuda_ctx,
    ):
        """
        Create instance of HW-accelerated video decoder.
        :param enc_file: Full path to the MP4 file that needs to be decoded.
        :param device_id: id of video card which will be used for decoding & processing.
        :param cuda_ctx: A cuda context object.
        """
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx
        # Demuxer is instantiated only to collect required information about
        # certain video file properties.
        nvDemux = nvc.PyFFmpegDemuxer(enc_file)
        self.w, self.h = nvDemux.Width(), nvDemux.Height()
        self.fps = nvDemux.Framerate()
        self.total_frames = nvDemux.Numframes()

        # In case sample aspect ratio isn't 1:1 we will re-scale the decoded
        # frame to maintain uniform 1:1 ratio across the pipeline.
        sar = 8.0 / 9.0
        self.fixed_h = self.h
        self.fixed_w = int(self.w * sar)

        self.pix_fmt = nvDemux.Format()
        is_yuv420 = (
            nvc.PixelFormat.YUV420 == self.pix_fmt
            or nvc.PixelFormat.NV12 == self.pix_fmt
        )
        is_yuv444 = nvc.PixelFormat.YUV444 == self.pix_fmt

        # Set CVCUDA color conversion code to do YUV->RGB
        self.cvcuda_code = None
        if is_yuv420:
            self.cvcuda_code = cvcuda.ColorConversion.YUV2RGB_NV12
        elif is_yuv444:
            self.cvcuda_code = cvcuda.ColorConversion.YUV2RGB

        codec = nvDemux.Codec()
        is_hevc = nvc.CudaVideoCodec.HEVC == codec

        # YUV420 or YUV444 sampling formats are supported by Nvdec
        self.is_hw_dec = is_yuv420 or is_yuv444

        # But YUV444 HW decode is supported for HEVC only
        if self.is_hw_dec and is_yuv444 and not is_hevc:
            self.is_hw_dec = False

        if self.is_hw_dec:
            # Nvdec supports NV12 (resampled YUV420) and YUV444 formats
            if self.cuda_ctx:
                self.nvDec = nvc.PyNvDecoder(
                    input=enc_file,
                    context=self.cuda_ctx.handle,
                    stream=cvcuda.Stream.current.handle,
                )
            else:
                self.nvDec = nvc.PyNvDecoder(
                    input=enc_file,
                    gpu_id=self.device_id,
                )
        else:
            raise ValueError(
                "Current combination of hardware and the video file being read does not "
                "hardware accelerated decoding."
            )

    # docs_tag: begin_imp_nvdecoder
    def decode_hw(self, seek_ctx=None):
        """
        Decode single video frame with Nvdec, convert it to planar RGB.
        """
        # Decode with HW decoder
        if seek_ctx is None:
            dec_surface = self.nvDec.DecodeSingleSurface()
        else:
            dec_surface = self.nvDec.DecodeSingleSurface(seek_ctx)
        if not dec_surface or dec_surface.Empty():
            raise RuntimeError("Can not decode frame.")

        return dec_surface

    def decode_to_tensor(self, *args, **kwargs):
        """
        Decode single video frame, convert it to torch.cuda.FloatTensor.
        Image will be planar RGB normalized to range [0.0; 1.0].
        """
        if self.is_hw_dec:
            dec_surface = self.decode_hw(*args, **kwargs)
        else:
            raise ValueError(
                "Current combination of hardware and the video file being read does not "
                "hardware accelerated decoding."
            )

        if not dec_surface or dec_surface.Empty():
            raise RuntimeError("Can not decode surface.")

        surf_plane = dec_surface.PlanePtr()

        img_tensor = pnvc.makefromDevicePtrUint8(
            surf_plane.GpuMem(),
            surf_plane.Width(),
            surf_plane.Height(),
            surf_plane.Pitch(),
            surf_plane.ElemSize(),
        )
        if img_tensor is None:
            raise RuntimeError("Can not export to tensor.")

        return img_tensor

    # docs_tag: end_imp_nvdecoder


class nvencoder:
    def __init__(
        self,
        device_id,
        width,
        height,
        fps,
        enc_file,
        cuda_ctx,
    ):
        """
        Create instance of HW-accelerated video encoder.
        :param device_id: id of video card which will be used for encoding & processing.
        :param width: encoded frame width.
        :param height: encoded frame height.
        :param fps: The FPS at which the encoding should happen.
        :param enc_file: path to encoded video file.
        :param cuda_ctx: A cuda context object
        """
        self.device_id = device_id
        self.fps = round(Fraction(fps), 6)
        self.enc_file = enc_file
        self.cuda_ctx = cuda_ctx

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
            cvcuda.Stream.current.handle,
        )
        self.pts_time = 0
        self.delta_t = 1  # Increment the packets' timestamp by this much.
        self.encoded_frame = np.ndarray(shape=(0), dtype=np.uint8)
        self.container = av.open(enc_file, "w")
        self.avstream = self.container.add_stream("h264", rate=fps)
        self.avstream.width = width
        self.avstream.height = height
        self.avstream.time_base = 1 / Fraction(fps)  # 1/fps would be our scale.
        self.surface = None
        self.surf_plane = None

    def width(self):
        """
        Gets the actual video frame width from the encoder.
        """
        return self.nvEnc.Width()

    def height(self):
        """
        Gets the actual video frame height from the encoder.
        """
        return self.nvEnc.Height()

    # docs_tag: begin_imp_nvencoder
    def tensor_to_surface(self, img_tensor):
        """
        Converts torch float tensor into a planar RGB surface.
        """
        if not self.surface:
            if self.cuda_ctx:
                self.surface = nvc.Surface.Make(
                    format=nvc.PixelFormat.NV12,
                    width=self.width(),
                    height=self.height(),
                    context=self.cuda_ctx.handle,
                )
            else:
                self.surface = nvc.Surface.Make(
                    format=nvc.PixelFormat.NV12,
                    width=self.width(),
                    height=self.height(),
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

    def encode_from_tensor(self, tensor):
        """
        Encode single video frame from torch.cuda.FloatTensor.
        Tensor must have planar RGB format and be normalized to range [0.0; 1.0].
        Shape of the tensor must be (3, height, width).
        """
        assert tensor.dim() == 3
        assert tensor.device.index == self.device_id

        dst_surface = self.tensor_to_surface(tensor)

        if dst_surface.Empty():
            raise RuntimeError("Can not convert to yuv444.")

        success = self.nvEnc.EncodeSingleSurface(dst_surface, self.encoded_frame)

        if success:
            self.write_frame(
                self.encoded_frame,
                self.pts_time,
                self.fps,
                self.avstream,
                self.container,
            )
            self.pts_time += self.delta_t

    # docs_tag: end_imp_nvencoder

    # docs_tag: begin_writeframe_nvencoder
    def write_frame(self, encoded_frame, pts_time, fps, stream, container):
        encoded_bytes = bytearray(encoded_frame)
        pkt = av.packet.Packet(encoded_bytes)
        pkt.pts = pts_time
        pkt.dts = pts_time
        pkt.stream = stream
        pkt.time_base = 1 / Fraction(fps)
        container.mux(pkt)

    def flush(self):
        packets = np.ndarray(shape=(0), dtype=np.uint8)

        success = self.nvEnc.Flush(packets)
        if success:
            self.write_frame(
                self.encoded_frame,
                self.pts_time,
                self.fps,
                self.avstream,
                self.container,
            )
            self.pts_time += self.delta_t

    # docs_tag: end_writeframe_nvencoder
