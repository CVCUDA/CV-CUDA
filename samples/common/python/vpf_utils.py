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


import av
import torch
import numpy as np
from fractions import Fraction
import PyNvCodec as nvc
import PytorchNvCodec as pnvc


class nvdecoder:
    def __init__(self, enc_file: str, gpu_id: int):
        """
        Create instance of HW-accelerated video decoder.
        :param gpu_id: id of video card which will be used for decoding & processing.
        :param enc_file: path to encoded video file.
        """
        # Demuxer is instantiated only to collect required information about
        # certain video file properties.
        self.device_id = gpu_id
        nvDemux = nvc.PyFFmpegDemuxer(enc_file)
        self.w, self.h = nvDemux.Width(), nvDemux.Height()
        self.fps = nvDemux.Framerate()
        self.total_frames = nvDemux.Numframes()

        # Determine color space and color range for accurate conversion to RGB.
        self.cspace = nvDemux.ColorSpace()
        self.crange = nvDemux.ColorRange()
        if nvc.ColorSpace.UNSPEC == self.cspace:
            self.cspace = nvc.ColorSpace.BT_601
        if nvc.ColorRange.UDEF == self.crange:
            self.crange = nvc.ColorRange.JPEG
        self.cc_ctx = nvc.ColorspaceConversionContext(self.cspace, self.crange)

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

        codec = nvDemux.Codec()
        is_hevc = nvc.CudaVideoCodec.HEVC == codec

        # YUV420 or YUV444 sampling formats are supported by Nvdec
        self.is_hw_dec = is_yuv420 or is_yuv444

        # But YUV444 HW decode is supported for HEVC only
        if self.is_hw_dec and is_yuv444 and not is_hevc:
            self.is_hw_dec = False

        if self.is_hw_dec:
            # Nvdec supports NV12 (resampled YUV420) and YUV444 formats
            self.nvDec = nvc.PyNvDecoder(enc_file, self.device_id)
        else:
            # No HW decoding acceleration, fall back to CPU back-end.
            self.to_gpu = nvc.PyFrameUploader(
                self.w, self.h, self.pix_fmt, self.device_id
            )

        if is_yuv420:
            # YUV420 videos will be decoded by Nvdec to NV12 which is the
            # same thing but resampled (U and V planes interleaved).
            self.to_rgb = nvc.PySurfaceConverter(
                self.w,
                self.h,
                nvc.PixelFormat.NV12,
                nvc.PixelFormat.RGB,
                self.device_id,
            )
        elif is_yuv444:
            self.to_rgb = nvc.PySurfaceConverter(
                self.w, self.h, self.pix_fmt, nvc.PixelFormat.RGB, self.device_id
            )
        else:
            self.to_rgb = nvc.PySurfaceConverter(
                self.w, self.h, self.pix_fmt, nvc.PixelFormat.RGB_PLANAR, self.device_id
            )
            self.cc_conv = nvc.PySurfaceColorconversion(
                self.w, self.h, nvc.PixelFormat.YUV422, self.device_id
            )

        self.to_pln = nvc.PySurfaceConverter(
            self.w,
            self.h,
            nvc.PixelFormat.RGB,
            nvc.PixelFormat.RGB_PLANAR,
            self.device_id,
        )

        if self.h != self.fixed_h:
            self.to_sar = nvc.PySurfaceResizer(
                self.fixed_w, self.fixed_h, nvc.PixelFormat.RGB_PLANAR, self.device_id
            )
        else:
            self.to_sar = None

    def decode_sw(self, dec_frame, *args, **kwargs) -> nvc.Surface:
        """
        This is called when input video isn't supported by Nvdec HW.
        Fallback decode single video frame on CPU.
        Upload to GPU, convert from YUV422 to planar RGB.
        """
        # Upload to GPU
        dec_surf = self.to_gpu.UploadSingleFrame(dec_frame)
        if not dec_surf or dec_surf.Empty():
            raise RuntimeError("Can not upload frame to gpu.")

        # Convert to planar RGB
        # So far YUV422 > RGB_PLANAR doesn't support MPEG color range, so have
        # to do an ugly fix.
        if nvc.PixelFormat.YUV422 == self.pix_fmt:
            rgb_pln = self.cc_conv.Execute(dec_surf)
        else:
            # Convert to packed RGB
            rgb_int = self.to_rgb.Execute(dec_surf, self.cc_ctx)
            if not rgb_int or rgb_int.Empty():
                raise RuntimeError("Can not convert nv12 -> rgb.")

            # Convert to planar RGB
            rgb_pln = self.to_pln.Execute(rgb_int, self.cc_ctx)
            if not rgb_pln or rgb_pln.Empty():
                raise RuntimeError("Can not convert rgb -> rgb planar.")

        if not rgb_pln or rgb_pln.Empty():
            raise RuntimeError("Can not convert rgb -> rgb planar.")

        # Resize if necessary to maintain 1:1 SAR
        if self.to_sar is not None:
            rgb_pln = self.to_sar.Execute(rgb_pln)

        return rgb_pln

    def decode_hw(self, seek_ctx=None) -> nvc.Surface:
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

        # Convert to packed RGB
        rgb_int = self.to_rgb.Execute(dec_surface, self.cc_ctx)
        if not rgb_int or rgb_int.Empty():
            raise RuntimeError("Can not convert nv12 -> rgb.")

        # Convert to planar RGB
        rgb_pln = self.to_pln.Execute(rgb_int, self.cc_ctx)
        if not rgb_pln or rgb_pln.Empty():
            raise RuntimeError("Can not convert rgb -> rgb planar.")

        # Resize if necessary to maintain 1:1 SAR
        if self.to_sar:
            rgb_pln = self.to_sar.Execute(rgb_pln)

        return rgb_pln

    def decode_to_tensor(self, *args, **kwargs) -> torch.Tensor:
        """
        Decode single video frame, convert it to torch.cuda.FloatTensor.
        Image will be planar RGB normalized to range [0.0; 1.0].
        """
        dec_surface = None

        if self.is_hw_dec:
            dec_surface = self.decode_hw(*args, **kwargs)
        else:
            dec_surface = self.decode_sw(*args, **kwargs)

        if not dec_surface or dec_surface.Empty():
            raise RuntimeError("Can not convert rgb -> rgb planar.")

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

        img_tensor.resize_(3, int(surf_plane.Height() / 3), surf_plane.Width())

        return img_tensor


class cconverter:
    """
    Colorspace conversion chain.
    """

    def __init__(self, width: int, height: int, gpu_id: int):
        self.gpu_id = gpu_id
        self.w = width
        self.h = height
        self.chain = []

    def add(self, src_fmt: nvc.PixelFormat, dst_fmt: nvc.PixelFormat) -> None:
        self.chain.append(
            nvc.PySurfaceConverter(self.w, self.h, src_fmt, dst_fmt, self.gpu_id)
        )

    def Execute(self, src_surface: nvc.Surface, cc=None) -> nvc.Surface:
        surf = src_surface
        if not cc:
            cc = nvc.ColorspaceConversionContext(
                nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG
            )

        for cvt in self.chain:
            surf = cvt.Execute(surf, cc)
            if surf.Empty():
                raise RuntimeError("Failed to perform color conversion")

        return surf.Clone(self.gpu_id)


class nvencoder:
    def __init__(
        self,
        gpu_id: int,
        width: int,
        height: int,
        fps: float,
        enc_file: str,
    ) -> None:
        """
        Create instance of HW-accelerated video encoder.
        :param gpu_id: id of video card which will be used for encoding & processing.
        :param width: encoded frame width.
        :param height: encoded frame height.
        :param enc_file: path to encoded video file.
        :param options: dictionary with encoder initialization options.
        """

        self.to_nv12 = cconverter(width, height, gpu_id=gpu_id)
        self.to_nv12.add(nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB)
        self.to_nv12.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
        self.to_nv12.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)
        self.cc_ctx = nvc.ColorspaceConversionContext(
            nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG
        )
        fps = round(Fraction(fps), 6)

        opts = {
            "preset": "P5",
            "tuning_info": "high_quality",
            "codec": "h264",
            "fps": str(fps),
            "s": str(width) + "x" + str(height),
            "bitrate": "10M",
        }

        self.gpu_id = gpu_id
        self.fps = fps
        self.enc_file = enc_file
        self.nvEnc = nvc.PyNvEncoder(opts, gpu_id)
        self.pts_time = 0
        self.delta_t = 1  # Increment the packets' timestamp by this much.
        self.encoded_frame = np.ndarray(shape=(0), dtype=np.uint8)
        self.container = av.open(enc_file, "w")
        self.stream = self.container.add_stream("h264", rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.time_base = 1 / Fraction(fps)  # 1/fps would be our scale.

    def width(self) -> int:
        """
        Get video frame width.
        """
        return self.nvEnc.Width()

    def height(self) -> int:
        """
        Get video frame height.
        """
        return self.nvEnc.Height()

    def tensor_to_surface(self, img_tensor: torch.tensor) -> nvc.Surface:
        """
        Converts cuda float tensor to planar rgb surface.
        """
        if len(img_tensor.shape) != 3 and img_tensor.shape[0] != 3:
            raise RuntimeError("Shape of the tensor must be (3, height, width)")

        _, tensor_h, tensor_w = img_tensor.shape
        assert tensor_w == self.width() and tensor_h == self.height()

        surface = nvc.Surface.Make(
            nvc.PixelFormat.RGB_PLANAR,
            tensor_w,
            tensor_h,
            self.gpu_id,
        )
        surf_plane = surface.PlanePtr()
        pnvc.TensorToDptr(
            img_tensor,
            surf_plane.GpuMem(),
            surf_plane.Width(),
            surf_plane.Height(),
            surf_plane.Pitch(),
            surf_plane.ElemSize(),
        )

        return surface.Clone()

    def encode_from_tensor(self, tensor: torch.Tensor):
        """
        Encode single video frame from torch.cuda.FloatTensor.
        Tensor must have planar RGB format and be normalized to range [0.0; 1.0].
        Shape of the tensor must be (3, height, width).
        """
        assert tensor.dim() == 3
        assert self.gpu_id == tensor.device.index

        surface_rgb = self.tensor_to_surface(tensor)
        dst_surface = self.to_nv12.Execute(surface_rgb, self.cc_ctx)
        if dst_surface.Empty():
            raise RuntimeError("Can not convert to yuv444.")

        success = self.nvEnc.EncodeSingleSurface(dst_surface, self.encoded_frame)
        if success:
            encoded_bytes = bytearray(self.encoded_frame)
            pkt = av.packet.Packet(encoded_bytes)
            pkt.pts = self.pts_time
            pkt.dts = self.pts_time
            pkt.stream = self.stream
            pkt.time_base = 1 / Fraction(self.fps)
            self.pts_time += self.delta_t
            self.container.mux(pkt)

    def flush(self):
        packets = np.ndarray(shape=(0), dtype=np.uint8)

        success = self.nvEnc.Flush(packets)
        if success:
            encoded_bytes = bytearray(packets)
            pkt = av.packet.Packet(encoded_bytes)
            pkt.pts = self.pts_time
            pkt.dts = self.pts_time
            pkt.stream = self.stream
            pkt.time_base = 1 / Fraction(self.fps)
            self.pts_time += self.delta_t
            self.container.mux(pkt)
        else:
            print("Error during flush")
