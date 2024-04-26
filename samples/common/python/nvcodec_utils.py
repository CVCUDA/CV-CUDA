# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
nvcodec_utils

This file hosts various helpers for NV codecs exist.
"""


import os
import sys
import av
import logging
import glob
import numpy as np
import torch
import nvcv
import cvcuda
from fractions import Fraction
import itertools
import PyNvVideoCodec as nvvc
from nvidia import nvimgcodec

from pathlib import Path

# Bring module folders from the samples directory into our path so that
# we can import modules from it.
samples_dir = Path(os.path.abspath(__file__)).parents[2]  # samples/
sys.path.insert(0, os.path.join(samples_dir, ""))

from common.python.batch import Batch  # noqa: E402

pixel_format_to_cvcuda_code = {
    nvvc.Pixel_Format.YUV444: cvcuda.ColorConversion.YUV2RGB,
    nvvc.Pixel_Format.NV12: cvcuda.ColorConversion.YUV2RGB_NV12,
}


class AppCAI:
    def __init__(self, shape, stride, typestr, gpualloc):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "strides": stride,
            "data": (int(gpualloc), False),
            "typestr": typestr,
            "version": 3,
        }


# docs_tag: begin_videobatchdecoder_pyvideocodec
class VideoBatchDecoder:
    def __init__(
        self,
        input_path,
        batch_size,
        device_id,
        cuda_ctx,
        cuda_stream,
        cvcuda_perf,
    ):
        # docs_tag: begin_init_videobatchdecoder_pyvideocodec
        self.logger = logging.getLogger(__name__)
        self.input_path = input_path
        self.batch_size = batch_size
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx
        self.cuda_stream = cuda_stream
        self.cvcuda_perf = cvcuda_perf
        self.total_decoded = 0
        self.batch_idx = 0
        self.decoder = None
        self.cvcuda_RGBtensor_batch = None
        nvDemux = nvvc.PyNvDemuxer(self.input_path)
        self.fps = nvDemux.FrameRate()
        self.logger.info("Using PyNvVideoCodec decoder version: %s" % nvvc.__version__)
        # docs_tag: end_init_videobatchdecoder_pyvideocodec

    # docs_tag: begin_call_videobatchdecoder_pyvideocodec
    def __call__(self):
        self.cvcuda_perf.push_range("decoder.pyVideoCodec")

        # docs_tag: begin_alloc_videobatchdecoder_pyvideocodec
        # Check if we need to allocate the decoder for its first use.
        if self.decoder is None:
            self.decoder = nvVideoDecoder(
                self.input_path, self.device_id, self.cuda_ctx, self.cuda_stream
            )
        # docs_tag: end_alloc_videobatchdecoder_pyvideocodec

        # docs_tag: begin_decode_videobatchdecoder_pyvideocodec
        # Get the NHWC YUV tensor from the decoder
        cvcuda_YUVtensor = self.decoder.get_next_frames(self.batch_size)

        # Check if we are done decoding
        if cvcuda_YUVtensor is None:
            self.cvcuda_perf.pop_range()
            return None

        # Check the code for the color conversion based in the pixel format
        cvcuda_code = pixel_format_to_cvcuda_code.get(self.decoder.pixelFormat)
        if cvcuda_code is None:
            raise ValueError(f"Unsupported pixel format: {self.decoder.pixelFormat}")

        # Check layout to make sure it is what we expected
        if cvcuda_YUVtensor.layout != "NHWC":
            raise ValueError("Unexpected tensor layout, NHWC expected.")

        # this may be different than batch size since last frames may not be a multiple of batch size
        actual_batch_size = cvcuda_YUVtensor.shape[0]

        # docs_tag: end_decode_videobatchdecoder_pyvideocodec

        # docs_tag: begin_convert_videobatchdecoder_pyvideocodec
        # Create a CVCUDA tensor for color conversion YUV->RGB
        # Allocate only for the first time or for the last batch.
        if not self.cvcuda_RGBtensor_batch or actual_batch_size != self.batch_size:
            self.cvcuda_RGBtensor_batch = cvcuda.Tensor(
                (actual_batch_size, self.decoder.h, self.decoder.w, 3),
                nvcv.Type.U8,
                nvcv.TensorLayout.NHWC,
            )

        # Convert from YUV to RGB. Conversion code is based on the pixel format.
        cvcuda.cvtcolor_into(self.cvcuda_RGBtensor_batch, cvcuda_YUVtensor, cvcuda_code)

        self.total_decoded += actual_batch_size
        # docs_tag: end_convert_videobatchdecoder_pyvideocodec

        # docs_tag: begin_batch_videobatchdecoder_pyvideocodec
        # Create a batch instance and set its properties.
        batch = Batch(
            batch_idx=self.batch_idx,
            data=self.cvcuda_RGBtensor_batch,
            fileinfo=self.input_path,
        )
        self.batch_idx += 1

        self.cvcuda_perf.pop_range()
        return batch
        # docs_tag: end_call_videobatchdecoder_pyvideocodec

    def start(self):
        pass

    def join(self):
        pass


# docs_tag: end_videobatchdecoder_pyvideocodec

# docs_tag: begin_imp_nvvideodecoder
class nvVideoDecoder:
    def __init__(self, enc_file, device_id, cuda_ctx, stream):
        """
        Create instance of HW-accelerated video decoder.
        :param enc_file: Full path to the MP4 file that needs to be decoded.
        :param device_id: id of video card which will be used for decoding & processing.
        :param cuda_ctx: A cuda context object.
        """
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx
        self.input_path = enc_file
        self.stream = stream
        # Demuxer is instantiated only to collect required information about
        # certain video file properties.
        self.nvDemux = nvvc.PyNvDemuxer(self.input_path)
        self.nvDec = nvvc.CreateDecoder(
            gpuid=0,
            codec=self.nvDemux.GetNvCodecId(),
            cudacontext=self.cuda_ctx.handle,
            cudastream=self.stream.handle,
            enableasyncallocations=False,
        )

        self.w, self.h = self.nvDemux.Width(), self.nvDemux.Height()
        self.pixelFormat = self.nvDec.GetPixelFormat()
        # In case sample aspect ratio isn't 1:1 we will re-scale the decoded
        # frame to maintain uniform 1:1 ratio across the pipeline.
        sar = 8.0 / 9.0
        self.fixed_h = self.h
        self.fixed_w = int(self.w * sar)

    # frame iterator
    def generate_decoded_frames(self):
        for packet in self.nvDemux:
            for decodedFrame in self.nvDec.Decode(packet):
                nvcvTensor = nvcv.as_tensor(
                    nvcv.as_image(decodedFrame.nvcv_image(), nvcv.Format.U8)
                )
                if nvcvTensor.layout == "NCHW":
                    # This will re-format the NCHW tensor to a NHWC tensor which will create
                    # a copy in the CUDA device decoded frame will go out of scope and the
                    # backing memory will be available by the decoder.
                    yield cvcuda.reformat(nvcvTensor, "NHWC")
                else:
                    raise ValueError("Unexpected tensor layout, NCHW expected.")

    def get_next_frames(self, N):
        decoded_frames = list(itertools.islice(self.generate_decoded_frames(), N))
        if len(decoded_frames) == 0:
            return None
        elif len(decoded_frames) == 1:  # this case we dont need stack the tensor
            return decoded_frames[0]
        else:
            # convert from list of tensors to a single tensor (NHWC)
            tensorNHWC = cvcuda.stack(decoded_frames)
            return tensorNHWC


# docs_tag: end_imp_nvvideodecoder

# docs_tag: begin_init_videobatchencoder_pyvideocodec
class VideoBatchEncoder:
    def __init__(
        self,
        output_path,
        fps,
        device_id,
        cuda_ctx,
        cuda_stream,
        cvcuda_perf,
    ):
        self.logger = logging.getLogger(__name__)
        self.output_path = output_path
        self.fps = fps
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx
        self.cuda_stream = cuda_stream
        self.cvcuda_perf = cvcuda_perf

        self.encoder = None
        self.cvcuda_HWCtensor_batch = None
        self.cvcuda_YUVtensor_batch = None
        self.input_layout = "NCHW"
        self.gpu_input = True
        self.output_file_name = None

        self.logger.info("Using PyNvVideoCodec encoder version: %s" % nvvc.__version__)
        # docs_tag: end_init_videobatchencoder_pyvideocodec

    # docs_tag: begin_call_videobatchencoder_pyvideocodec
    def __call__(self, batch):
        self.cvcuda_perf.push_range("encoder.pyVideoCodec")

        # Get the name of the original video file read by the decoder. We would use
        # the same filename to save the output video.
        file_name = os.path.splitext(os.path.basename(batch.fileinfo))[0]
        self.output_file_name = os.path.join(self.output_path, "out_%s.mp4" % file_name)

        assert isinstance(batch.data, torch.Tensor)

        # docs_tag: begin_alloc_cvcuda_videobatchencoder_pyvideocodec
        # Check if we need to allocate the encoder for its first use.
        if self.encoder is None:
            self.encoder = nvVideoEncoder(
                self.device_id,
                batch.data.shape[3],
                batch.data.shape[2],
                self.fps,
                self.output_file_name,
                self.cuda_ctx,
                self.cuda_stream,
                "NV12",
            )
        # docs_tag: end_alloc_cvcuda_videobatchencoder_pyvideocodec

        # docs_tag: begin_convert_videobatchencoder_pyvideocodec

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

        # Convert RGB to NV12, in batch, before sending it over to pyVideoCodec.
        # Convert to CVCUDA tensor
        cvcuda_tensor = cvcuda.as_tensor(batch.data, nvcv.TensorLayout.NCHW)

        # Reformat NCHW to NHWC
        cvcuda.reformat_into(self.cvcuda_HWCtensor_batch, cvcuda_tensor)

        # Color convert from RGB to YUV_NV12
        cvcuda.cvtcolor_into(
            self.cvcuda_YUVtensor_batch,
            self.cvcuda_HWCtensor_batch,
            cvcuda.ColorConversion.RGB2YUV_NV12,
        )

        # Convert back to torch tensor we are NV12
        tensor = torch.as_tensor(self.cvcuda_YUVtensor_batch.cuda(), device="cuda")
        # docs_tag: end_convert_videobatchencoder_pyvideocodec

        # docs_tag: begin_encode_videobatchencoder_pyvideocodec
        # Encode frames from the batch one by one using pyVideoCodec.
        for img_idx in range(tensor.shape[0]):
            img = tensor[img_idx]
            self.encoder.encode_from_tensor(img)

        self.cvcuda_perf.pop_range()

    def start(self):
        pass

    def join(self):
        # self.encoder.flush()
        self.logger.info("Wrote: %s" % self.output_file_name)


# docs_tag: end_init_videobatchencoder_pyvideocodec

# docs_tag: begin_imp_nvvideoencoder
class nvVideoEncoder:
    def __init__(
        self,
        device_id,
        width,
        height,
        fps,
        enc_file,
        cuda_ctx,
        cuda_stream,
        format,
    ):
        """
        Create instance of HW-accelerated video encoder.
        :param device_id: id of video card which will be used for encoding & processing.
        :param width: encoded frame width.
        :param height: encoded frame height.
        :param fps: The FPS at which the encoding should happen.
        :param enc_file: path to encoded video file.
        :param cuda_ctx: A cuda context object
        :param format: The format of the encoded video file.
                (e.g. "NV12", "YUV444" see NvPyVideoEncoder docs for more info)
        """
        self.device_id = device_id
        self.fps = round(Fraction(fps), 6)
        self.enc_file = enc_file
        self.cuda_ctx = cuda_ctx
        self.cuda_stream = cuda_stream

        self.pts_time = 0
        self.delta_t = 1  # Increment the packets' timestamp by this much.
        self.encoded_frame = np.ndarray(shape=(0), dtype=np.uint8)
        self.container = av.open(enc_file, "w")
        self.avstream = self.container.add_stream("h264", rate=self.fps)

        aligned_value = 0
        if width % 16 != 0:
            aligned_value = 16 - (width % 16)
        aligned_width = width + aligned_value
        width = aligned_width

        self.avstream.width = width
        self.avstream.height = height

        self.avstream.time_base = 1 / Fraction(self.fps)
        self.surface = None
        self.surf_plane = None

        self.tmpTensor = None

        self.nvEnc = nvvc.CreateEncoder(
            self.avstream.width,
            self.avstream.height,
            format,
            codec="h264",
            preset="P4",
            cudastream=cuda_stream.handle,
        )

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

    # docs_tag: begin_imp_nvvideoencoder

    def encode_from_tensor(self, tensor):

        # Create a CUDA array interface object wit 2 planes one for luma and CrCb for NV12
        objCAI = []
        # Need to compute the address of the Y plane and the interleaved chroma plane
        data = (
            tensor.storage().data_ptr()
            + tensor.storage_offset() * tensor.element_size()
        )
        objCAI.append(
            AppCAI(
                (self.avstream.height, self.avstream.width, 1),
                (self.avstream.width, 1, 1),
                "|u1",
                data,
            )
        )
        chromaAlloc = int(data) + self.avstream.width * self.avstream.height
        objCAI.append(
            AppCAI(
                (int(self.avstream.height / 2), int(self.avstream.width / 2), 2),
                (self.avstream.width, 2, 1),
                "|u1",
                chromaAlloc,
            )
        )
        # Encode the frame takes CUDA array interface object as input
        self.encoded_frame = self.nvEnc.Encode(objCAI)
        self.write_frame(
            self.encoded_frame,
            self.pts_time,
            self.fps,
            self.avstream,
            self.container,
        )
        self.pts_time += self.delta_t

    # docs_tag: end_imp_nvvideoencoder

    # docs_tag: begin_writeframe_nvvideoencoder
    def write_frame(self, encoded_frame, pts_time, fps, stream, container):
        encoded_bytes = bytearray(encoded_frame)
        pkt = av.packet.Packet(encoded_bytes)
        pkt.pts = pts_time
        pkt.dts = pts_time
        pkt.stream = stream
        pkt.time_base = 1 / Fraction(fps)
        container.mux(pkt)

    # docs_tag: end_writeframe_nvvideoencoder

    def flush(self):
        encoded_bytes = self.nvEnc.EndEncode()
        if encoded_bytes:
            self.write_frame(
                encoded_bytes,
                self.pts_time,
                self.fps,
                self.avstream,
                self.container,
            )
        self.pts_time += self.delta_t
        self.container.close()


# docs_tag: end_imp_nvvideoencoder

# docs_tag: begin_imagebatchdecoder_nvimagecodec
class ImageBatchDecoder:
    def __init__(
        self,
        input_path,
        batch_size,
        device_id,
        cuda_ctx,
        cuda_stream,
        cvcuda_perf,
    ):

        # docs_tag: begin_init_imagebatchdecoder_nvimagecodec
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.input_path = input_path
        self.device_id = device_id
        self.total_decoded = 0
        self.batch_idx = 0
        self.cuda_ctx = cuda_ctx
        self.cuda_stream = cuda_stream
        self.cvcuda_perf = cvcuda_perf
        self.decoder = nvimgcodec.Decoder(device_id=device_id)

        # docs_tag: begin_parse_imagebatchdecoder_nvimagecodec
        if os.path.isfile(self.input_path):
            if os.path.splitext(self.input_path)[1] == ".jpg":
                # Read the input image file.
                self.file_names = [self.input_path] * self.batch_size
                # We will use the nvImageCodec based decoder on the GPU in case of images.
                # This will be allocated once during the first run or whenever a batch
                # size change happens.
            else:
                raise ValueError("Unable to read file %s as image." % self.input_path)

        elif os.path.isdir(self.input_path):
            # It is a directory. Grab file names of all JPG images.
            self.file_names = glob.glob(os.path.join(self.input_path, "*.jpg"))
            self.logger.info("Found a total of %d JPEG images." % len(self.file_names))

        else:
            raise ValueError(
                "Unknown expression given as input_path: %s." % self.input_path
            )

        # docs_tag: end_parse_imagebatchdecoder_nvimagecodec

        # docs_tag: begin_batch_imagebatchdecoder_nvimagecodec
        self.file_name_batches = [
            self.file_names[i : i + self.batch_size]  # noqa: E203
            for i in range(0, len(self.file_names), self.batch_size)
        ]
        # docs_tag: end_batch_imagebatchdecoder_nvimagecodec

        self.max_image_size = 1024 * 1024 * 3  # Maximum possible image size.

        self.logger.info(
            "Using nvImageCodec decoder version: %s" % nvimgcodec.__version__
        )

        # docs_tag: end_init_imagebatchdecoder_nvimagecodec

    def __call__(self):
        if self.total_decoded == len(self.file_names):
            return None

        # docs_tag: begin_call_imagebatchdecoder_nvimagecodec
        self.cvcuda_perf.push_range("decoder.nvimagecodec")

        file_name_batch = self.file_name_batches[self.batch_idx]

        data_batch = [open(path, "rb").read() for path in file_name_batch]

        # docs_tag: begin_decode_imagebatchdecoder_nvimagecodec

        tensor_list = []
        image_list = self.decoder.decode(data_batch, cuda_stream=self.cuda_stream)

        # Convert the decoded images to nvcv tensors in a list.
        for i in range(len(image_list)):
            tensor_list.append(cvcuda.as_tensor(image_list[i], "HWC"))

        # Stack the list of tensors to a single NHWC tensor.
        cvcuda_decoded_tensor = cvcuda.stack(tensor_list)
        self.total_decoded += len(tensor_list)
        # docs_tag: end_decode_imagebatchdecoder_nvimagecodec

        # docs_tag: begin_return_imagebatchdecoder_nvimagecodec
        batch = Batch(
            batch_idx=self.batch_idx,
            data=cvcuda_decoded_tensor,
            fileinfo=file_name_batch,
        )
        self.batch_idx += 1

        # docs_tag: end_return_imagebatchdecoder_nvimagecodec

        self.cvcuda_perf.pop_range()
        # docs_tag: end_call_imagebatchdecoder_nvimagecodec
        return batch

    def start(self):
        pass

    def join(self):
        pass


# docs_tag: end_imagebatchdecoder_nvimagecodec

# docs_tag: begin_imagebatchencoder_nvimagecodec
class ImageBatchEncoder:
    def __init__(
        self,
        output_path,
        device_id,
        cvcuda_perf,
    ):
        # docs_tag: begin_init_imagebatchencoder_nvimagecodec
        self.logger = logging.getLogger(__name__)
        self.encoder = nvimgcodec.Encoder(device_id=device_id)
        self.input_layout = "NHWC"
        self.gpu_input = True
        self.output_path = output_path
        self.device_id = device_id
        self.cvcuda_perf = cvcuda_perf

        self.logger.info(
            "Using nvImageCodec encoder version: %s" % nvimgcodec.__version__
        )
        # docs_tag: end_init_init_imagebatchencoder_nvimagecodec

    # docs_tag: begin_call_imagebatchencoder_nvimagecodec
    def __call__(self, batch):
        self.cvcuda_perf.push_range("encoder.nvimagecodec")

        assert isinstance(batch.data, torch.Tensor)

        image_tensors_nchw = batch.data
        # Create an empty list to store filenames
        filenames = []
        chwtensor_list = []
        # Iterate through each image to prepare the filenames
        for img_idx in range(image_tensors_nchw.shape[0]):
            img_name = os.path.splitext(os.path.basename(batch.fileinfo[img_idx]))[0]
            results_path = os.path.join(self.output_path, f"out_{img_name}.jpg")
            self.logger.info(f"Preparing to save the image to: {results_path}")
            # Add the filename to the list
            filenames.append(results_path)
            # Add the image tensor CAI to a CAI list from an NCHW tensor
            # (this was a stacked tensor if N images)
            chwtensor_list.append(image_tensors_nchw[img_idx].cuda())

        # Pass the image tensors and filenames to the encoder.
        self.encoder.write(filenames, chwtensor_list)
        self.cvcuda_perf.pop_range()
        # docs_tag: end_call_imagebatchencoder_nvimagecodec

    def start(self):
        pass

    def join(self):
        pass


# docs_tag: end_imagebatchencoder_nvimagecodec
