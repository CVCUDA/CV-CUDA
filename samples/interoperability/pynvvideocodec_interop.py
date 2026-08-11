# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Example showcasing how to use CV-CUDA with PyNvVideoCodec.

PyNvVideoCodec videos can be used to create CV-CUDA tensors and vice verse.
"""

# docs_tag: begin_imports
from __future__ import annotations

from pathlib import Path
import sys

import cvcuda

try:
    import PyNvVideoCodec as nvc
except ModuleNotFoundError as error:
    if error.name != "PyNvVideoCodec" or sys.version_info < (3, 13):
        raise
    nvc = None

# docs_tag: end_imports


def main():
    """PyNvVideoCodec <-> CV-CUDA interoperability example.

    This sample demonstrates video decoding, CV-CUDA processing,
    and video encoding (if hardware video encoder (NVENC) is available)
    """
    if nvc is None:
        python_version = f"{sys.version_info[0]}.{sys.version_info[1]}"
        print(
            "Skipping PyNvVideoCodec interoperability sample: "
            f"PyNvVideoCodec is unavailable for Python {python_version}."
        )
        return

    # docs_tag: begin_init_pynvvideocodec
    # Setup paths
    video_path = (
        Path(__file__).parent.parent
        / "assets"
        / "videos"
        / "pexels-chiel-slotman-4423925-1920x1080-25fps.mp4"
    )
    cvcuda_root = Path(__file__).parent.parent.parent
    output_dir = cvcuda_root / ".cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pexels-chiel-slotman-640x480.h264"
    device_id = 0

    # Create decoder
    decoder = nvc.SimpleDecoder(
        str(video_path),
        gpu_id=device_id,
        output_color_type=nvc.OutputColorType.RGB,
        need_scanned_stream_metadata=False,
        max_width=1920,
        max_height=1080,
        use_device_memory=True,
    )

    # docs_tag: end_init_pynvvideocodec

    # Read video and store frames in CV-CUDA tensors
    # Resize and cvtcolor on the frames as we read the images
    # PyNvVideoCodec will re-use the same buffers, so we cannot
    # use zero-copy as_tensor and maintain the original data (without copying)
    # docs_tag: begin_read_and_process_video
    processed_frames: list[cvcuda.Tensor] = []
    frame_idx = 0
    num_frames = decoder.get_stream_metadata().num_frames
    while frame_idx < num_frames:
        batch_size = min(10, num_frames - frame_idx)
        frame_chunk = decoder.get_batch_frames(batch_size)
        for frame in frame_chunk:
            # get the CV-CUDA tensor from the PyNvVideoCodec DecodedFrame
            cvcuda_tensor = cvcuda.as_tensor(frame, "HWC")
            # process the frame with CV-CUDA
            # importantly since CV-CUDA did not copy from the PyNvVideoCodec buffer
            # we must use the buffer before we read the next batch of frames
            # here we process one frame at a time for simplicity
            resized = cvcuda.resize(cvcuda_tensor, (480, 640, 3), cvcuda.Interp.LINEAR)
            nv12_frame = cvcuda.cvtcolor(resized, cvcuda.ColorConversion.RGB2YUV_NV12)
            # store the new CV-CUDA tensor (resize and cvtcolor make new buffers)
            processed_frames.append(nv12_frame)
            frame_idx += 1
    # docs_tag: end_read_and_process_video

    # docs_tag: begin_encode_frames
    # Encode frames (only if encoder hardware is available)
    try:
        encoder = nvc.CreateEncoder(
            width=640,
            height=480,
            fmt="NV12",
            codec="h264",
            usecpuinputbuffer=False,
        )
    except Exception as e:
        print(f"Encoder not available - will skip encoding step: {e}")
        return
    with output_path.open("wb") as f:
        for frame in processed_frames:
            bitstream = encoder.Encode(frame)
            f.write(bytearray(bitstream))
        bitstream = encoder.EndEncode()
        f.write(bytearray(bitstream))
    print(f"Successfully encoded video to {output_path}")
    # docs_tag: end_encode_frames


if __name__ == "__main__":
    main()
