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

import argparse

import torch
import torchvision

import cvcuda


def parse_arguments():
    """Parse this program script arguments."""

    parser = argparse.ArgumentParser(prog="label", description="Labels an input image.")

    parser.add_argument("input", type=str, help="Input image png file path.")
    parser.add_argument(
        "output",
        nargs="?",
        default="out.png",
        type=str,
        help="Output image png file path.  Defaults to out.png.",
    )
    parser.add_argument(
        "--max_labels",
        default=1000,
        type=int,
        help="Maximum number of labels.  Defaults to 1000.",
    )
    parser.add_argument(
        "--min_threshold",
        default=None,
        type=int,
        help="Minimum threshold to binarize input.  Defaults to no minimum threshold.",
    )
    parser.add_argument(
        "--max_threshold",
        default=None,
        type=int,
        help="Maximum threshold to binarize input.  Defaults to no maximum threshold.",
    )
    parser.add_argument(
        "--min_size",
        default=None,
        type=int,
        help="Minimum size to prevent a region to be removed.  Defaults to no minimum size (no removals).",
    )
    parser.add_argument(
        "--mask",
        action=argparse.BooleanOptionalAction,
        help="Apply mask to protect center islands (small regions).  Defaults to no mask.",
    )
    parser.add_argument(
        "--background_label",
        default=0,
        type=int,
        help="Background label. Defaults to zero.",
    )

    return parser.parse_args()


def color_labels(
    h_labels_hw,
    bgl,
    bgc=torch.as_tensor([0, 0, 0], dtype=torch.uint8),
    fgc=torch.as_tensor([255, 255, 255], dtype=torch.uint8),
    cmap=None,
):
    """Convert labels to colors

    Args:
        h_labels_hw (Tensor): Tensor with labels
        bgl (int): Background label
        bgc (Tensor): Background color, this color is used for the background label
        fgc (Tensor): Foreground color, this color is used when cmap is None
        cmap (function): Colormap, e.g. matplotlib.colormaps["jet"]

    Returns:
        Tensor: Tensor with colors
    """
    # Create an empty Tensor with same height and width as the labels Tensor and Channel = 3 for RGB
    h_out_hwc = torch.empty(
        (h_labels_hw.shape[0], h_labels_hw.shape[1], 3), dtype=torch.uint8
    )

    # Set all values to be the background color
    h_out_hwc[:, :] = bgc

    # Get the unique set of labels except background label from the labels Tensor
    h_uniq = torch.unique(h_labels_hw)
    h_uniq = h_uniq[h_uniq != bgl]

    # Set the label RGB color to be the foreground color
    label_rgb = fgc

    for i, label in enumerate(h_uniq):
        if cmap is not None:
            # If a color map was provided, use it to generate the label color
            label_rgb = [int(c * 255) for c in cmap(i / h_uniq.shape[0])[:3]]
            label_rgb = torch.as_tensor(label_rgb, dtype=torch.uint8)

        h_out_hwc[h_labels_hw == label] = label_rgb

    return h_out_hwc


if __name__ == "__main__":

    args = parse_arguments()

    print(
        f"I Reading input image: {args.input}\nI Writing output image: {args.output}\n"
        f"I Minimum threshold: {args.min_threshold}\nI Maximum threshold: {args.max_threshold}\n"
        f"I Minimum size: {args.min_size}\nI Apply mask: {args.mask}\n"
        f"I Background label: {args.background_label}"
    )

    # Use torchvision to read an input image, convert it to gray and store it as a CHW Tensor
    h_in_chw = torchvision.io.read_image(args.input, torchvision.io.ImageReadMode.GRAY)

    # Convert the image read from Pytorch Tensor to CVCUDA Tensor with zero copy
    d_in_chw = cvcuda.as_tensor(h_in_chw.cuda(), layout="CHW")

    # Reshape CVCUDA Tensor from CHW to HW (Channel is 1) with zero copy
    d_in_hw = d_in_chw.reshape(d_in_chw.shape[1:], "HW")

    # Tensors are initialized first in host (h_) and then copied to device (d_), using Pytorch's .as_tensor()
    # and .cuda() methods, and then converted to CVCUDA with zero copy, using CVCUDA's .as_tensor() method
    h_bgl = torch.as_tensor([args.background_label], dtype=h_in_chw.dtype)
    d_bgl = cvcuda.as_tensor(h_bgl.cuda(), layout="N")

    # Tensors for min/max thresholds min size and mask are optional
    d_min_thrs = None
    d_max_thrs = None
    d_min_size = None
    d_mask_hw = None

    if args.min_threshold:
        h_min_thrs = torch.as_tensor([args.min_threshold], dtype=h_in_chw.dtype)
        d_min_thrs = cvcuda.as_tensor(h_min_thrs.cuda(), layout="N")

    if args.max_threshold:
        h_max_thrs = torch.as_tensor([args.max_threshold], dtype=h_in_chw.dtype)
        d_max_thrs = cvcuda.as_tensor(h_max_thrs.cuda(), layout="N")

    if args.min_size:
        h_min_size = torch.as_tensor([args.min_size], dtype=torch.int32)
        d_min_size = cvcuda.as_tensor(h_min_size.cuda(), layout="N")

    if args.mask:
        # Below are slices in between 10% and 90% (a center box) to be considered inside the mask
        s_h_in_mask = slice(int(0.1 * h_in_chw.shape[1]), int(0.9 * h_in_chw.shape[1]))
        s_w_in_mask = slice(int(0.1 * h_in_chw.shape[2]), int(0.9 * h_in_chw.shape[2]))

        # The mask in host is first initialized with zeros
        h_mask_hw = torch.zeros(h_in_chw.shape[1:], dtype=h_in_chw.dtype)

        # Then the center of the mask defined by the slices is set to 1
        h_mask_hw[s_h_in_mask, s_w_in_mask] = 1

        # The Pytorch Tensor mask is copied to CUDA and converted to CVCUDA Tensor
        d_mask_hw = cvcuda.as_tensor(h_mask_hw.cuda(), layout="HW")

    # Call CVCUDA label operator using the arguments set above
    d_out, d_count, d_stats = cvcuda.label(
        src=d_in_hw,
        connectivity=cvcuda.CONNECTIVITY_4_2D,
        assign_labels=cvcuda.LABEL.SEQUENTIAL,
        mask_type=cvcuda.REMOVE_ISLANDS_OUTSIDE_MASK_ONLY,
        count=True,
        stats=True,
        max_labels=args.max_labels,
        bg_label=d_bgl,
        min_thresh=d_min_thrs,
        max_thresh=d_max_thrs,
        min_size=d_min_size,
        mask=d_mask_hw,
    )

    # Convert CVCUDA output Tensors to Pytorch with zero copy, using CVCUDA's .cuda() method, then copy the
    # Pytorch Tensor to the CPU, using Pytorch's .cpu() method
    h_out = torch.as_tensor(d_out.cuda()).cpu()
    h_count = torch.as_tensor(d_count.cuda()).cpu()
    h_stats = torch.as_tensor(d_stats.cuda()).cpu()

    print(f"I Number of labels found: {h_count[0]}")

    # The stats Tensor (with statistics) has a region mark at index 6 that is set to 1 for removed regions
    # and set to 2 for regions in the mask that cannot be removed
    num_removed = sum([1 if h_stats[0, si, 6] == 1 else 0 for si in range(h_count[0])])
    num_in_mask = sum([1 if h_stats[0, si, 6] == 2 else 0 for si in range(h_count[0])])

    print(f"I Number of labeled regions removed: {num_removed}")
    print(f"I Number of labeled regions in the mask: {num_in_mask}")
    print(f"I Number of labeled regions kept: {h_count[0] - num_removed}")

    # Color the labels using default behavior: white foreground and black background
    h_out_rgb_hwc = color_labels(h_out, h_bgl[0])

    # Use torchvision to write the output image from a CHW Tensor
    torchvision.io.write_png(h_out_rgb_hwc.permute(2, 0, 1), args.output)
