[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Normalize U8 control migration

The active U8/C1 NHWC Tensor control changes from B128 to B512 at 1080p to
address the campaign's 500-us measurement floor. B512 is a provisional
calibration geometry, not a measured latency or a speedup claim. All four
shipping/candidate × C++/Python calibration means must exceed 500 us before
qualification. Only this new shape needs a real shipping BEFORE seed.

The retired case is
`normalize_scalar_advanced[InOutDataType=uint8][shape=128x1080x1920][layout=NHWC][inputKind=Tensor]`.
Its immutable baseline remains in Git at
`1ec14a46da81ad8f0754f47a1225d97dbe596bd0`; its original raw observations,
including the failed cohort, remain campaign evidence. Retirement is not a
waiver of its floor failure or permission to erase that history.

The original scalar entry is a Cartesian product. To retire exactly one case,
the float32 cases keep `normalize_scalar_advanced`, while the two unchanged U8
VarShape cases move to `normalize_scalar_u8_varshape_advanced` and the unchanged
U8 B32 4K Tensor case moves to `normalize_scalar_u8_tensor_4k_advanced`.
Only those three config-key prefixes change: every non-key axis, fixture
setting and measured baseline value is copied exactly. This identity migration
is applied consistently to the shipping checkpoint and candidate result; it
does not create new measurements or reset existing comparators.

`normalize_scalar_u8_nhwc_b512_1080p_advanced` declares only the new Tensor
shape. The full declared surface remains 69 cases. Native storage, PerSample
parameters, stddev mode, scalar constants and warmup count are unchanged.
The old geometry is no longer declared as an active Tensor case. The existing
U8 VarShape and float32 controls are not recalibrated or remeasured by this work.
