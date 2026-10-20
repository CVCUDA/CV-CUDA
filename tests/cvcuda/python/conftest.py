# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings", "error:ExternalStream is deprecated:DeprecationWarning"
    )
