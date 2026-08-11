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
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import pytest

import cvcuda
import cvcuda_types as cv_types


def _create_tensor(shape: tuple, layout: str, dtype: cvcuda.Type) -> cvcuda.Tensor:
    return cvcuda.Tensor(shape, cv_types.as_cvcuda_dtype(dtype), layout)


def _create_tensor_batch(
    shape: tuple, layout: str, dtype: cvcuda.Type, count: int
) -> cvcuda.TensorBatch:
    tensor_batch = cvcuda.TensorBatch(count)
    for _ in range(count):
        tensor_batch.pushback(_create_tensor(shape, layout, dtype))
    return tensor_batch


def _create_image(size: tuple, fmt: cvcuda.Format) -> cvcuda.Image:
    return cvcuda.Image(size, fmt)


def _create_image_batch(
    size: tuple, fmt: cvcuda.Format, count: int
) -> cvcuda.ImageBatchVarShape:
    img_batch = cvcuda.ImageBatchVarShape(count)
    for _ in range(count):
        img_batch.pushback(_create_image(size, fmt))
    return img_batch


def _dtype_to_format(
    dtype: cvcuda.Type | cvcuda.Format,
    channels: int,
) -> cvcuda.Format:
    if isinstance(dtype, cvcuda.Format):
        return dtype
    return cv_types.dtype_channels_to_format(dtype, channels)


def _create_image_from_dtype(
    size: tuple,
    dtype: cvcuda.Type | cvcuda.Format,
    channels: int,
) -> cvcuda.Image:
    fmt = _dtype_to_format(dtype, channels)
    return _create_image(size, fmt)


def _create_image_batch_from_dtype(
    size: tuple, dtype: cvcuda.Type | cvcuda.Format, channels: int, count: int
) -> cvcuda.ImageBatchVarShape:
    fmt = _dtype_to_format(dtype, channels)
    return _create_image_batch(size, fmt, count)


def _resolve_wrapper(
    wrapper: str,
    size: tuple[int, int] = (24, 24),
    channels: int = 3,
    layout: str | None = None,
) -> Callable[
    ...,
    cvcuda.Tensor | cvcuda.Image | cvcuda.ImageBatchVarShape | cvcuda.TensorBatch,
]:
    layout = "NHWC" if layout is None else layout
    shape = cv_types.resolve_shape(layout, channels, size)

    if wrapper == "tensor":
        create_func = partial(_create_tensor, shape=shape, layout=layout)
    elif wrapper == "tensor_batch":
        create_func = partial(_create_tensor_batch, shape=shape, layout=layout, count=2)
    elif wrapper == "image":
        create_func = partial(_create_image_from_dtype, size=size, channels=channels)
    elif wrapper == "image_batch":
        create_func = partial(
            _create_image_batch_from_dtype, size=size, channels=channels, count=2
        )
    else:
        raise ValueError(f"Invalid wrapper: {wrapper}")

    return create_func


def _runner(
    op: Callable[
        ...,
        cvcuda.Tensor | cvcuda.Image | cvcuda.ImageBatchVarShape | cvcuda.TensorBatch,
    ],
    data: cvcuda.Tensor | cvcuda.Image | cvcuda.ImageBatchVarShape | cvcuda.TensorBatch,
    *,
    negative: bool = False,
    exceptions: list[type[Exception]] | None = None,
) -> None:
    if not negative:
        op(data)
        cvcuda.Stream.current.sync()
    else:
        # Default: NVCV_ERROR is converted to RuntimeError
        exc_tuple = tuple(exceptions) if exceptions else (RuntimeError,)
        with pytest.raises(exc_tuple):
            op(data)
            cvcuda.Stream.current.sync()


def assert_dtypes(
    op: Callable[..., cvcuda.Tensor],
    dtypes: list[cvcuda.Type] | cvcuda.Type,
    wrapper: str = "tensor",
    size: tuple[int, int] = (24, 24),
    channels: int = 3,
    layout: str | None = None,
    *,
    negative: bool = False,
    exceptions: list[type[Exception]] | None = None,
) -> None:
    if wrapper not in {"tensor", "image", "image_batch", "tensor_batch"}:
        raise ValueError(f"Invalid wrapper: {wrapper}")
    if not isinstance(dtypes, list):
        dtypes = [dtypes]
    for dtype in dtypes:
        create_func = _resolve_wrapper(wrapper, size, channels, layout)
        data = create_func(dtype=dtype)
        _runner(op, data, negative=negative, exceptions=exceptions)


def assert_layouts(
    op: Callable[..., cvcuda.Tensor],
    layouts: list[str] | str,
    dtype: cvcuda.Type,
    wrapper: str = "tensor",
    size: tuple[int, int] = (24, 24),
    channels: int = 3,
    *,
    negative: bool = False,
    exceptions: list[type[Exception]] | None = None,
) -> None:
    if wrapper not in {"tensor", "image", "image_batch", "tensor_batch"}:
        raise ValueError(f"Invalid wrapper: {wrapper}")
    if not isinstance(layouts, list):
        layouts = [layouts]
    for layout in layouts:
        create_func = _resolve_wrapper(wrapper, size, channels, layout)
        data = create_func(dtype=dtype)
        _runner(op, data, negative=negative, exceptions=exceptions)


def assert_formats(
    op: Callable[
        ...,
        cvcuda.Tensor | cvcuda.Image | cvcuda.ImageBatchVarShape | cvcuda.TensorBatch,
    ],
    formats: list[cvcuda.Format],
    wrapper: str = "tensor",
    size: tuple[int, int] = (24, 24),
    *,
    negative: bool = False,
    exceptions: list[type[Exception]] | None = None,
) -> None:
    if wrapper not in {"tensor", "image", "image_batch", "tensor_batch"}:
        raise ValueError(f"Invalid wrapper: {wrapper}")
    create_func = _resolve_wrapper(wrapper, size)
    for fmt in formats:
        data = create_func(dtype=fmt)
        _runner(op, data, negative=negative, exceptions=exceptions)


def _compute_extra_param_combos(
    extra_params: dict[str, set] | None,
) -> tuple[list[str], list[tuple]]:
    """
    Compute cross-product combinations for extra parameters.

    Args:
        extra_params: Dict mapping param name to set of values.

    Returns:
        Tuple of (keys list, combos list). If extra_params is None/empty,
        returns ([], [()]) for a single empty combo to simplify iteration.
    """
    if extra_params:
        keys = list(extra_params.keys())
        values = [list(extra_params[k]) for k in keys]
        return keys, list(itertools.product(*values))
    return [], [()]


def _make_extra_params_dict(keys: list[str], combo: tuple) -> dict:
    """Convert a combo tuple to a dict of extra params."""
    return dict(zip(keys, combo)) if keys else {}


def _get_first_extra_params(extra_params: dict[str, set] | None) -> dict:
    """Get first value from each extra_params set for negative tests."""
    return {k: next(iter(v)) for k, v in extra_params.items()} if extra_params else {}


# Type alias for runner info: (wrapper, op, param_factory)
# - wrapper: "tensor", "image_batch", "tensor_batch", "image"
# - op: The operator function to call
# - param_factory: Optional function (dtype, layout, channels) -> dict of kwargs
RunnerInfo = tuple[str, Callable, Optional[Callable]]


@dataclass
class _RunnerContext:
    """
    Encapsulates runner information and extra params for test generation.

    This dataclass provides a unified interface for both format-based and
    dtype/layout/channel-based test generation, eliminating code duplication.
    """

    runner_info: list[RunnerInfo]
    extra_params: dict[str, set] | None = None
    exclude_extra_params: list[tuple[str, str]] | None = None
    extra_param_keys: list[str] = field(default_factory=list)
    extra_param_combos: list[tuple] = field(default_factory=lambda: [()])

    def __post_init__(self):
        """Compute extra param combinations after initialization."""
        self.extra_param_keys, self.extra_param_combos = _compute_extra_param_combos(
            self.extra_params
        )
        # Build exclusion lookup: {wrapper: set of excluded param names}
        self._exclusions: dict[str, set[str]] = {}
        if self.exclude_extra_params:
            for param_name, wrapper in self.exclude_extra_params:
                if wrapper not in self._exclusions:
                    self._exclusions[wrapper] = set()
                self._exclusions[wrapper].add(param_name)

    @property
    def labels(self) -> list[str]:
        """Generate runner labels for test IDs."""
        return [wrapper for wrapper, _, _ in self.runner_info]

    @property
    def runner_count(self) -> int:
        """Number of runners."""
        return len(self.runner_info)

    def get_wrapper(self, idx: int) -> str:
        """Get wrapper type for runner at index."""
        return self.runner_info[idx][0]

    def get_op(
        self,
        idx: int,
        dtype: cvcuda.Type | None,
        layout: str | None,
        channels: int | None,
        extra_dict: dict | None = None,
    ) -> Callable:
        """
        Get the operator function for a runner, with param_factory applied if present.

        Args:
            idx: Runner index.
            dtype: Data type (None for format mode).
            layout: Layout string (None for format mode).
            channels: Channel count (None for format mode).
            extra_dict: Additional parameters to pass to param_factory.

        Returns:
            Callable operator function, potentially wrapped with partial.
        """
        _, op, param_factory = self.runner_info[idx]
        if param_factory is not None:
            params = param_factory(dtype, layout, channels, **(extra_dict or {}))
            return partial(op, **params)
        return op

    def is_param_excluded(self, param_name: str, wrapper: str) -> bool:
        """Check if a param is excluded for a given wrapper."""
        return param_name in self._exclusions.get(wrapper, set())

    def filter_extra_dict(self, extra_dict: dict, wrapper: str) -> dict:
        """Filter out excluded params for the given wrapper."""
        excluded = self._exclusions.get(wrapper, set())
        if not excluded:
            return extra_dict
        return {k: v for k, v in extra_dict.items() if k not in excluded}

    def make_extra_dict(self, combo: tuple, wrapper: str | None = None) -> dict:
        """Convert a combo tuple to extra params dict, optionally filtering for wrapper."""
        extra_dict = _make_extra_params_dict(self.extra_param_keys, combo)
        if wrapper is not None:
            return self.filter_extra_dict(extra_dict, wrapper)
        return extra_dict

    def get_first_extra_params(self, wrapper: str | None = None) -> dict:
        """Get first value from each extra param for negative tests."""
        extra_dict = _get_first_extra_params(self.extra_params)
        if wrapper is not None:
            return self.filter_extra_dict(extra_dict, wrapper)
        return extra_dict


def _matches_exclude_pattern(
    dtype: cvcuda.Type,
    layout: str,
    channels: int,
    exclude_pattern: tuple[cvcuda.Type | None, str | None, int | None],
) -> bool:
    """
    Check if a (dtype, layout, channels) combo matches an exclude pattern.
    None in the pattern acts as a wildcard (matches anything).
    """
    pattern_dtype, pattern_layout, pattern_channels = exclude_pattern
    if pattern_dtype is not None and dtype != pattern_dtype:
        return False
    if pattern_layout is not None and layout != pattern_layout:
        return False
    if pattern_channels is not None and channels != pattern_channels:
        return False
    return True


def _is_excluded(
    dtype: cvcuda.Type,
    layout: str,
    channels: int,
    exclude_dlc: list[tuple[cvcuda.Type | None, str | None, int | None]] | None,
) -> bool:
    """Check if a (dtype, layout, channels) combo should be excluded."""
    if exclude_dlc is None:
        return False
    return any(
        _matches_exclude_pattern(dtype, layout, channels, pattern)
        for pattern in exclude_dlc
    )


def _has_format_mapping(dtype: cvcuda.Type, channels: int) -> bool:
    """Check if a dtype/channel combo has a valid format mapping for image creation."""
    return (dtype, channels) in cv_types.DTYPE_CHANNELS_TO_FORMAT


def _make_extra_param_negative_tests(
    name: str,
    extra_params_negative: dict[str, set] | None,
    ctx: _RunnerContext,
    get_op_func: Callable[[int, dict], Callable],
    get_assert_kwargs: Callable[[int], dict],
    assert_func: Callable,
    negative_exceptions: list[type[Exception]] | None = None,
) -> dict[str, Callable]:
    """
    Generate negative tests for extra params - shared by both format and dtype modes.

    Tests are parametrized over all runners and all unsupported values.

    Args:
        name: Operator name for test function naming.
        extra_params_negative: Dict mapping param name to unsupported values.
        ctx: Runner context with extra params info.
        get_op_func: Function (runner_idx, extra_dict) -> operator callable.
        get_assert_kwargs: Function (runner_idx) -> kwargs dict for assert_func.
        assert_func: Assertion function to use (assert_formats or assert_layouts).

    Returns:
        Dict of test function names to pytest-parametrized test functions.
    """
    if not extra_params_negative:
        return {}

    result = {}
    for param_name, unsupported_values in extra_params_negative.items():
        # Cross-product of runners and bad values, excluding param/wrapper combos
        test_params = [
            (runner_idx, bad_value)
            for runner_idx in range(ctx.runner_count)
            for bad_value in unsupported_values
            if not ctx.is_param_excluded(param_name, ctx.get_wrapper(runner_idx))
        ]

        # Skip if no valid test params after filtering
        if not test_params:
            continue

        def _test_id(params, _param_name=param_name):
            runner_idx, bad_value = params
            return f"{ctx.labels[runner_idx]}-{_param_name}={bad_value}"

        @pytest.mark.parametrize(
            "runner_idx,bad_value",
            test_params,
            ids=[_test_id(p) for p in test_params],
        )
        def test_extra_param_negative(
            runner_idx,
            bad_value,
            _param_name=param_name,
            _get_op=get_op_func,
            _get_assert_kwargs=get_assert_kwargs,
            _assert_func=assert_func,
            _ctx=ctx,
            _negative_exceptions=negative_exceptions,
        ):
            wrapper = _ctx.get_wrapper(runner_idx)
            extra_dict = _ctx.get_first_extra_params(wrapper)
            extra_dict[_param_name] = bad_value
            op = _get_op(runner_idx, extra_dict)
            _assert_func(
                op,
                **_get_assert_kwargs(runner_idx),
                negative=True,
                exceptions=_negative_exceptions,
            )

        result[f"test_op_{name}_{param_name}_negative"] = test_extra_param_negative

    return result


def _format_test_id(
    runner_labels: list[str], runner_idx: int, fmt: cvcuda.Format, extra_combo: tuple
) -> str:
    """Generate test ID for format-based tests."""
    base_id = f"{runner_labels[runner_idx]}-{fmt.name}"
    if extra_combo:
        extra_str = "-".join(str(v) for v in extra_combo)
        return f"{base_id}-{extra_str}"
    return base_id


def _dtype_test_id(
    runner_labels: list[str],
    runner_idx: int,
    dtype: cvcuda.Type,
    layout: str,
    channels: int,
    extra_combo: tuple,
) -> str:
    """Generate test ID for dtype/layout/channel-based tests."""
    base_id = f"{runner_labels[runner_idx]}-{dtype.name}-{layout}-{channels}ch"
    if extra_combo:
        extra_str = "-".join(str(v) for v in extra_combo)
        return f"{base_id}-{extra_str}"
    return base_id


def _build_negative_test_params(
    ctx: _RunnerContext,
    negative_values: list,
    skip_image_wrappers: bool = False,
    format_check_func: Callable[[object], bool] | None = None,
) -> list[tuple[int, object]]:
    """
    Build negative test parameters for a given set of unsupported values.

    Args:
        ctx: Runner context containing runner information.
        negative_values: List of unsupported values to test.
        skip_image_wrappers: If True, skip image/image_batch wrappers entirely.
        format_check_func: Optional function to check if a value has a valid
            format mapping. If provided, image wrappers are only included
            when this returns True for the value.

    Returns:
        List of (runner_idx, value) tuples for parametrization.
    """
    params = []
    for runner_idx in range(ctx.runner_count):
        wrapper = ctx.get_wrapper(runner_idx)
        is_image_wrapper = wrapper in ("image_batch", "image")

        if skip_image_wrappers and is_image_wrapper:
            continue

        for value in negative_values:
            if is_image_wrapper and format_check_func is not None:
                if not format_check_func(value):
                    continue
            params.append((runner_idx, value))

    return params


def _make_negative_test(
    params: list[tuple[int, object]],
    id_func: Callable[[int, object], str],
    test_body: Callable[[int, object], None],
) -> Callable | None:
    """
    Create a parametrized negative test function.

    Args:
        params: List of (runner_idx, value) tuples.
        id_func: Function (runner_idx, value) -> test ID string.
        test_body: Function (runner_idx, value) -> None that runs the test.

    Returns:
        Parametrized test function, or None if params is empty.
    """
    if not params:
        return None

    @pytest.mark.parametrize(
        "runner_idx,value",
        params,
        ids=[id_func(idx, val) for idx, val in params],
    )
    def test_negative(runner_idx, value, _body=test_body):
        _body(runner_idx, value)

    return test_negative


def _collect_test_results(
    name: str,
    primary_test: Callable,
    primary_suffix: str,
    negative_tests: list[tuple[str, Callable | None]],
    extra_neg_tests: dict[str, Callable],
) -> dict[str, Callable]:
    """
    Collect all test functions into a result dictionary.

    Args:
        name: Operator name.
        primary_test: The main positive test function.
        primary_suffix: Suffix for primary test (e.g., "input" or "format_input").
        negative_tests: List of (suffix, test_func_or_none) tuples.
        extra_neg_tests: Dict of extra param negative tests.

    Returns:
        Dict mapping test names to test functions.
    """
    result = {f"test_op_{name}_{primary_suffix}": primary_test}
    for suffix, test_func in negative_tests:
        if test_func is not None:
            result[f"test_op_{name}_{suffix}"] = test_func
    result.update(extra_neg_tests)
    return result


def make_op_tests(
    name: str,
    runner_info: list[RunnerInfo],
    # Format Mode - tests image formats
    supported_formats: set[cvcuda.Format] | None = None,
    # DType Mode - tests dtype/layout/channel combinations
    keystone_dlc: tuple[cvcuda.Type, str, int] | None = None,
    supported_dtypes: set[cvcuda.Type] | None = None,
    supported_layouts: set[str] | None = None,
    supported_channels: set[int] | None = None,
    # Exclusions + Extra params to finetune test matrix
    exclude_dlc: list[tuple[cvcuda.Type | None, str | None, int | None]] | None = None,
    extra_params: dict[str, set] | None = None,
    extra_params_negative: dict[str, set] | None = None,
    exclude_extra_params: list[tuple[str, str]] | None = None,
    negative_exceptions: list[type[Exception]] | None = None,
) -> dict[str, Callable]:
    """
    Generate pytest test functions for CV-CUDA operator input validation.

    Overview
    --------
    Creates parameterized tests that verify operators handle supported and
    unsupported inputs correctly. Tests are organized in two modes:

    - **Format Mode**: Test image formats directly. Requires image/image_batch
      wrappers. Use when operator behavior depends on cvcuda.Format values.
    - **DType Mode**: Test dtype/layout/channel combinations. Works with any
      wrapper type. Use for most operators that accept various data types.

    These generated tests validate the Python-declared input contract against the
    runtime guards in the operator implementation. They do not machine-check the
    public header documentation, and they are not a substitute for output-value
    correctness tests.

    Image-backed coverage is intentionally narrower than tensor-backed coverage:
    layout negatives are only meaningful for tensor wrappers, and image dtype
    negatives are limited to dtype/channel combinations that can be expressed as
    valid image formats.

    Both modes generate:
      - Positive tests: Verify operator works with supported inputs
      - Negative tests: Verify operator raises RuntimeError, or configured exceptions,
        for unsupported inputs

    Summary:
    - Supported dtypes/formats (positive tests)
    - Unsupported dtypes/formats (negative tests - expect RuntimeError or configured exceptions)
    - Layout validation (tensor wrappers only)
    - Channel count validation
    - Extra parameter validation

    Call Flow
    ---------
    ::

        make_op_tests(name, runner_info, ...)
            |
            +-- [Format Mode] _make_format_tests()
            |       |
            |       +-- Positive test: for each (runner, format, extra_combo):
            |       |       1. _resolve_wrapper(wrapper) -> create_func
            |       |       2. create_func(dtype=format) -> data container
            |       |       3. param_factory(None, None, None, **extra) -> op_kwargs
            |       |       4. op(data, **op_kwargs) -> verify no exception
            |       |
            |       +-- Negative test: for each unsupported format:
            |               pytest.raises(RuntimeError or configured exceptions)
            |               during op(data)
            |
            +-- [DType Mode] _make_dtype_layout_channel_tests()
                    |
                    +-- Positive test: for each (runner, dtype, layout, channels, extra):
                    |       1. cv_types.resolve_shape(layout, channels, size) -> shape
                    |       2. _resolve_wrapper(wrapper, layout=layout) -> create_func
                    |       3. create_func(dtype=dtype) -> data container
                    |       4. param_factory(dtype, layout, channels, **extra) -> op_kwargs
                    |       5. op(data, **op_kwargs) -> verify no exception
                    |
                    +-- Negative tests for dtype/layout/channels:
                            Uses keystone_dlc as "known good" baseline,
                            varies one dimension at a time to test unsupported values

    Default Values
    --------------
    These defaults are defined in this module and cvcuda_types:

    - **Image size**: (24, 24) width x height - from ``_resolve_wrapper()``
    - **Layout**: "NHWC" when not specified - from ``_resolve_wrapper()``
    - **Batch count**: 2 for tensor_batch and image_batch - from ``_resolve_wrapper()``
    - **Shape resolution**: Uses ``cvcuda_types.LAYOUT_TO_SHAPE`` mapping
    - **Valid layouts**: ``cvcuda_types.IMAGE_LAYOUTS`` = {"NHWC", "HWC", "NCHW", ...}
    - **Valid channels**: ``cvcuda_types.CHANNELS`` = {1, 2, 3, 4, 5, 6}
    - **Scalar dtypes**: ``cvcuda_types.SCALAR_TYPES_SET`` = {U8, U16, S8, S16, ...}
    - **Scalar formats**: ``cvcuda_types.SCALAR_FORMATS_SET`` = {U8, U16, F32, ...}

    Container Types (Wrappers)
    --------------------------
    - ``"tensor"``: cvcuda.Tensor - single tensor
    - ``"tensor_batch"``: cvcuda.TensorBatch - batch of 2 tensors
    - ``"image"``: cvcuda.Image - single image
    - ``"image_batch"``: cvcuda.ImageBatchVarShape - batch of 2 images

    Generated Test Names
    --------------------
    Tests are named following the pattern ``test_op_{name}_{test_type}``:

    - ``test_op_{name}_input`` - positive dtype/layout/channels (DType Mode)
    - ``test_op_{name}_format_input`` - positive formats (Format Mode)
    - ``test_op_{name}_dtype_negative`` - unsupported dtypes
    - ``test_op_{name}_layout_negative`` - unsupported layouts
    - ``test_op_{name}_channels_negative`` - unsupported channels
    - ``test_op_{name}_format_negative`` - unsupported formats
    - ``test_op_{name}_{param}_negative`` - unsupported extra param values

    Test IDs follow patterns like:
      - ``tensor-U8-NHWC-3ch`` (DType Mode)
      - ``image_batch-RGB8`` (Format Mode)
      - ``tensor-U8-NHWC-3ch-LINEAR`` (with extra_params)

    Parameters
    ----------
    name : str
        Operator name used in generated test function names.

    runner_info : list[RunnerInfo]
        List of (wrapper, op, param_factory) tuples defining test runners:

        - **wrapper**: One of "tensor", "tensor_batch", "image", "image_batch"
        - **op**: The cvcuda operator function (e.g., ``cvcuda.resize``)
        - **param_factory**: Optional callable with signature::

              def param_factory(
                  dtype: cvcuda.Type | None,
                  layout: str | None,
                  channels: int | None,
                  **extra_params
              ) -> dict:
                  '''Return kwargs to pass to op alongside source data.'''

          For Format Mode, dtype/layout/channels are all None.

    supported_formats : set[cvcuda.Format], optional
        Set of cvcuda.Format values the operator accepts (Format Mode).
        Requires at least one "image" or "image_batch" runner.
        Negative tests use ``SCALAR_FORMATS_SET - supported_formats``.

    keystone_dlc : tuple[cvcuda.Type, str, int], optional
        **REQUIRED for DType Mode.** The "keystone" is a known-good baseline
        configuration (dtype, layout, channels) used when generating negative
        tests. When testing an unsupported dtype, the keystone's layout and
        channels are used to ensure only one dimension varies at a time.

        Example: ``(cvcuda.Type.U8, "NHWC", 3)`` means U8/NHWC/3-channel is
        the baseline for negative tests.

    supported_dtypes : set[cvcuda.Type], optional
        Set of cvcuda.Type values the operator accepts (DType Mode).
        Negative tests use ``SCALAR_TYPES_SET - supported_dtypes``.

    supported_layouts : set[str], optional
        Set of layout strings the operator accepts (e.g., {"NHWC", "HWC"}).
        Must be subset of ``IMAGE_LAYOUTS`` from cvcuda_types.
        Negative tests use ``IMAGE_LAYOUTS - supported_layouts``.

    supported_channels : set[int], optional
        Set of channel counts the operator accepts (e.g., {1, 3, 4}).
        Negative tests use ``CHANNELS - supported_channels``.

    exclude_dlc : list[tuple], optional
        List of (dtype, layout, channels) patterns to exclude from positive
        tests. None acts as wildcard in any position.

        Example: ``[(cvcuda.Type.F32, None, 2)]`` excludes all F32 2-channel
        combinations regardless of layout.

    extra_params : dict[str, set], optional
        Additional parameters to cross-product with test cases. Each key is
        a parameter name, and the value is a set of values to test.

        Example: ``{"interp": {cvcuda.Interp.LINEAR, cvcuda.Interp.CUBIC}}``
        generates tests for both interpolation methods.

    extra_params_negative : dict[str, set], optional
        Unsupported values for extra parameters. Generates negative tests
        using keystone as baseline, varying only the extra param.

    exclude_extra_params : list[tuple[str, str]], optional
        List of (param_name, wrapper) tuples specifying which extra params
        to exclude from specific wrappers.

        Example: ``[("mask_channels", "image_batch")]`` excludes mask_channels
        from being passed to image_batch runners (useful when a parameter
        only applies to certain wrapper types).

    negative_exceptions : list[type[Exception]], optional
        Exception types accepted by negative tests. When not provided, negative
        tests expect RuntimeError. When provided, any listed exception type is
        accepted as a valid negative outcome.

    Returns
    -------
    dict[str, Callable]
        Dict mapping test function names to pytest-parametrized functions.
        Use ``globals().update(make_op_tests(...))`` to register tests.

    Examples
    --------
    Basic DType Mode usage::

        def _resize_params(dtype, layout, channels):
            return {
                "shape": cv_types.resolve_shape(layout, channels, (10, 20)),
                "interp": cvcuda.Interp.LINEAR,
            }

        globals().update(
            cv_tools.make_op_tests(
                name="resize",
                runner_info=[
                    ("tensor", cvcuda.resize, _resize_params),
                    ("image_batch", cvcuda.resize, _resize_varshape_params),
                ],
                keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
                supported_dtypes={cvcuda.Type.U8, cvcuda.Type.F32},
                supported_layouts={"NHWC", "HWC"},
                supported_channels={1, 3, 4},
            )
        )

    Format Mode usage::

        globals().update(
            cv_tools.make_op_tests(
                name="channelreorder",
                runner_info=[("image_batch", _channelreorder, None)],
                supported_formats={
                    cvcuda.Format.U8,
                    cvcuda.Format.RGB8,
                    cvcuda.Format.RGBA8,
                },
            )
        )

    With extra parameters and exclusions::

        globals().update(
            cv_tools.make_op_tests(
                name="composite",
                runner_info=[
                    ("tensor", _composite, _composite_params),
                    ("image_batch", _composite_varshape, _composite_params),
                ],
                keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
                supported_dtypes={cvcuda.Type.U8},
                supported_layouts={"NHWC", "HWC"},
                supported_channels={3},
                extra_params={"out_channels": {3, 4}},
                extra_params_negative={
                    "mask_channels": {2, 3, 4},
                    "out_channels": {0, 2, 5},
                },
                exclude_extra_params=[
                    ("mask_channels", "image_batch"),
                ],
                negative_exceptions=[RuntimeError, ValueError],
            )
        )
    """
    format_mode = supported_formats is not None
    dtype_mode = (
        supported_dtypes is not None
        or supported_layouts is not None
        or supported_channels is not None
    )

    if not format_mode and not dtype_mode:
        raise ValueError(
            "Must specify either supported_formats or "
            "one of (supported_dtypes, supported_layouts, supported_channels)"
        )

    if dtype_mode and keystone_dlc is None:
        raise ValueError(
            "keystone_dlc is required when using dtype mode "
            "(supported_dtypes, supported_layouts, or supported_channels)"
        )

    # Pre-filter runners by type
    format_runners = [r for r in runner_info if r[0] in ("image", "image_batch")]

    result = {}

    # format-based tests (requires image or image_batch wrapper)
    if format_mode:
        if not format_runners:
            raise ValueError(
                "supported_formats requires at least one runner with 'image' or 'image_batch' wrapper"
            )
        result.update(
            _make_format_tests(
                name,
                format_runners,
                supported_formats,
                extra_params,
                extra_params_negative,
                exclude_extra_params,
                negative_exceptions,
            )
        )

    # dtype/layout/channel-based tests (runs on any container wrappers)
    if dtype_mode:
        result.update(
            _make_dtype_layout_channel_tests(
                name,
                runner_info,
                supported_dtypes,
                supported_layouts,
                supported_channels,
                keystone_dlc,
                exclude_dlc,
                extra_params,
                extra_params_negative,
                exclude_extra_params,
                negative_exceptions,
            )
        )

    return result


def _make_format_tests(
    name: str,
    runner_info: list[RunnerInfo],
    supported_formats: set[cvcuda.Format],
    extra_params: dict[str, set] | None = None,
    extra_params_negative: dict[str, set] | None = None,
    exclude_extra_params: list[tuple[str, str]] | None = None,
    negative_exceptions: list[type[Exception]] | None = None,
) -> dict[str, Callable]:
    """Generate format-based test functions.

    Creates a positive test parametrized over every (runner, format, extra_combo)
    combination, a negative test that expects RuntimeError for each unsupported
    format (SCALAR_FORMATS_SET minus supported_formats), and per-extra-param
    negative tests when extra_params_negative is provided.

    Returns a dict of ``test_op_{name}_*`` names to pytest-parametrized callables.
    """
    ctx = _RunnerContext(runner_info, extra_params, exclude_extra_params)
    unsupported_formats = list(cv_types.SCALAR_FORMATS_SET - supported_formats)

    # --- Positive test ---
    format_params = list(
        itertools.product(
            range(ctx.runner_count), supported_formats, ctx.extra_param_combos
        )
    )

    def _pos_id(params):
        runner_idx, fmt, extra_combo = params
        return _format_test_id(ctx.labels, runner_idx, fmt, extra_combo)

    @pytest.mark.parametrize(
        "runner_idx,img_format,extra_combo",
        format_params,
        ids=[_pos_id(p) for p in format_params],
    )
    def test_format_input(runner_idx, img_format, extra_combo):
        wrapper = ctx.get_wrapper(runner_idx)
        extra_dict = ctx.make_extra_dict(extra_combo, wrapper)
        op = ctx.get_op(runner_idx, None, None, None, extra_dict)
        assert_formats(op, [img_format], wrapper=wrapper)

    # --- Negative format test ---
    format_neg_params = _build_negative_test_params(ctx, unsupported_formats)

    def _format_neg_body(runner_idx, img_format):
        wrapper = ctx.get_wrapper(runner_idx)
        extra_dict = ctx.get_first_extra_params(wrapper)
        op = ctx.get_op(runner_idx, None, None, None, extra_dict)
        assert_formats(
            op,
            [img_format],
            wrapper=wrapper,
            negative=True,
            exceptions=negative_exceptions,
        )

    test_format_negative = _make_negative_test(
        format_neg_params,
        id_func=lambda idx, fmt: f"{ctx.labels[idx]}-{fmt.name}",
        test_body=_format_neg_body,
    )

    # --- Negative extra_params tests ---
    extra_neg_tests: dict[str, Callable] = {}
    if supported_formats:
        keystone_format = [next(iter(supported_formats))]

        def _get_op_for_extra_neg(runner_idx, extra_dict):
            return ctx.get_op(runner_idx, None, None, None, extra_dict)

        def _get_assert_kwargs(runner_idx):
            return {"formats": keystone_format, "wrapper": ctx.get_wrapper(runner_idx)}

        extra_neg_tests = _make_extra_param_negative_tests(
            name,
            extra_params_negative,
            ctx,
            _get_op_for_extra_neg,
            _get_assert_kwargs,
            assert_formats,
            negative_exceptions,
        )

    # --- Build result ---
    return _collect_test_results(
        name,
        test_format_input,
        "format_input",
        [("format_negative", test_format_negative)],
        extra_neg_tests,
    )


def _make_dtype_layout_channel_tests(
    name: str,
    runner_info: list[RunnerInfo],
    supported_dtypes: set[cvcuda.Type] | None,
    supported_layouts: set[str] | None,
    supported_channels: set[int] | None,
    keystone_dlc: tuple[cvcuda.Type, str, int],
    exclude_dlc: list[tuple[cvcuda.Type | None, str | None, int | None]] | None = None,
    extra_params: dict[str, set] | None = None,
    extra_params_negative: dict[str, set] | None = None,
    exclude_extra_params: list[tuple[str, str]] | None = None,
    negative_exceptions: list[type[Exception]] | None = None,
) -> dict[str, Callable]:
    """Generate dtype/layout/channel-based test functions.

    Creates a positive test parametrized over every valid (runner, dtype, layout,
    channels, extra_combo) combination (after applying exclude_dlc filters),
    plus three negative tests that each vary one dimension against the keystone
    baseline: unsupported dtypes, unsupported layouts (tensor wrappers only),
    and unsupported channel counts.  Per-extra-param negative tests are added
    when extra_params_negative is provided.

    A ``None`` value for any supported set means "not specified": positive tests
    fall back to the keystone value for that dimension and negative tests are
    skipped.

    Returns a dict of ``test_op_{name}_*`` names to pytest-parametrized callables.
    """
    ctx = _RunnerContext(runner_info, extra_params, exclude_extra_params)
    keystone_dtype, keystone_layout, keystone_channels = keystone_dlc

    # Default None to keystone value for testing, empty for negatives
    dtypes_to_test = (
        supported_dtypes if supported_dtypes is not None else {keystone_dtype}
    )
    layouts_to_test = (
        supported_layouts if supported_layouts is not None else {keystone_layout}
    )
    channels_to_test = (
        supported_channels if supported_channels is not None else {keystone_channels}
    )

    unsupported_dtypes = (
        list(cv_types.SCALAR_TYPES_SET - dtypes_to_test)
        if supported_dtypes is not None
        else []
    )
    unsupported_layouts = (
        list(cv_types.IMAGE_LAYOUTS - layouts_to_test)
        if supported_layouts is not None
        else []
    )
    unsupported_channels = (
        list(cv_types.CHANNELS - channels_to_test)
        if supported_channels is not None
        else []
    )

    # --- Positive test ---
    input_params = []
    for runner_idx, dtype, layout, channels, extra_combo in itertools.product(
        range(ctx.runner_count),
        dtypes_to_test,
        layouts_to_test,
        channels_to_test,
        ctx.extra_param_combos,
    ):
        wrapper = ctx.get_wrapper(runner_idx)
        # Skip image wrappers for dtype/channel combos without format mappings
        if wrapper in ("image_batch", "image") and not _has_format_mapping(
            dtype, channels
        ):
            continue
        # Skip excluded combinations
        if _is_excluded(dtype, layout, channels, exclude_dlc):
            continue
        input_params.append((runner_idx, dtype, layout, channels, extra_combo))

    def _input_id(params):
        runner_idx, dtype, layout, channels, extra_combo = params
        return _dtype_test_id(
            ctx.labels, runner_idx, dtype, layout, channels, extra_combo
        )

    @pytest.mark.parametrize(
        "runner_idx,dtype,layout,channels,extra_combo",
        input_params,
        ids=[_input_id(p) for p in input_params],
    )
    def test_input(runner_idx, dtype, layout, channels, extra_combo):
        wrapper = ctx.get_wrapper(runner_idx)
        extra_dict = ctx.make_extra_dict(extra_combo, wrapper)
        op = ctx.get_op(runner_idx, dtype, layout, channels, extra_dict)
        assert_layouts(op, layout, dtype=dtype, wrapper=wrapper, channels=channels)

    # --- Negative dtype test ---
    dtype_neg_params = _build_negative_test_params(
        ctx,
        unsupported_dtypes,
        format_check_func=lambda dtype: _has_format_mapping(dtype, keystone_channels),
    )

    def _dtype_neg_body(runner_idx, dtype):
        wrapper = ctx.get_wrapper(runner_idx)
        extra_dict = ctx.get_first_extra_params(wrapper)
        op = ctx.get_op(
            runner_idx, dtype, keystone_layout, keystone_channels, extra_dict
        )
        assert_dtypes(
            op,
            dtype,
            wrapper=wrapper,
            channels=keystone_channels,
            layout=keystone_layout,
            negative=True,
            exceptions=negative_exceptions,
        )

    test_dtype_negative = _make_negative_test(
        dtype_neg_params,
        id_func=lambda idx, d: f"{ctx.labels[idx]}-{d.name}",
        test_body=_dtype_neg_body,
    )

    # --- Negative layout test (skip image wrappers - no layout validation) ---
    layout_neg_params = _build_negative_test_params(
        ctx, unsupported_layouts, skip_image_wrappers=True
    )

    def _layout_neg_body(runner_idx, layout):
        wrapper = ctx.get_wrapper(runner_idx)
        extra_dict = ctx.get_first_extra_params(wrapper)
        op = ctx.get_op(
            runner_idx, keystone_dtype, layout, keystone_channels, extra_dict
        )
        assert_layouts(
            op,
            layout,
            dtype=keystone_dtype,
            wrapper=wrapper,
            channels=keystone_channels,
            negative=True,
            exceptions=negative_exceptions,
        )

    test_layout_negative = _make_negative_test(
        layout_neg_params,
        id_func=lambda idx, layout: f"{ctx.labels[idx]}-{layout}",
        test_body=_layout_neg_body,
    )

    # --- Negative channels test ---
    channels_neg_params = _build_negative_test_params(
        ctx,
        unsupported_channels,
        format_check_func=lambda ch: _has_format_mapping(keystone_dtype, ch),
    )

    def _channels_neg_body(runner_idx, channels):
        wrapper = ctx.get_wrapper(runner_idx)
        extra_dict = ctx.get_first_extra_params(wrapper)
        op = ctx.get_op(
            runner_idx, keystone_dtype, keystone_layout, channels, extra_dict
        )
        assert_layouts(
            op,
            keystone_layout,
            dtype=keystone_dtype,
            wrapper=wrapper,
            channels=channels,
            negative=True,
            exceptions=negative_exceptions,
        )

    test_channels_negative = _make_negative_test(
        channels_neg_params,
        id_func=lambda idx, ch: f"{ctx.labels[idx]}-{ch}ch",
        test_body=_channels_neg_body,
    )

    # --- Negative extra_params tests ---
    def _get_op_for_extra_neg(runner_idx, extra_dict):
        return ctx.get_op(
            runner_idx, keystone_dtype, keystone_layout, keystone_channels, extra_dict
        )

    def _get_assert_kwargs(runner_idx):
        return {
            "layouts": keystone_layout,
            "dtype": keystone_dtype,
            "wrapper": ctx.get_wrapper(runner_idx),
            "channels": keystone_channels,
        }

    extra_neg_tests = _make_extra_param_negative_tests(
        name,
        extra_params_negative,
        ctx,
        _get_op_for_extra_neg,
        _get_assert_kwargs,
        assert_layouts,
        negative_exceptions,
    )

    # --- Build result ---
    return _collect_test_results(
        name,
        test_input,
        "input",
        [
            ("dtype_negative", test_dtype_negative),
            ("layout_negative", test_layout_negative),
            ("channels_negative", test_channels_negative),
        ],
        extra_neg_tests,
    )
