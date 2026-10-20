/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CVCUDA_PRIV_PIXEL_LAYOUT_COMMON_CUH
#define CVCUDA_PRIV_PIXEL_LAYOUT_COMMON_CUH

#include <cvcuda/Types.h>
#include <nvcv/BorderType.h>
#include <nvcv/Exception.hpp>

#include <type_traits>

namespace cvcuda::priv {

inline void ValidateBorderMode(NVCVBorderType borderMode, bool allowCoverage = false)
{
    if (allowCoverage && borderMode == NVCV_BORDER_CONSTANT_COVERAGE)
        return;

    if (!(borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REPLICATE || borderMode == NVCV_BORDER_REFLECT
          || borderMode == NVCV_BORDER_WRAP || borderMode == NVCV_BORDER_REFLECT101))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid border mode %d",
                              static_cast<int>(borderMode));
    }
}

// The NEAREST/LINEAR/CUBIC set shared by the warp family, Rotate and RandomResizedCrop; Resize admits more.
inline void ValidateInterpolation(int interpolation)
{
    if (!(interpolation == NVCV_INTERP_NEAREST || interpolation == NVCV_INTERP_LINEAR
          || interpolation == NVCV_INTERP_CUBIC))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid interpolation %d", interpolation);
    }
}

// Binds the runtime border mode to a compile-time tag so the wraps can specialize on it.
template<class Cb>
inline void DispatchBorderMode(NVCVBorderType borderMode, const Cb &cb)
{
    switch (borderMode)
    {
#define CVCUDA_PRIV_BORDER_CASE(BORDERTYPE)                   \
case BORDERTYPE:                                              \
    cb(std::integral_constant<NVCVBorderType, BORDERTYPE>{}); \
    return

        CVCUDA_PRIV_BORDER_CASE(NVCV_BORDER_CONSTANT);
        CVCUDA_PRIV_BORDER_CASE(NVCV_BORDER_REPLICATE);
        CVCUDA_PRIV_BORDER_CASE(NVCV_BORDER_REFLECT);
        CVCUDA_PRIV_BORDER_CASE(NVCV_BORDER_WRAP);
        CVCUDA_PRIV_BORDER_CASE(NVCV_BORDER_REFLECT101);

#undef CVCUDA_PRIV_BORDER_CASE
    default:
        break;
    }

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid border mode %d", static_cast<int>(borderMode));
}

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_PIXEL_LAYOUT_COMMON_CUH
