/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NVCV_TENSORLAYOUT_H
#define NVCV_TENSORLAYOUT_H

#include "Export.h"
#include "Status.h"
#include "detail/CompilerUtils.h"

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Maximum number of dimensions of a tensor */
#define NVCV_TENSOR_MAX_RANK (15)

/** Represents the tensor layout.
 * It assigns labels to each tensor dimension.
 */
typedef struct NVCVTensorLayoutRec
{
    // Not to be used directly.
    char    data[NVCV_TENSOR_MAX_RANK + 1]; // +1 for '\0'
    int32_t rank;
} NVCVTensorLayout;

typedef enum
{
    NVCV_TLABEL_BATCH   = 'N',
    NVCV_TLABEL_CHANNEL = 'C',
    NVCV_TLABEL_FRAME   = 'F',
    NVCV_TLABEL_DEPTH   = 'D',
    NVCV_TLABEL_HEIGHT  = 'H',
    NVCV_TLABEL_WIDTH   = 'W'
} NVCVTensorLabel;

#ifdef __cplusplus
#    define NVCV_TENSOR_LAYOUT_MAKE(layout) \
        NVCVTensorLayout                    \
        {                                   \
            layout, sizeof(layout) - 1      \
        }
#else
#    define NVCV_TENSOR_LAYOUT_MAKE(layout)            \
        {                                              \
            .data = layout, .rank = sizeof(layout) - 1 \
        }
#endif

NVCV_CONSTEXPR static const NVCVTensorLayout NVCV_TENSOR_NONE = NVCV_TENSOR_LAYOUT_MAKE("");

#define NVCV_DETAIL_DEF_TLAYOUT(LAYOUT) \
    NVCV_CONSTEXPR static const NVCVTensorLayout NVCV_TENSOR_##LAYOUT = NVCV_TENSOR_LAYOUT_MAKE(#LAYOUT);
#include "TensorLayoutDef.inc"
#undef NVCV_DETAIL_DEF_TLAYOUT

// clang-format off
NVCV_CONSTEXPR static const NVCVTensorLayout NVCV_TENSOR_IMPLICIT[7] =
{
    // Can't use the NVCV_TENSOR_* identifiers directly,
    // clang complains they are not compile-time constants.
    // We must resort to the make macros instead.
    NVCV_TENSOR_LAYOUT_MAKE(""), // none
    NVCV_TENSOR_LAYOUT_MAKE("W"),
    NVCV_TENSOR_LAYOUT_MAKE("HW"),
    NVCV_TENSOR_LAYOUT_MAKE("NHW"),
    NVCV_TENSOR_LAYOUT_MAKE("NCHW"),
    NVCV_TENSOR_LAYOUT_MAKE("NCDHW"),
    NVCV_TENSOR_LAYOUT_MAKE("NCFDHW"),
};
// clang-format on

/** Makes a tensor layout from a string defining the labels of its dimensions.
 *
 * The number of dimensions is taken from the string length.
 *
 * @param [in] descr Zero-terminated string,
 *                   + Must have at most @NVCV_TENSOR_MAX_RANK characters.
 * @param [out] layout Where the output layout will be written to
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorLayoutMake(const char *descr, NVCVTensorLayout *layout);

/** Makes a tensor layout from a label range.
 *
 * The number of dimensions is taken from the range length.
 *
 * @param [in] beg, end Label range.
 *                   + Must not be NULL
 *                   + Must specify at most @NVCV_TENSOR_MAX_RANK characters
 *
 * @param [out] layout Where the output layout will be written to
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorLayoutMakeRange(const char *beg, const char *end, NVCVTensorLayout *layout);

/** Makes a tensor layout from the first n labels of an existing layout.
 *
 * @param [in] layout Layout to copy from
 * @param [in] n Number of labels from start of the layout.
 *               - If 0, returns an empty layout.
 *               - If negative, returns the -n last labels instead.
 *               - If >= rank or <= -rank, returns a copy of the input layout.
 *
 * @param [out] layout Where the output layout will be written to
 *              + Must not be NULL
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorLayoutMakeFirst(NVCVTensorLayout in, int32_t n, NVCVTensorLayout *layout);

/** Makes a tensor layout from the last n labels of an existing layout.
 *
 * @param [in] layout Layout to copy from
 * @param [in] n Number of labels from end of the layout.
 *               - If >= rank or <= -rank, returns a copy of the input layout.
 *               - If 0, returns an empty layout.
 *               - If negative, returns the -n first labels instead.
 *
 * @param [out] layout Where the output layout will be written to
 *              + Must not be NULL
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorLayoutMakeLast(NVCVTensorLayout in, int32_t n, NVCVTensorLayout *layout);

/** Makes a tensor layout from the labels of a subrange of an existing layout .
 *
 * @param [in] layout Layout to copy from
 * @param [in] beg,end Range from input layout to be copied.
 *                   - If >= rank, consider at rank
 *                   - If <= -rank, consider at 0
 *                   - If < 0, consider at rank+beg (rank+end).
 *
 * @param [out] layout Where the output layout will be written to
 *              + Must not be NULL
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorLayoutMakeSubRange(NVCVTensorLayout in, int32_t beg, int32_t end,
                                                    NVCVTensorLayout *layout);

/** Returns the index of the dimension in the layout given its label.
 *
 * @param [in] layout Layout where the label is to be searched.
 * @param [in] dimLabel Label of the dimension to be queried
 * @param [in] idxStart Index to start searching.
 *
 * @returns Index of the dimension in the layout, or -1 if not found.
 */
inline static int32_t nvcvTensorLayoutFindDimIndex(NVCVTensorLayout layout, char dimLabel, int idxStart)
{
    if (idxStart < 0)
    {
        idxStart = layout.rank + idxStart;
    }

    int n = layout.rank - idxStart;
    if (n > 0)
    {
        void *p = memchr(layout.data + idxStart, dimLabel, n);
        if (p != NULL)
        {
            return (int32_t)((char *)p - (char *)layout.data);
        }
    }
    return -1;
}

/** Returns the layout label at the given index.
 *
 * @param [in] layout Layout to be queried
 * @param [in] idx Index of the label
 *
 * @returns If @p idx >= 0 and < layout size, returns the correspondign label.
 *          Returns '\0' otherwise.
 */
NVCV_CONSTEXPR inline static char nvcvTensorLayoutGetLabel(NVCVTensorLayout layout, int idx)
{
    // Must be all a single statement for C++11 compatibility
    return idx < 0 ? (0 <= layout.rank + idx && layout.rank + idx < layout.rank ? layout.data[layout.rank + idx] : '\0')
                   : (0 <= idx && idx < layout.rank ? layout.data[idx] : '\0');
}

/** Returns the number of dimensions of the tensor layout
 *
 * @param [in] layout Layout to be queried
 *
 * @returns Number of dimensions.
 */
NVCV_CONSTEXPR inline static int32_t nvcvTensorLayoutGetNumDim(NVCVTensorLayout layout)
{
    return layout.rank;
}

/** Compares the two layouts.
 *
 * @param [in] a,b Layouts to be compared
 *
 * @returns <0, 0 or >0 if @p a compares less than, equal to, or greater than @p b, respectivelt.
 */
inline static int32_t nvcvTensorLayoutCompare(NVCVTensorLayout a, NVCVTensorLayout b)
{
    if (a.rank == b.rank)
    {
        return memcmp(a.data, b.data, a.rank);
    }
    else
    {
        return a.rank - b.rank;
    }
}

/** Returns whether the layout starts with a given test layout.
 *
 * If test layout is empty, it's always considered to be found at the start.
 *
 * @param [in] layout Layout to be queried
 * @param [in] test Layout to be found
 *
 * @returns !=0 if found. 0 otherwise.
 */
inline static int32_t nvcvTensorLayoutStartsWith(NVCVTensorLayout layout, NVCVTensorLayout test)
{
    if (test.rank <= layout.rank)
    {
        return memcmp(test.data, layout.data, test.rank) == 0;
    }
    else
    {
        return 0;
    }
}

/** Returns whether the layout ends with a given test layout.
 *
 * If test layout is empty, it's always considered to be found at the end.
 *
 * @param [in] layout Layout to be queried
 * @param [in] test Layout to be found
 *
 * @returns !=0 if found. 0 otherwise.
 */
inline static int32_t nvcvTensorLayoutEndsWith(NVCVTensorLayout layout, NVCVTensorLayout test)
{
    if (test.rank <= layout.rank)
    {
        return memcmp(test.data, layout.data + layout.rank - test.rank, test.rank) == 0;
    }
    else
    {
        return 0;
    }
}

/** Returns the string representation of the tensor layout
 *
 * @param [in] layout Layout to be queried
 *
 * @returns Null-terminated string with the layout name.
 */
NVCV_CONSTEXPR inline static const char *nvcvTensorLayoutGetName(const NVCVTensorLayout *layout)
{
    return layout == NULL ? "" : layout->data;
}

#ifdef __cplusplus
}
#endif

#endif // NVCV_TENSORLAYOUT_H
