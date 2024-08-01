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

/**
 * @file Status.hpp
 *
 * @brief Declaration of NVCV C++ status codes handling functions.
 */

#ifndef NVCV_STATUS_HPP
#define NVCV_STATUS_HPP

#include "Status.h"

#include <cstdint>
#include <ostream>

namespace nvcv {

/**
 * @brief Enum class representing various status codes for operations.
 *
 * This enum is coupled to NVCVStatus, the status codes are the same.
 * For further details, see \ref NVCVStatus.
 * @defgroup NVCV_CPP_UTIL_STATUS Status Codes
 * @{
 */
enum class Status : int8_t
{
    SUCCESS                    = NVCV_SUCCESS,
    ERROR_NOT_IMPLEMENTED      = NVCV_ERROR_NOT_IMPLEMENTED,
    ERROR_INVALID_ARGUMENT     = NVCV_ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_IMAGE_FORMAT = NVCV_ERROR_INVALID_IMAGE_FORMAT,
    ERROR_INVALID_OPERATION    = NVCV_ERROR_INVALID_OPERATION,
    ERROR_DEVICE               = NVCV_ERROR_DEVICE,
    ERROR_NOT_READY            = NVCV_ERROR_NOT_READY,
    ERROR_OUT_OF_MEMORY        = NVCV_ERROR_OUT_OF_MEMORY,
    ERROR_INTERNAL             = NVCV_ERROR_INTERNAL,
    ERROR_NOT_COMPATIBLE       = NVCV_ERROR_NOT_COMPATIBLE,
    ERROR_OVERFLOW             = NVCV_ERROR_OVERFLOW,
    ERROR_UNDERFLOW            = NVCV_ERROR_UNDERFLOW
};

/**
 * @brief Retrieves the name (string representation) of the given status.
 *
 * @param status Status code whose name is to be retrieved.
 * @return String representation of the status.
 */
inline const char *GetName(Status status)
{
    return nvcvStatusGetName(static_cast<NVCVStatus>(status));
}

/**
 * @brief Overloads the stream insertion operator for Status enum.
 *
 * @param out Output stream to which the status string will be written.
 * @param status Status code to be output.
 * @return Reference to the modified output stream.
 */
inline std::ostream &operator<<(std::ostream &out, Status status)
{
    return out << static_cast<NVCVStatus>(status);
}

/**
 * @brief Overloads the stream insertion operator for NVCVStatus.
 *
 * @param out Output stream to which the status string will be written.
 * @param status NVCVStatus code to be output.
 * @return Reference to the modified output stream.
 */
inline std::ostream &operator<<(std::ostream &out, NVCVStatus status)
{
    return out << nvcvStatusGetName(status);
}

/**@}*/

} // namespace nvcv

#endif // NVCV_STATUS_HPP
