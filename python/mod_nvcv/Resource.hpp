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

#ifndef NVCV_PYTHON_PRIV_RESOURCE_HPP
#define NVCV_PYTHON_PRIV_RESOURCE_HPP

#include "Object.hpp"
#include "Stream.hpp"

#include <nvcv/detail/CudaFwd.h>
#include <nvcv/python/LockMode.hpp>
#include <pybind11/pybind11.h>

#include <memory>
#include <mutex>

// fwd declaration from driver_types.h
typedef struct CUevent_st *cudaEvent_t;

namespace nvcvpy::priv {
namespace py = pybind11;

/**
 * @brief A class representing a CUDA resource.
 *
 * This class encapsulates a CUDA resource and provides methods for synchronization
 * with CUDA streams.
 */
class PYBIND11_EXPORT Resource : public virtual Object
{
public:
    /**
     * @brief Destructor.
     */
    ~Resource();

    /**
     * @brief Export the Resource class to Python.
     *
     * @param m The Python module to export the class to.
     */
    static void Export(py::module &m);

    /**
     * @brief Get the unique identifier of the resource.
     *
     * @return uint64_t The unique identifier of the resource.
     */
    uint64_t id() const;

    /**
     * @brief Submit the resource for synchronization with a CUDA stream.
     *
     * This method synchronizes the resource with the specified CUDA stream.
     *
     * @param stream The CUDA stream to synchronize with.
     */
    void submitSync(Stream &stream);

    /**
     * @brief Get a shared pointer to this resource.
     *
     * @return std::shared_ptr<Resource> A shared pointer to this resource.
     */
    std::shared_ptr<Resource> shared_from_this();

    /**
     * @brief Get a shared pointer to this const resource.
     *
     * @return std::shared_ptr<const Resource> A shared pointer to this const resource.
     */
    std::shared_ptr<const Resource> shared_from_this() const;

protected:
    Resource();

private:
    uint64_t                                     m_id;         /**< The unique identifier of the resource. */
    cudaEvent_t                                  m_event;      /**< The CUDA event used for synchronization. */
    std::optional<std::shared_ptr<const Stream>> m_lastStream; /**< Cache the last stream used for this resource. */
    std::mutex                                   m_mtx;        /**< Lock reads and writes to the resource.  */

    cudaEvent_t event();
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_RESOURCE_HPP
