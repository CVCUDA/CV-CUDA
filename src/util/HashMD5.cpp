/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "HashMD5.hpp"

#include "Assert.h"

#include <openssl/evp.h>

#include <cstring>

namespace nvcv::util {

struct HashMD5::Impl
{
    EVP_MD_CTX *ctx;
};

HashMD5::HashMD5()
    : pimpl{std::make_unique<Impl>()}
{
    pimpl->ctx = EVP_MD_CTX_create();
    NVCV_ASSERT(pimpl->ctx != nullptr);

    int ret = EVP_DigestInit_ex(pimpl->ctx, EVP_md5(), NULL);
    NVCV_ASSERT(ret == 1);
}

HashMD5::~HashMD5()
{
    EVP_MD_CTX_destroy(pimpl->ctx);
}

void HashMD5::operator()(const void *data, size_t lenBytes)
{
    int ret = EVP_DigestUpdate(pimpl->ctx, data, lenBytes);
    NVCV_ASSERT(ret == 1);
}

std::array<uint8_t, 16> HashMD5::getHashAndReset()
{
    unsigned char buf[EVP_MAX_MD_SIZE];
    unsigned int  nwritten = sizeof(buf);
    // it also resets the context
    int           ret = EVP_DigestFinal(pimpl->ctx, buf, &nwritten);
    NVCV_ASSERT(ret == 1);

    // Be ready for a new run
    ret = EVP_DigestInit_ex(pimpl->ctx, EVP_md5(), NULL);
    NVCV_ASSERT(ret == 1);

    NVCV_ASSERT(nwritten == 16);
    std::array<uint8_t, 16> hash;
    memcpy(&hash[0], buf, sizeof(hash));
    return hash;
}

void Update(HashMD5 &hash, const char *value)
{
    if (value)
    {
        Update(hash, std::string_view(value));
    }
    else
    {
        Update(hash, -28374);
    }
}

} // namespace nvcv::util
