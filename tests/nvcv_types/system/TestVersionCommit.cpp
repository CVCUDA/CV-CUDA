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

#include "Definitions.hpp"

#include <nvcv/detail/VersionUtils.h> // for NVCV_COMMIT

TEST(VersionTest, commit_hash_macro_exists)
{
#ifdef NVCV_COMMIT
    // Just some random, valid commit hash
    EXPECT_EQ(strlen("5335ae6bb8161d8e7f05896288f289cd84517e47"), strlen(NVCV_COMMIT));
#else
    FAIL() << "NVCV_COMMIT not defined";
#endif
}
