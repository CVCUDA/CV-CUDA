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

__attribute__((weak)) void *weak_nv_data = (void *)0;

void *strong_nv_data = (void *)0;

void *strong_common_data;

__attribute__((weak)) void weak_nv_func() {}

void strong_nv_func() {}

__attribute__((weak)) extern void *weak_undef_data;

extern void *strong_undef_data;

__attribute__((weak)) extern void weak_undef_func();

extern void strong_undef_func();

__thread void *tls_strong_nv_data = (void *)0;

static void ifunc_aux() {}

static void (*ifunc_resolver())()
{
    return ifunc_aux;
}

/* Define it before strong_ifunc on purpose, weak ifuncs must be
 * emitted after all strong ifuncs are emitted
 * gcc doesn't allow us to define directly a weak indirect function, we must do
 * it via a weak alias to a strong ifunc
 */
extern void weak_ifunc() __attribute__((weak, alias("strong_ifunc")));

extern void strong_ifunc() __attribute__((ifunc("ifunc_resolver")));
