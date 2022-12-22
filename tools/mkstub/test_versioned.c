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

__asm__(".symver weak_ver1_data,weak_data@ver_1.0");
__attribute__((weak)) void *weak_ver1_data = (void *)0;
;

__asm__(".symver weak_ver2_data,weak_data@@ver_2.0");
__attribute__((weak)) void *weak_ver2_data = (void *)0;

__asm__(".symver strong_ver1_data,strong_data@ver_1.0");
void *strong_ver1_data = (void *)0;
;

__asm__(".symver strong_ver2_data,strong_data@@ver_2.0");
void *strong_ver2_data = (void *)0;
;

__asm__(".symver weak_ver1_func,weak_func@ver_1.0");

__attribute__((weak)) void weak_ver1_func() {}

__asm__(".symver weak_ver2_func,weak_func@@ver_2.0");

__attribute__((weak)) void weak_ver2_func() {}

__asm__(".symver strong_ver1_func,strong_func@ver_1.0");

void strong_ver1_func() {}

__asm__(".symver strong_ver2_func,strong_func@@ver_2.0");

void strong_ver2_func() {}

__asm__(".symver clash_ver2_func,clash_func_@@ver_2.0");

void clash_ver2_func() {}

void clash_func() {}
