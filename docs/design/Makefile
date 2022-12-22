# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.PHONY: all

DIAGRAMS = allocator_classes \
	   resource_allocator_classes \
	   image_classes image_future_classes \
	   image_data_classes image_data_future_classes

all: $(addsuffix .eps,$(DIAGRAMS))

%.eps: %.puml
	plantuml -teps $^ $@

clean:
	rm -f $(addsuffix .eps,$(DIAGRAMS))
