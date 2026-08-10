
[//]: # "SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"
[//]: # ""
[//]: # "Licensed under the Apache License, Version 2.0 (the 'License');"
[//]: # "you may not use this file except in compliance with the License."
[//]: # "You may obtain a copy of the License at"
[//]: # "http://www.apache.org/licenses/LICENSE-2.0"
[//]: # ""
[//]: # "Unless required by applicable law or agreed to in writing, software"
[//]: # "distributed under the License is distributed on an 'AS IS' BASIS"
[//]: # "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied."
[//]: # "See the License for the specific language governing permissions and"
[//]: # "limitations under the License."

# CV-CUDA Samples

CV-CUDA Python samples showcase the use of various CV-CUDA APIs to construct fully functional end-to-end deep learning inference pipelines.

## Quick Start

### Quick Test (Hello World Only)

For quick testing with just the `hello_world.py` sample:

**For CUDA 12:**
```shell
python3 -m venv venv_samples
source venv_samples/bin/activate
python3 -m pip install -r requirements.samples.hello_world_cu12.txt
python3 applications/hello_world.py
```

**For CUDA 13:**
```shell
python3 -m venv venv_samples
source venv_samples/bin/activate
python3 -m pip install -r requirements.samples.hello_world_cu13.txt
python3 applications/hello_world.py
```

This installs only 4 packages (CV-CUDA, NumPy, cuda-python, nvImageCodec).

### Full Installation (All Samples)

Install CV-CUDA and sample dependencies using the installation script:

```shell
cd samples
./install_samples_dependencies.sh
```

This script will:
- Detect your CUDA version (12 or 13)
- Create a virtual environment at `venv_samples`
- Install all required dependencies including CV-CUDA, PyTorch, NumPy, and sample-specific packages (including interoperability dependencies like CuPy, PyCUDA, PyNvVideoCodec) from self-contained requirements files

**Note:** Full samples require Python 3.10-3.13 on x86_64/amd64 platforms

After installation, activate the virtual environment:

```shell
source venv_samples/bin/activate
```

### Running Samples

Run individual samples:

```shell
python3 operators/label.py
python3 applications/classification.py
python3 interoperability/pytorch_interop.py
```

Or run all samples at once:

```shell
./run_samples.sh     # All sample categories (operators, applications, interoperability, etc.)
```

## Documentation

For detailed documentation, tutorials, and API reference:

- **[CV-CUDA Samples Documentation](https://cvcuda.github.io/CV-CUDA/samples.html)** - Complete samples guide
  - [Installation Instructions](https://cvcuda.github.io/CV-CUDA/samples.html#samples-venv-installation) - Virtual environment setup
  - [Hello World Tutorial](https://cvcuda.github.io/CV-CUDA/samples.html#cv-cuda-hello-world) - Getting started
  - [Running the Samples](https://cvcuda.github.io/CV-CUDA/samples.html#running-the-samples) - Execution guide
  - [Sample Index](https://cvcuda.github.io/CV-CUDA/samples.html#sample-index) - Browse all samples

- **[Installation Guide](https://cvcuda.github.io/CV-CUDA/installation.html)** - CV-CUDA installation options
  - [Python Wheels (PyPI)](https://cvcuda.github.io/CV-CUDA/installation.html#python-wheels-pypi) - Quick pip install
  - [Building from Source](https://cvcuda.github.io/CV-CUDA/installation.html#building-from-source) - Custom builds
  - [Prerequisites](https://cvcuda.github.io/CV-CUDA/installation.html#prerequisites) - System requirements

- **[Interoperability Guide](https://cvcuda.github.io/CV-CUDA/samples/interoperability.html)** - Using CV-CUDA with other frameworks
