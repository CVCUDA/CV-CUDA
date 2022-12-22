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

#include "TRTUtils.h"

#include "NvInfer.h"
#include "NvUtils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

class TRTLogger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override
    {
        switch (severity)
        {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        {
            std::cout << msg << std::endl;
            break;
        }
        case nvinfer1::ILogger::Severity::kERROR:
        {
            std::cout << msg << std::endl;
            break;
        }
        default:
        {
            break;
        }
        }
    }
};

struct TRTBackend::TRTImpl
{
    TRTImpl()
        : m_logger(new TRTLogger())
        , m_TRTRuntime(nullptr, [](nvinfer1::IRuntime *runtime) { runtime->destroy(); })
        , m_inferenceEngine(nullptr)
        , m_ownedInferenceEngine(nullptr, [](nvinfer1::ICudaEngine *eng) { eng->destroy(); })
        , m_inferContext(nullptr, [](nvinfer1::IExecutionContext *ectx) { ectx->destroy(); })
        , m_cudaStream(0)
    {
    }

    std::unique_ptr<TRTLogger>                                                            m_logger;
    std::unique_ptr<nvinfer1::IRuntime, void (*)(nvinfer1::IRuntime *)>                   m_TRTRuntime;
    nvinfer1::ICudaEngine                                                                *m_inferenceEngine;
    std::unique_ptr<nvinfer1::ICudaEngine, void (*)(nvinfer1::ICudaEngine *)>             m_ownedInferenceEngine;
    std::unique_ptr<nvinfer1::IExecutionContext, void (*)(nvinfer1::IExecutionContext *)> m_inferContext;

    cudaStream_t                         m_cudaStream;
    int                                  m_bindingsCount = 0;
    int                                  m_batchSize     = 1;
    std::unordered_map<std::string, int> m_blobMap;

    void loadNetWorkFromFile(const char *modelFilePath);
    // find the input/output bindings
    void setupIO(int batchSize);
};

void TRTBackend::TRTImpl::loadNetWorkFromFile(const char *modelFilePath)
{
    // Initialize TRT engine and deserialize it from file
    std::ifstream           trtModelFStream(modelFilePath, std::ios::binary);
    std::unique_ptr<char[]> trtModelContent;
    size_t                  trtModelContentSize = 0;
    if (!trtModelFStream.good())
    {
        std::cerr << "Model File: " << modelFilePath << std::endl;
        throw std::runtime_error("TensorRT: Model file not found.");
    }
    else
    {
        trtModelFStream.seekg(0, trtModelFStream.end);
        trtModelContentSize = trtModelFStream.tellg();
        trtModelFStream.seekg(0, trtModelFStream.beg);
        trtModelContent.reset(new char[trtModelContentSize]);
        trtModelFStream.read(trtModelContent.get(), trtModelContentSize);
        trtModelFStream.close();
        std::cout << "Deserializing engine from: " << modelFilePath;
    }
    m_TRTRuntime.reset(nvinfer1::createInferRuntime(*(m_logger.get())));
    m_inferenceEngine = dynamic_cast<nvinfer1::ICudaEngine *>(
        m_TRTRuntime->deserializeCudaEngine(trtModelContent.get(), trtModelContentSize, nullptr));
    m_ownedInferenceEngine.reset(m_inferenceEngine);
    m_inferContext.reset(m_inferenceEngine->createExecutionContext());
    m_inferContext->setOptimizationProfile(0);
}

void TRTBackend::TRTImpl::setupIO(int batchSize)
{
    // @TODO: use getBindingDimensions to avoid re-setting the IO.
    m_bindingsCount = m_inferenceEngine->getNbBindings();
    for (int i = 0; i < m_bindingsCount; i++)
    {
        m_blobMap[std::string(m_inferenceEngine->getBindingName(i))] = i;
        if (m_inferenceEngine->bindingIsInput(i))
        {
            nvinfer1::Dims  dims_i(m_inferenceEngine->getBindingDimensions(i));
            nvinfer1::Dims4 inputDims{batchSize, dims_i.d[1], dims_i.d[2], dims_i.d[3]};
            m_inferContext->setBindingDimensions(i, inputDims);
        }
    }
}

TRTBackend::TRTBackend(const char *modelFilePath, int batchSize)
    : m_pImpl(new TRTImpl())
{
    m_pImpl->m_batchSize = batchSize;
    m_pImpl->loadNetWorkFromFile(modelFilePath);
    m_pImpl->setupIO(m_pImpl->m_batchSize);
}

TRTBackend::~TRTBackend() {}

void TRTBackend::infer(void **buffer, int batchSize, cudaStream_t stream)
{
    m_pImpl->setupIO(batchSize);

    bool success = true;
    if (!m_pImpl->m_inferenceEngine->hasImplicitBatchDimension())
    {
        m_pImpl->m_inferContext->enqueueV2(buffer, stream, nullptr);
    }
    else
    {
        m_pImpl->m_inferContext->enqueue(batchSize, buffer, stream, nullptr);
    }

    if (!success)
    {
        throw std::runtime_error("TensorRT: Inference failed");
    }
}

int TRTBackend::getBlobCount() const
{
    return m_pImpl->m_bindingsCount;
}

TRTBackendBlobSize TRTBackend::getTRTBackendBlobSize(int blobIndex) const
{
    if (blobIndex >= m_pImpl->m_bindingsCount)
    {
        throw std::runtime_error("blobIndex out of range");
    }
    auto               dim = m_pImpl->m_inferenceEngine->getBindingDimensions(blobIndex);
    TRTBackendBlobSize blobSize;
    if (dim.nbDims == 2)
    {
        blobSize = {1, dim.d[0], dim.d[1]};
    }
    else if (dim.nbDims == 3)
    {
        blobSize = {dim.d[0], dim.d[1], dim.d[2]};
    }
    else if (dim.nbDims == 4)
    {
        blobSize = {dim.d[1], dim.d[2], dim.d[3]};
    }
    else
    {
        throw std::runtime_error("Unknown TensorRT binding dimension!");
    }
    return blobSize;
}

int TRTBackend::getBlobLinearSize(int blobIndex) const
{
    const TRTBackendBlobSize shape = getTRTBackendBlobSize(blobIndex);
    nvinfer1::Dims3          dims_val{shape.channels, shape.height, shape.width};
    int                      blobSize = 1;
    for (int i = 0; i < 3; i++)
    {
        blobSize *= dims_val.d[i] <= 0 ? 1 : dims_val.d[i];
    }
    return blobSize;
}

int TRTBackend::getBlobIndex(const char *blobName) const
{
    auto blobItr = m_pImpl->m_blobMap.find(std::string(blobName));
    if (blobItr == m_pImpl->m_blobMap.end())
    {
        throw std::runtime_error("blobName not found");
    }
    return blobItr->second;
}

bool TRTBackend::bindingIsInput(const int index) const
{
    return m_pImpl->m_inferenceEngine->bindingIsInput(index);
}

const char *TRTBackend::getLayerName(const int index) const
{
    return m_pImpl->m_ownedInferenceEngine->getBindingName(index);
}
