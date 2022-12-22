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

#include "ClassificationUtils.hpp"

#include <common/NvDecoder.h>
#include <common/TRTUtils.h>
#include <cuda_runtime_api.h>
#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/OpNormalize.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpResize.hpp>
#include <math.h>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <fstream>
#include <iostream>
#include <numeric>

/**
 * @brief Image classification sample.
 *
 * The image classification sample uses Resnet50 based model trained on Imagenet
 * The sample app pipeline includes preprocessing, inference and post process stages
 * which takes as input a batch of images and returns the TopN classification results
 * of each image.
 *
 */

/**
 * @brief Preprocess function
 *
 * @details Preprocessing includes the following sequence of operations.
 * Resize -> DataType Convert(U8->F32) -> Normalize( Apply mean and std deviation) -> Interleaved to Planar
 *
 * @param [in] inTensor CVCUDA Tensor containing the batched input images
 * @param [in] batchSize Batch size of the input tensor
 * @param [in] inputLayerWidth Input Layer width of the network
 * @param [in] inputLayerHeight Input Layer height of the network
 * @param [in] stream Cuda stream
 *
 * @param [out] outTensor  CVCUDA Tensor containing the preprocessed image batch
 *
 */

void PreProcess(nvcv::TensorWrapData &inTensor, uint32_t batchSize, int inputLayerWidth, int inputLayerHeight,
                cudaStream_t stream, nvcv::TensorWrapData &outTensor)
{
#ifdef PROFILE_SAMPLE
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif

    // Resize to the dimensions of input layer of network
    nvcv::Tensor   resizedTensor(batchSize, {inputLayerWidth, inputLayerHeight}, nvcv::FMT_RGB8);
    cvcuda::Resize resizeOp;
    resizeOp(stream, inTensor, resizedTensor, NVCV_INTERP_LINEAR);

    // Convert to data format expected by network (F32). Apply scale 1/255f.
    nvcv::Tensor      floatTensor(batchSize, {inputLayerWidth, inputLayerHeight}, nvcv::FMT_RGBf32);
    cvcuda::ConvertTo convertOp;
    convertOp(stream, resizedTensor, floatTensor, 1.0f / 255.f, 0.0f);

    // The input to the network needs to be normalized based on the mean and std deviation values
    // to standardize the input data.

    // Create a Tensor to store the standard deviation values for R,G,B
    nvcv::Tensor::Requirements reqsScale       = nvcv::Tensor::CalcRequirements(1, {1, 1}, nvcv::FMT_RGBf32);
    int64_t                    scaleBufferSize = CalcTotalSizeBytes(nvcv::Requirements{reqsScale.mem}.cudaMem());
    nvcv::TensorDataStridedCuda::Buffer bufScale;
    std::copy(reqsScale.strides, reqsScale.strides + NVCV_TENSOR_MAX_RANK, bufScale.strides);
    CHECK_CUDA_ERROR(cudaMalloc(&bufScale.basePtr, scaleBufferSize));
    nvcv::TensorDataStridedCuda scaleIn(nvcv::TensorShape{reqsScale.shape, reqsScale.rank, reqsScale.layout},
                                        nvcv::DataType{reqsScale.dtype}, bufScale);
    nvcv::TensorWrapData        scaleTensor(scaleIn);

    // Create a Tensor to store the mean values for R,G,B
    nvcv::TensorDataStridedCuda::Buffer bufBase;
    nvcv::Tensor::Requirements          reqsBase       = nvcv::Tensor::CalcRequirements(1, {1, 1}, nvcv::FMT_RGBf32);
    int64_t                             baseBufferSize = CalcTotalSizeBytes(nvcv::Requirements{reqsBase.mem}.cudaMem());
    std::copy(reqsBase.strides, reqsBase.strides + NVCV_TENSOR_MAX_RANK, bufBase.strides);
    CHECK_CUDA_ERROR(cudaMalloc(&bufBase.basePtr, baseBufferSize));
    nvcv::TensorDataStridedCuda baseIn(nvcv::TensorShape{reqsBase.shape, reqsBase.rank, reqsBase.layout},
                                       nvcv::DataType{reqsBase.dtype}, bufBase);
    nvcv::TensorWrapData        baseTensor(baseIn);

    // Copy the values from Host to Device
    // The R,G,B scale and mean will be applied to all the pixels across the batch of input images
    const auto *baseData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(scaleTensor.exportData());
    const auto *scaleData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(baseTensor.exportData());
    float       scale[3]  = {0.229, 0.224, 0.225};
    float       base[3]   = {0.485f, 0.456f, 0.406f};

    // Flag to set the scale value as standard deviation i.e use 1/scale
    uint32_t flags = CVCUDA_NORMALIZE_SCALE_IS_STDDEV;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(scaleData->basePtr(), scale, 3 * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(baseData->basePtr(), base, 3 * sizeof(float), cudaMemcpyHostToDevice, stream));

    nvcv::Tensor normTensor(batchSize, {inputLayerWidth, inputLayerHeight}, nvcv::FMT_RGBf32);

    // Normalize
    cvcuda::Normalize normOp;
    normOp(stream, floatTensor, baseTensor, scaleTensor, normTensor, 1.0f, 0.0f, 0.0f, flags);

    // Convert the data layout from interleaved to planar
    cvcuda::Reformat reformatOp;
    reformatOp(stream, normTensor, outTensor);

#ifdef PROFILE_SAMPLE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float preprocessms = 0;
    cudaEventElapsedTime(&preprocessms, start, stop);
    std::cout << "\nTime for Preprocess : " << preprocessms << " ms" << std::endl;
#endif
}

/**
 * @brief Postprocess function
 *
 * @details Postprocessing function normalizes the classification score from the network and sorts
 *           the scores to get the TopN classification scores.
 *
 * @param [in] outputCudaBuffer Classification scores from the network
 * @param [in] stream Cuda Stream
 *
 * @param [out] scores Vector to store the sorted scores
 * @param [out] indices Vector to store the sorted indices
 */
void PostProcess(float *outputCudaBuffer, std::vector<std::vector<float>> &scores,
                 std::vector<std::vector<int>> &indices, cudaStream_t stream)
{
#ifdef PROFILE_SAMPLE
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif

    uint32_t batchSize  = scores.size();
    uint32_t numClasses = scores[0].size();

    // Copy the network classification scores from Device to Host
    for (int i = 0; i < batchSize; i++)
    {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(scores[i].data(), outputCudaBuffer + i * numClasses,
                                         numClasses * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    for (int i = 0; i < batchSize; i++)
    {
        // Apply softmax to normalize the scores in the range 0-1
        std::transform(scores[i].begin(), scores[i].end(), scores[i].begin(), [](float val) { return std::exp(val); });

        auto sum = std::accumulate(scores[i].begin(), scores[i].end(), 0.0);
        std::transform(scores[i].begin(), scores[i].end(), scores[i].begin(), [sum](float val) { return val / sum; });
        // Sort the indices based on scores
        std::iota(indices[i].begin(), indices[i].end(), 0);
        std::sort(indices[i].begin(), indices[i].end(),
                  [&scores, i](int i1, int i2) { return scores[i][i1] > scores[i][i2]; });
    }
#ifdef PROFILE_SAMPLE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float postprocessms = 0;
    cudaEventElapsedTime(&postprocessms, start, stop);
    std::cout << "Time for Postprocess : " << postprocessms << " ms" << std::endl;
#endif
}

int main(int argc, char *argv[])
{
    // Default parameters
    std::string modelPath = "./engines/resnet50.engine";
    std::string imagePath = "./samples/assets/tabby_tiger_cat.jpg";
    std::string labelPath = "./engines/imagenet-classes.txt";
    uint32_t    batchSize = 1;

    // Parse the command line paramaters to override the default parameters
    int retval = ParseArgs(argc, argv, modelPath, imagePath, labelPath, batchSize);
    if (retval != 0)
    {
        return retval;
    }

    // The total images is set to the batch size for testing
    uint32_t             totalImages  = batchSize;
    nvjpegOutputFormat_t outputFormat = NVJPEG_OUTPUT_BGRI;

    // Allocate the maximum memory neeed for the input image batch
    // Note : This needs to be changed in case of testing with different test images

    int maxImageWidth  = 720;
    int maxImageHeight = 720;
    int maxChannels    = 3;

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // Allocating memory for input image batch
    nvcv::TensorDataStridedCuda::Buffer inBuf;
    inBuf.strides[3] = sizeof(uint8_t);
    inBuf.strides[2] = maxChannels * inBuf.strides[3];
    inBuf.strides[1] = maxImageWidth * inBuf.strides[2];
    inBuf.strides[0] = maxImageHeight * inBuf.strides[1];
    CHECK_CUDA_ERROR(cudaMallocAsync(&inBuf.basePtr, batchSize * inBuf.strides[0], stream));

    nvcv::Tensor::Requirements inReqs
        = nvcv::Tensor::CalcRequirements(batchSize, {maxImageWidth, maxImageHeight}, nvcv::FMT_RGB8);

    nvcv::TensorDataStridedCuda inData(nvcv::TensorShape{inReqs.shape, inReqs.rank, inReqs.layout},
                                       nvcv::DataType{inReqs.dtype}, inBuf);
    nvcv::TensorWrapData        inTensor(inData);

    // NvJpeg is used to load the images to create a batched input device buffer.
    uint8_t *gpuInput = reinterpret_cast<uint8_t *>(inBuf.basePtr);
    NvDecode(imagePath, batchSize, totalImages, outputFormat, gpuInput);

    // TensorRT is used for the inference which loads the serialized engine file which is generated from the onnx model.
    // Initialize TensorRT backend
    std::unique_ptr<TRTBackend> trtBackend;
    trtBackend.reset(new TRTBackend(modelPath.c_str()));

    // Get number of input and output Layers
    auto numBindings = trtBackend->getBlobCount();
    if (numBindings != 2)
    {
        std::cerr << "Number of bindings should be 2\n";
        return -1;
    }

    // Initialize TensorRT Buffers
    std::vector<void *> buffers;
    buffers.resize(numBindings);

    // Get dimensions of input and output layers
    TRTBackendBlobSize inputDims, outputDims;
    uint32_t           inputBindingIndex, outputBindingIndex;
    for (int i = 0; i < numBindings; i++)
    {
        if (trtBackend->bindingIsInput(i))
        {
            inputDims         = trtBackend->getTRTBackendBlobSize(i);
            inputBindingIndex = i;
        }
        else
        {
            outputDims         = trtBackend->getTRTBackendBlobSize(i);
            outputBindingIndex = i;
        }
    }

    // Allocate input layer buffer based on input layer dimensions and batch size
    // Calculates the resource requirements needed to create a tensor with given shape
    nvcv::Tensor::Requirements reqsInputLayer
        = nvcv::Tensor::CalcRequirements(batchSize, {inputDims.width, inputDims.height}, nvcv::FMT_RGBf32p);
    // Calculates the total buffer size needed based on the requirements
    int64_t inputLayerSize = CalcTotalSizeBytes(nvcv::Requirements{reqsInputLayer.mem}.cudaMem());
    nvcv::TensorDataStridedCuda::Buffer bufInputLayer;
    std::copy(reqsInputLayer.strides, reqsInputLayer.strides + NVCV_TENSOR_MAX_RANK, bufInputLayer.strides);
    // Allocate buffer size needed for the tensor
    CHECK_CUDA_ERROR(cudaMalloc(&bufInputLayer.basePtr, inputLayerSize));
    // Wrap the tensor as a CVCUDA tensor
    nvcv::TensorDataStridedCuda inputLayerTensorData(
        nvcv::TensorShape{reqsInputLayer.shape, reqsInputLayer.rank, reqsInputLayer.layout},
        nvcv::DataType{reqsInputLayer.dtype}, bufInputLayer);
    nvcv::TensorWrapData inputLayerTensor(inputLayerTensorData);

    // Allocate ouput layer buffer based on the output layer dimensions and batch size
    nvcv::Tensor::Requirements reqsOutputLayer
        = nvcv::Tensor::CalcRequirements(batchSize, {outputDims.width, 1}, nvcv::FMT_RGBf32p);
    int64_t outputLayerSize = CalcTotalSizeBytes(nvcv::Requirements{reqsOutputLayer.mem}.cudaMem());
    nvcv::TensorDataStridedCuda::Buffer bufOutputLayer;
    std::copy(reqsOutputLayer.strides, reqsOutputLayer.strides + NVCV_TENSOR_MAX_RANK, bufOutputLayer.strides);
    CHECK_CUDA_ERROR(cudaMalloc(&bufOutputLayer.basePtr, outputLayerSize));
    nvcv::TensorDataStridedCuda outputLayerTensorData(
        nvcv::TensorShape{reqsOutputLayer.shape, reqsOutputLayer.rank, reqsOutputLayer.layout},
        nvcv::DataType{reqsOutputLayer.dtype}, bufOutputLayer);
    nvcv::TensorWrapData outputLayerTensor(outputLayerTensorData);

    // Run preprocess on the input image batch
    PreProcess(inTensor, batchSize, inputDims.width, inputDims.height, stream, inputLayerTensor);

    // Setup the TensortRT Buffer needed for inference
    const auto *inputData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(inputLayerTensor.exportData());
    const auto *outputData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(outputLayerTensor.exportData());

    buffers[inputBindingIndex]  = inputData->basePtr();
    buffers[outputBindingIndex] = outputData->basePtr();

#ifdef PROFILE_SAMPLE
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    // Inference call
    trtBackend->infer(&buffers[inputBindingIndex], batchSize, stream);
#ifdef PROFILE_SAMPLE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float inferms = 0;
    cudaEventElapsedTime(&inferms, start, stop);
    std::cout << "Time for Inference : " << inferms << " ms" << std::endl;
#endif

    // Post Process to normalize and sort the classifications scores
    uint32_t                        numClasses = outputDims.width;
    std::vector<std::vector<float>> scores(batchSize, std::vector<float>(numClasses));
    std::vector<std::vector<int>>   indices(batchSize, std::vector<int>(numClasses));
    PostProcess((float *)outputData->basePtr(), scores, indices, stream);

    // Display Results
    DisplayResults(scores, indices, labelPath);

    // Clean up
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
