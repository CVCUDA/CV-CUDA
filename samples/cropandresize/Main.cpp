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

#include <common/NvDecoder.h>
#include <common/TestUtils.h>
#include <cuda_runtime_api.h>
#include <cvcuda/OpCustomCrop.hpp>
#include <cvcuda/OpResize.hpp>
#include <getopt.h>
#include <math.h>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>

/**
 * @brief Crop and Resize sample app.
 *
 * The Crop and Resize is a simple pipeline which demonstrates usage of
 * CVCuda Tensor along with a few operators.
 *
 * Input Batch Tensor -> Crop -> Resize -> WriteImage
 */

/**
 * @brief Utility to show usage of sample app
 *
 **/
void showUsage()
{
    std::cout << "usage: ./nvcv_cropandresize_app -i <image file path or  image directory -b <batch size>" << std::endl;
}

/**
 * @brief Utility to parse the command line arguments
 *
 **/
int ParseArgs(int argc, char *argv[], std::string &imagePath, uint32_t &batchSize)
{
    static struct option long_options[] = {
        {     "help",       no_argument, 0, 'h'},
        {"imagePath", required_argument, 0, 'i'},
        {    "batch", required_argument, 0, 'b'},
        {          0,                 0, 0,   0}
    };

    int long_index = 0;
    int opt        = 0;
    while ((opt = getopt_long(argc, argv, "hi:b:", long_options, &long_index)) != -1)
    {
        switch (opt)
        {
        case 'h':
            showUsage();
            return -1;
            break;
        case 'i':
            imagePath = optarg;
            break;
        case 'b':
            batchSize = std::stoi(optarg);
            break;
        case ':':
            showUsage();
            return -1;
        default:
            break;
        }
    }
    std::ifstream imageFile(imagePath);
    if (!imageFile.good())
    {
        showUsage();
        std::cerr << "Image path '" + imagePath + "' does not exist\n";
        return -1;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    // Default parameters
    std::string imagePath = "./samples/assets/tabby_tiger_cat.jpg";
    uint32_t    batchSize = 1;

    // Parse the command line paramaters to override the default parameters
    int retval = ParseArgs(argc, argv, imagePath, batchSize);
    if (retval != 0)
    {
        return retval;
    }

    // NvJpeg is used to decode the images to the color format required.
    // Since we need a contiguous buffer for batched input, a buffer is
    // preallocated based on the  maximum image dimensions and  batch size
    // for NvJpeg to write into.

    // Note : The maximum input image dimensions needs to be updated in case
    // of testing with different test images

    int maxImageWidth  = 720;
    int maxImageHeight = 720;
    int maxChannels    = 3;

    // tag: Create the cuda stream
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // tag: Allocate input tensor
    // Allocating memory for RGBI input image batch of uint8_t data type
    // without padding since NvDecode utility currently doesnt support
    // Padded buffers.

    nvcv::TensorDataStridedCuda::Buffer inBuf;
    inBuf.strides[3] = sizeof(uint8_t);
    inBuf.strides[2] = maxChannels * inBuf.strides[3];
    inBuf.strides[1] = maxImageWidth * inBuf.strides[2];
    inBuf.strides[0] = maxImageHeight * inBuf.strides[1];
    CHECK_CUDA_ERROR(cudaMallocAsync(&inBuf.basePtr, batchSize * inBuf.strides[0], stream));

    // tag: Tensor Requirements
    // Calculate the requirements for the RGBI uint8_t Tensor which include
    // pitch bytes, alignment, shape  and tensor layout
    nvcv::Tensor::Requirements inReqs
        = nvcv::Tensor::CalcRequirements(batchSize, {maxImageWidth, maxImageHeight}, nvcv::FMT_RGB8);

    // Create a tensor buffer to store the data pointer and pitch bytes for each plane
    nvcv::TensorDataStridedCuda inData(nvcv::TensorShape{inReqs.shape, inReqs.rank, inReqs.layout},
                                       nvcv::DataType{inReqs.dtype}, inBuf);

    // TensorWrapData allows for interoperation of external tensor representations with CVCUDA Tensor.
    nvcv::TensorWrapData inTensor(inData);

    // tag: Image Loading
    // NvJpeg is used to load the images to create a batched input device buffer.
    uint8_t             *gpuInput = reinterpret_cast<uint8_t *>(inBuf.basePtr);
    // The total images is set to the same value as batch size for testing
    uint32_t             totalImages = batchSize;
    // Format in which the decoded output will be saved
    nvjpegOutputFormat_t outputFormat = NVJPEG_OUTPUT_RGBI;

    NvDecode(imagePath, batchSize, totalImages, outputFormat, gpuInput);

    // tag: The input buffer is now ready to be used by the operators

    // Set parameters for Crop and Resize
    // ROI dimensions to crop in the input image
    int cropX      = 150;
    int cropY      = 50;
    int cropWidth  = 400;
    int cropHeight = 300;

    // Set the resize dimensions
    int resizeWidth  = 320;
    int resizeHeight = 240;

    //  Initialize the CVCUDA ROI struct
    NVCVRectI crpRect = {cropX, cropY, cropWidth, cropHeight};

    // tag: Allocate Tensors for Crop and Resize
    // Create a CVCUDA Tensor based on the crop window size.
    nvcv::Tensor cropTensor(batchSize, {cropWidth, cropHeight}, nvcv::FMT_RGB8);
    // Create a CVCUDA Tensor based on resize dimensions
    nvcv::Tensor resizedTensor(batchSize, {resizeWidth, resizeHeight}, nvcv::FMT_RGB8);

#ifdef PROFILE_SAMPLE
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    // tag: Initialize operators for Crop and Resize
    cvcuda::CustomCrop cropOp;
    cvcuda::Resize     resizeOp;

    // tag: Executes the CustomCrop operation on the given cuda stream
    cropOp(stream, inTensor, cropTensor, crpRect);

    // Resize operator can now be enqueued into the same stream
    resizeOp(stream, cropTensor, resizedTensor, NVCV_INTERP_LINEAR);

    // tag: Profile section
#ifdef PROFILE_SAMPLE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float operatorms = 0;
    cudaEventElapsedTime(&operatorms, start, stop);
    std::cout << "Time for Crop and Resize : " << operatorms << " ms" << std::endl;
#endif

    // tag: Copy the buffer to CPU and write resized image into .bmp file
    WriteRGBITensor(resizedTensor, stream);

    // tag: Clean up
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    // tag: End of Sample
}
