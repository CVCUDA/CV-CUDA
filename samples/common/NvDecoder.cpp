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

#include "NvDecoder.h"

#include "TestUtils.h"

int decode_images(const FileData &img_data, const std::vector<size_t> &img_len, std::vector<nvjpegImage_t> &out,
                  decode_params_t &params, double &time)
{
    CHECK_CUDA(cudaStreamSynchronize(params.stream));
    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    float       loopTime = 0;

    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    std::vector<const unsigned char *> batched_bitstreams;
    std::vector<size_t>                batched_bitstreams_size;
    std::vector<nvjpegImage_t>         batched_output;

    // bit-streams that batched decode cannot handle
    std::vector<const unsigned char *> otherdecode_bitstreams;
    std::vector<size_t>                otherdecode_bitstreams_size;
    std::vector<nvjpegImage_t>         otherdecode_output;

    if (params.hw_decode_available)
    {
        for (int i = 0; i < params.batch_size; i++)
        {
            // extract bitstream meta data to figure out whether a bit-stream can be decoded
            nvjpegJpegStreamParseHeader(params.nvjpeg_handle, (const unsigned char *)img_data[i].data(), img_len[i],
                                        params.jpeg_streams[0]);
            int isSupported = -1;
            nvjpegDecodeBatchedSupported(params.nvjpeg_handle, params.jpeg_streams[0], &isSupported);

            if (isSupported == 0)
            {
                batched_bitstreams.push_back((const unsigned char *)img_data[i].data());
                batched_bitstreams_size.push_back(img_len[i]);
                batched_output.push_back(out[i]);
            }
            else
            {
                otherdecode_bitstreams.push_back((const unsigned char *)img_data[i].data());
                otherdecode_bitstreams_size.push_back(img_len[i]);
                otherdecode_output.push_back(out[i]);
            }
        }
    }
    else
    {
        for (int i = 0; i < params.batch_size; i++)
        {
            otherdecode_bitstreams.push_back((const unsigned char *)img_data[i].data());
            otherdecode_bitstreams_size.push_back(img_len[i]);
            otherdecode_output.push_back(out[i]);
        }
    }

    CHECK_CUDA(cudaEventRecord(startEvent, params.stream));

    if (batched_bitstreams.size() > 0)
    {
        CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(params.nvjpeg_handle, params.nvjpeg_state, batched_bitstreams.size(),
                                                   1, params.fmt));

        CHECK_NVJPEG(nvjpegDecodeBatched(params.nvjpeg_handle, params.nvjpeg_state, batched_bitstreams.data(),
                                         batched_bitstreams_size.data(), batched_output.data(), params.stream));
    }

    if (otherdecode_bitstreams.size() > 0)
    {
        CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(params.nvjpeg_decoupled_state, params.device_buffer));
        int buffer_index = 0;
        CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(params.nvjpeg_decode_params, params.fmt));
        for (int i = 0; i < params.batch_size; i++)
        {
            CHECK_NVJPEG(nvjpegJpegStreamParse(params.nvjpeg_handle, otherdecode_bitstreams[i],
                                               otherdecode_bitstreams_size[i], 0, 0,
                                               params.jpeg_streams[buffer_index]));

            CHECK_NVJPEG(
                nvjpegStateAttachPinnedBuffer(params.nvjpeg_decoupled_state, params.pinned_buffers[buffer_index]));

            CHECK_NVJPEG(nvjpegDecodeJpegHost(params.nvjpeg_handle, params.nvjpeg_decoder,
                                              params.nvjpeg_decoupled_state, params.nvjpeg_decode_params,
                                              params.jpeg_streams[buffer_index]));

            CHECK_CUDA(cudaStreamSynchronize(params.stream));

            CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, params.nvjpeg_decoder,
                                                          params.nvjpeg_decoupled_state,
                                                          params.jpeg_streams[buffer_index], params.stream));

            buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode to avoid an extra sync

            CHECK_NVJPEG(nvjpegDecodeJpegDevice(params.nvjpeg_handle, params.nvjpeg_decoder,
                                                params.nvjpeg_decoupled_state, &otherdecode_output[i], params.stream));
        }
    }
    CHECK_CUDA(cudaEventRecord(stopEvent, params.stream));

    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
    time = 0.001 * static_cast<double>(loopTime); // cudaEventElapsedTime returns milliseconds

    return EXIT_SUCCESS;
}

int write_images(std::vector<nvjpegImage_t> &iout, std::vector<int> &widths, std::vector<int> &heights,
                 decode_params_t &params, FileNames &filenames)
{
    for (int i = 0; i < params.batch_size; i++)
    {
        // Get the file name, without extension.
        // This will be used to rename the output file.
        size_t      position = filenames[i].rfind("/");
        std::string sFileName
            = (std::string::npos == position) ? filenames[i] : filenames[i].substr(position + 1, filenames[i].size());
        position  = sFileName.rfind(".");
        sFileName = (std::string::npos == position) ? sFileName : sFileName.substr(0, position);
        std::string fname(params.output_dir + "/" + sFileName + ".bmp");

        int err;
        if (params.fmt == NVJPEG_OUTPUT_RGB || params.fmt == NVJPEG_OUTPUT_BGR)
        {
            err = writeBMP(fname.c_str(), iout[i].channel[0], iout[i].pitch[0], iout[i].channel[1], iout[i].pitch[1],
                           iout[i].channel[2], iout[i].pitch[2], widths[i], heights[i]);
        }
        else if (params.fmt == NVJPEG_OUTPUT_RGBI || params.fmt == NVJPEG_OUTPUT_BGRI)
        {
            // Write BMP from interleaved data
            err = writeBMPi(fname.c_str(), iout[i].channel[0], iout[i].pitch[0], widths[i], heights[i]);
        }
        if (err)
        {
            std::cout << "Cannot write output file: " << fname << std::endl;
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

double process_images(FileNames &image_names, decode_params_t &params, double &total, std::vector<nvjpegImage_t> &iout,
                      uint8_t *gpuWorkspace, std::vector<int> &widths, std::vector<int> &heights)
{
    // vector for storing raw files and file lengths
    FileData            file_data(params.batch_size);
    std::vector<size_t> file_len(params.batch_size);
    FileNames           current_names(params.batch_size);
    // we wrap over image files to process total_images of files
    FileNames::iterator file_iter = image_names.begin();

    // stream for decoding
    CHECK_CUDA(cudaStreamCreateWithFlags(&params.stream, cudaStreamNonBlocking));

    int total_processed = 0;

    // output buffer sizes, for convenience
    std::vector<nvjpegImage_t> isz(params.batch_size);

    for (long unsigned int i = 0; i < iout.size(); i++)
    {
        for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
        {
            iout[i].channel[c] = NULL;
            iout[i].pitch[c]   = 0;
            isz[i].pitch[c]    = 0;
        }
    }

    double test_time = 0;
    int    warmup    = 0;
    while (total_processed < params.total_images)
    {
        if (read_next_batch(image_names, params.batch_size, file_iter, file_data, file_len, current_names))
            return EXIT_FAILURE;

        if (prepare_buffers(file_data, file_len, widths, heights, iout, isz, gpuWorkspace, current_names, params))
            return EXIT_FAILURE;

        double time;
        if (decode_images(file_data, file_len, iout, params, time))
            return EXIT_FAILURE;
        if (warmup < params.warmup)
        {
            warmup++;
        }
        else
        {
            total_processed += params.batch_size;
            test_time += time;
        }

        if (params.write_decoded)
            write_images(iout, widths, heights, params, current_names);
    }
    total = test_time;

    CHECK_CUDA(cudaStreamDestroy(params.stream));

    return EXIT_SUCCESS;
}

int NvDecode(std::string images_dir, int total_images, int batch_size, nvjpegOutputFormat_t outputFormat,
             uint8_t *gpuWorkspace)
{
    std::vector<nvjpegImage_t> iout;
    std::vector<int>           widths, heights;
    iout.resize(batch_size);
    widths.resize(batch_size);
    heights.resize(batch_size);

    decode_params_t params;
    params.input_dir           = images_dir;
    params.batch_size          = batch_size;
    params.warmup              = 0;
    params.fmt                 = outputFormat;
    params.write_decoded       = false;
    params.hw_decode_available = true;
    params.total_images        = (total_images / params.batch_size) * params.batch_size;
    params.output_dir          = "./";

    nvjpegDevAllocator_t    dev_allocator    = {&dev_malloc, &dev_free};
    nvjpegPinnedAllocator_t pinned_allocator = {&host_malloc, &host_free};
    nvjpegStatus_t          status = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, &dev_allocator, &pinned_allocator,
                                                    NVJPEG_FLAGS_DEFAULT, &params.nvjpeg_handle);
    if (status == NVJPEG_STATUS_ARCH_MISMATCH)
    {
        std::cout << "Hardware Decoder not supported. Falling back to default backend" << std::endl;
        CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &pinned_allocator, NVJPEG_FLAGS_DEFAULT,
                                    &params.nvjpeg_handle));
        params.hw_decode_available = false;
    }
    else
    {
        CHECK_NVJPEG(status);
    }

    CHECK_NVJPEG(nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));

    create_decoupled_api_handles(params);

    // read source images
    FileNames image_names;
    readInput(params.input_dir, image_names);

    double total;
    if (process_images(image_names, params, total, iout, gpuWorkspace, widths, heights))
        return EXIT_FAILURE;

#ifdef PROFILE_SAMPLE
    std::cout << "Total decoding time: " << total << " (s)" << std::endl;
    std::cout << "Avg decoding time per image: " << total / params.total_images << " (s)" << std::endl;
    std::cout << "Avg images per sec: " << params.total_images / total << std::endl;
    std::cout << "Avg decoding time per batch: "
              << total / ((params.total_images + params.batch_size - 1) / params.batch_size) << " (s)" << std::endl;
#endif

    destroy_decoupled_api_handles(params);

    CHECK_NVJPEG(nvjpegJpegStateDestroy(params.nvjpeg_state));
    CHECK_NVJPEG(nvjpegDestroy(params.nvjpeg_handle));

    return EXIT_SUCCESS;
}
