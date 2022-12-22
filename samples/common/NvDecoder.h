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

#ifndef NVCV_NVDECODER_HPP
#define NVCV_NVDECODER_HPP

#include <cuda_runtime_api.h>
#include <dirent.h>
#include <nvjpeg.h>
#include <string.h> // strcmpi
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                                                          \
    {                                                                                                             \
        cudaError_t _e = (call);                                                                                  \
        if (_e != cudaSuccess)                                                                                    \
        {                                                                                                         \
            std::cout << "CUDA Runtime failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                                                              \
        }                                                                                                         \
    }

#define CHECK_NVJPEG(call)                                                                                  \
    {                                                                                                       \
        nvjpegStatus_t _e = (call);                                                                         \
        if (_e != NVJPEG_STATUS_SUCCESS)                                                                    \
        {                                                                                                   \
            std::cout << "NVJPEG failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                                                        \
        }                                                                                                   \
    }

int dev_malloc(void **p, size_t s)
{
    return (int)cudaMalloc(p, s);
}

int dev_free(void *p)
{
    return (int)cudaFree(p);
}

int host_malloc(void **p, size_t s, unsigned int f)
{
    return (int)cudaHostAlloc(p, s, f);
}

int host_free(void *p)
{
    return (int)cudaFreeHost(p);
}

typedef std::vector<std::string>       FileNames;
typedef std::vector<std::vector<char>> FileData;

struct decode_params_t
{
    std::string input_dir;
    int         batch_size;
    int         total_images;
    int         dev;
    int         warmup;

    nvjpegJpegState_t nvjpeg_state;
    nvjpegHandle_t    nvjpeg_handle;
    cudaStream_t      stream;

    // used with decoupled API
    nvjpegJpegState_t    nvjpeg_decoupled_state;
    nvjpegBufferPinned_t pinned_buffers[2]; // 2 buffers for pipelining
    nvjpegBufferDevice_t device_buffer;
    nvjpegJpegStream_t   jpeg_streams[2]; //  2 streams for pipelining
    nvjpegDecodeParams_t nvjpeg_decode_params;
    nvjpegJpegDecoder_t  nvjpeg_decoder;

    nvjpegOutputFormat_t fmt;
    bool                 write_decoded;
    std::string          output_dir;

    bool hw_decode_available;
};

int read_next_batch(FileNames &image_names, int batch_size, FileNames::iterator &cur_iter, FileData &raw_data,
                    std::vector<size_t> &raw_len, FileNames &current_names)
{
    int counter = 0;

    while (counter < batch_size)
    {
        if (cur_iter == image_names.end())
        {
            std::cerr << "Image list is too short to fill the batch, adding files "
                         "from the beginning of the image list"
                      << std::endl;
            cur_iter = image_names.begin();
        }

        if (image_names.size() == 0)
        {
            std::cerr << "No valid images left in the input list, exit" << std::endl;
            return EXIT_FAILURE;
        }

        // Read an image from disk.
        std::ifstream input(cur_iter->c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        if (!(input.is_open()))
        {
            std::cerr << "Cannot open image: " << *cur_iter << ", removing it from image list" << std::endl;
            image_names.erase(cur_iter);
            continue;
        }

        // Get the size
        long unsigned int file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if (raw_data[counter].size() < file_size)
        {
            raw_data[counter].resize(file_size);
        }
        if (!input.read(raw_data[counter].data(), file_size))
        {
            std::cerr << "Cannot read from file: " << *cur_iter << ", removing it from image list" << std::endl;
            image_names.erase(cur_iter);
            continue;
        }
        raw_len[counter] = file_size;

        current_names[counter] = *cur_iter;

        counter++;
        cur_iter++;
    }
    return EXIT_SUCCESS;
}

// prepare buffers for RGBi output format
int prepare_buffers(FileData &file_data, std::vector<size_t> &file_len, std::vector<int> &img_width,
                    std::vector<int> &img_height, std::vector<nvjpegImage_t> &ibuf, std::vector<nvjpegImage_t> &isz,
                    uint8_t *gpuWorkspace, FileNames &current_names, decode_params_t &params)
{
    int                       widths[NVJPEG_MAX_COMPONENT];
    int                       heights[NVJPEG_MAX_COMPONENT];
    int                       channels;
    nvjpegChromaSubsampling_t subsampling;

    for (long unsigned int i = 0; i < file_data.size(); i++)
    {
        CHECK_NVJPEG(nvjpegGetImageInfo(params.nvjpeg_handle, (unsigned char *)file_data[i].data(), file_len[i],
                                        &channels, &subsampling, widths, heights));

        img_width[i]  = widths[0];
        img_height[i] = heights[0];
        std::cout << "Processing: " << current_names[i] << std::endl;
#ifdef NVJPEG_DEBUG
        std::cout << "Image is " << channels << " channels." << std::endl;
        for (int c = 0; c < channels; c++)
        {
            std::cout << "Channel #" << c << " size: " << widths[c] << " x " << heights[c] << std::endl;
        }
        switch (subsampling)
        {
        case NVJPEG_CSS_444:
            std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
            break;
        case NVJPEG_CSS_440:
            std::cout << "YUV 4:4:0 chroma subsampling" << std::endl;
            break;
        case NVJPEG_CSS_422:
            std::cout << "YUV 4:2:2 chroma subsampling" << std::endl;
            break;
        case NVJPEG_CSS_420:
            std::cout << "YUV 4:2:0 chroma subsampling" << std::endl;
            break;
        case NVJPEG_CSS_411:
            std::cout << "YUV 4:1:1 chroma subsampling" << std::endl;
            break;
        case NVJPEG_CSS_410:
            std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
            break;
        case NVJPEG_CSS_GRAY:
            std::cout << "Grayscale JPEG " << std::endl;
            break;
        case NVJPEG_CSS_410V:
            std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
            break;
        case NVJPEG_CSS_UNKNOWN:
            std::cout << "Unknown chroma subsampling" << std::endl;
            return EXIT_FAILURE;
        }
#endif

        int mul = 1;
        // in the case of interleaved RGB output, write only to single channel, but
        // 3 samples at once
        if (params.fmt == NVJPEG_OUTPUT_RGBI || params.fmt == NVJPEG_OUTPUT_BGRI)
        {
            channels = 1;
            mul      = 3;
        }
        // in the case of rgb create 3 buffers with sizes of original image
        else if (params.fmt == NVJPEG_OUTPUT_RGB || params.fmt == NVJPEG_OUTPUT_BGR)
        {
            channels  = 3;
            widths[1] = widths[2] = widths[0];
            heights[1] = heights[2] = heights[0];
        }

        // realloc output buffer if required
        for (int c = 0; c < channels; c++)
        {
            int    aw        = mul * widths[c];
            int    ah        = heights[c];
            size_t sz        = aw * ah;
            ibuf[i].pitch[c] = aw;
            if (sz > isz[i].pitch[c])
            {
                ibuf[i].channel[c] = gpuWorkspace;
                gpuWorkspace       = gpuWorkspace + sz;
                isz[i].pitch[c]    = sz;
            }
        }
    }
    return EXIT_SUCCESS;
}

void create_decoupled_api_handles(decode_params_t &params)
{
    CHECK_NVJPEG(nvjpegDecoderCreate(params.nvjpeg_handle, NVJPEG_BACKEND_DEFAULT, &params.nvjpeg_decoder));
    CHECK_NVJPEG(nvjpegDecoderStateCreate(params.nvjpeg_handle, params.nvjpeg_decoder, &params.nvjpeg_decoupled_state));

    CHECK_NVJPEG(nvjpegBufferPinnedCreate(params.nvjpeg_handle, NULL, &params.pinned_buffers[0]));
    CHECK_NVJPEG(nvjpegBufferPinnedCreate(params.nvjpeg_handle, NULL, &params.pinned_buffers[1]));
    CHECK_NVJPEG(nvjpegBufferDeviceCreate(params.nvjpeg_handle, NULL, &params.device_buffer));

    CHECK_NVJPEG(nvjpegJpegStreamCreate(params.nvjpeg_handle, &params.jpeg_streams[0]));
    CHECK_NVJPEG(nvjpegJpegStreamCreate(params.nvjpeg_handle, &params.jpeg_streams[1]));

    CHECK_NVJPEG(nvjpegDecodeParamsCreate(params.nvjpeg_handle, &params.nvjpeg_decode_params));
}

void destroy_decoupled_api_handles(decode_params_t &params)
{
    CHECK_NVJPEG(nvjpegDecodeParamsDestroy(params.nvjpeg_decode_params));
    CHECK_NVJPEG(nvjpegJpegStreamDestroy(params.jpeg_streams[0]));
    CHECK_NVJPEG(nvjpegJpegStreamDestroy(params.jpeg_streams[1]));
    CHECK_NVJPEG(nvjpegBufferPinnedDestroy(params.pinned_buffers[0]));
    CHECK_NVJPEG(nvjpegBufferPinnedDestroy(params.pinned_buffers[1]));
    CHECK_NVJPEG(nvjpegBufferDeviceDestroy(params.device_buffer));
    CHECK_NVJPEG(nvjpegJpegStateDestroy(params.nvjpeg_decoupled_state));
    CHECK_NVJPEG(nvjpegDecoderDestroy(params.nvjpeg_decoder));
}

void release_buffers(std::vector<nvjpegImage_t> &ibuf)
{
    for (long unsigned int i = 0; i < ibuf.size(); i++)
    {
        for (auto c = 0; c < NVJPEG_MAX_COMPONENT; c++)
            if (ibuf[i].channel[c])
                CHECK_CUDA(cudaFree(ibuf[i].channel[c]));
    }
}

// *****************************************************************************
// reading input directory to file list
// -----------------------------------------------------------------------------
int readInput(const std::string &sInputPath, std::vector<std::string> &filelist)
{
    int         error_code = 1;
    struct stat s;

    if (stat(sInputPath.c_str(), &s) == 0)
    {
        if (s.st_mode & S_IFREG)
        {
            filelist.push_back(sInputPath);
        }
        else if (s.st_mode & S_IFDIR)
        {
            // processing each file in directory
            DIR           *dir_handle;
            struct dirent *dir;
            dir_handle = opendir(sInputPath.c_str());
            std::vector<std::string> filenames;
            if (dir_handle)
            {
                error_code = 0;
                while ((dir = readdir(dir_handle)) != NULL)
                {
                    if (dir->d_type == DT_REG)
                    {
                        std::string sFileName = sInputPath + dir->d_name;
                        filelist.push_back(sFileName);
                    }
                    else if (dir->d_type == DT_DIR)
                    {
                        std::string sname = dir->d_name;
                        if (sname != "." && sname != "..")
                        {
                            readInput(sInputPath + sname + "/", filelist);
                        }
                    }
                }
                closedir(dir_handle);
            }
            else
            {
                std::cout << "Cannot open input directory: " << sInputPath << std::endl;
                return error_code;
            }
        }
        else
        {
            std::cout << "Cannot open input: " << sInputPath << std::endl;
            return error_code;
        }
    }
    else
    {
        std::cout << "Cannot find input path " << sInputPath << std::endl;
        return error_code;
    }

    return 0;
}

// *****************************************************************************
// check for inputDirExists
// -----------------------------------------------------------------------------
int inputDirExists(const char *pathname)
{
    struct stat info;
    if (stat(pathname, &info) != 0)
    {
        return 0; // Directory does not exists
    }
    else if (info.st_mode & S_IFDIR)
    {
        // is a directory
        return 1;
    }
    else
    {
        // is not a directory
        return 0;
    }
}

// *****************************************************************************
// check for getInputDir
// -----------------------------------------------------------------------------
int getInputDir(std::string &input_dir, const char *executable_path)
{
    int found = 0;
    if (executable_path != 0)
    {
        std::string executable_name = std::string(executable_path);
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        // Windows path delimiter
        size_t delimiter_pos = executable_name.find_last_of('\\');
        executable_name.erase(0, delimiter_pos + 1);

        if (executable_name.rfind(".exe") != std::string::npos)
        {
            // we strip .exe, only if the .exe is found
            executable_name.resize(executable_name.size() - 4);
        }
#else
        // Linux & OSX path delimiter
        size_t delimiter_pos = executable_name.find_last_of('/');
        executable_name.erase(0, delimiter_pos + 1);
#endif

        // Search in default paths for input images.
        std::string pathname     = "";
        const char *searchPath[] = {"./images"};

        for (unsigned int i = 0; i < sizeof(searchPath) / sizeof(char *); ++i)
        {
            std::string pathname(searchPath[i]);
            size_t      executable_name_pos = pathname.find("<executable_name>");

            // If there is executable_name variable in the searchPath
            // replace it with the value
            if (executable_name_pos != std::string::npos)
            {
                pathname.replace(executable_name_pos, strlen("<executable_name>"), executable_name);
            }

            if (inputDirExists(pathname.c_str()))
            {
                input_dir = pathname + "/";
                found     = 1;
                break;
            }
        }
    }
    return found;
}

// write bmp, input - RGB, device
int writeBMP(const char *filename, const unsigned char *d_chanR, int pitchR, const unsigned char *d_chanG, int pitchG,
             const unsigned char *d_chanB, int pitchB, int width, int height)
{
    unsigned int headers[13];
    FILE        *outfile;
    int          extrabytes;
    int          paddedsize;
    int          x;
    int          y;
    int          n;
    int          red, green, blue;

    std::vector<unsigned char> vchanR(height * width);
    std::vector<unsigned char> vchanG(height * width);
    std::vector<unsigned char> vchanB(height * width);
    unsigned char             *chanR = vchanR.data();
    unsigned char             *chanG = vchanG.data();
    unsigned char             *chanB = vchanB.data();
    CHECK_CUDA(cudaMemcpy2D(chanR, (size_t)width, d_chanR, (size_t)pitchR, width, height, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy2D(chanG, (size_t)width, d_chanG, (size_t)pitchR, width, height, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy2D(chanB, (size_t)width, d_chanB, (size_t)pitchR, width, height, cudaMemcpyDeviceToHost));

    extrabytes = 4 - ((width * 3) % 4); // How many bytes of padding to add to each
    // horizontal line - the size of which must
    // be a multiple of 4 bytes.
    if (extrabytes == 4)
        extrabytes = 0;

    paddedsize = ((width * 3) + extrabytes) * height;

    // Headers...
    // Note that the "BM" identifier in bytes 0 and 1 is NOT included in these
    // "headers".

    headers[0] = paddedsize + 54; // bfSize (whole file size)
    headers[1] = 0;               // bfReserved (both)
    headers[2] = 54;              // bfOffbits
    headers[3] = 40;              // biSize
    headers[4] = width;           // biWidth
    headers[5] = height;          // biHeight

    // Would have biPlanes and biBitCount in position 6, but they're shorts.
    // It's easier to write them out separately (see below) than pretend
    // they're a single int, especially with endian issues...

    headers[7]  = 0;          // biCompression
    headers[8]  = paddedsize; // biSizeImage
    headers[9]  = 0;          // biXPelsPerMeter
    headers[10] = 0;          // biYPelsPerMeter
    headers[11] = 0;          // biClrUsed
    headers[12] = 0;          // biClrImportant

    if (!(outfile = fopen(filename, "wb")))
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return 1;
    }

    //
    // Headers begin...
    // When printing ints and shorts, we write out 1 character at a time to avoid
    // endian issues.
    //
    fprintf(outfile, "BM");

    for (n = 0; n <= 5; n++)
    {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    // These next 4 characters are for the biPlanes and biBitCount fields.

    fprintf(outfile, "%c", 1);
    fprintf(outfile, "%c", 0);
    fprintf(outfile, "%c", 24);
    fprintf(outfile, "%c", 0);

    for (n = 7; n <= 12; n++)
    {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    //
    // Headers done, now write the data...
    //

    for (y = height - 1; y >= 0; y--) // BMP image format is written from bottom to top...
    {
        for (x = 0; x <= width - 1; x++)
        {
            red   = chanR[y * width + x];
            green = chanG[y * width + x];
            blue  = chanB[y * width + x];

            if (red > 255)
                red = 255;
            if (red < 0)
                red = 0;
            if (green > 255)
                green = 255;
            if (green < 0)
                green = 0;
            if (blue > 255)
                blue = 255;
            if (blue < 0)
                blue = 0;
            // Also, it's written in (b,g,r) format...

            fprintf(outfile, "%c", blue);
            fprintf(outfile, "%c", green);
            fprintf(outfile, "%c", red);
        }
        if (extrabytes) // See above - BMP lines must be of lengths divisible by 4.
        {
            for (n = 1; n <= extrabytes; n++)
            {
                fprintf(outfile, "%c", 0);
            }
        }
    }

    fclose(outfile);
    return 0;
}

// write bmp, input - RGB, device
int writeBMPi(const char *filename, const unsigned char *d_RGB, int pitch, int width, int height)
{
    unsigned int headers[13];
    FILE        *outfile;
    int          extrabytes;
    int          paddedsize;
    int          x;
    int          y;
    int          n;
    int          red, green, blue;

    printf("Writing to %s %d %d %d\n", filename, pitch, width, height);
    std::vector<unsigned char> vchanRGB(height * width * 3);
    unsigned char             *chanRGB = vchanRGB.data();
    CHECK_CUDA(
        cudaMemcpy2D(chanRGB, (size_t)width * 3, d_RGB, (size_t)pitch, width * 3, height, cudaMemcpyDeviceToHost));

    extrabytes = 4 - ((width * 3) % 4); // How many bytes of padding to add to each
    // horizontal line - the size of which must
    // be a multiple of 4 bytes.
    if (extrabytes == 4)
        extrabytes = 0;

    paddedsize = ((width * 3) + extrabytes) * height;

    // Headers...
    // Note that the "BM" identifier in bytes 0 and 1 is NOT included in these
    // "headers".
    headers[0] = paddedsize + 54; // bfSize (whole file size)
    headers[1] = 0;               // bfReserved (both)
    headers[2] = 54;              // bfOffbits
    headers[3] = 40;              // biSize
    headers[4] = width;           // biWidth
    headers[5] = height;          // biHeight

    // Would have biPlanes and biBitCount in position 6, but they're shorts.
    // It's easier to write them out separately (see below) than pretend
    // they're a single int, especially with endian issues...

    headers[7]  = 0;          // biCompression
    headers[8]  = paddedsize; // biSizeImage
    headers[9]  = 0;          // biXPelsPerMeter
    headers[10] = 0;          // biYPelsPerMeter
    headers[11] = 0;          // biClrUsed
    headers[12] = 0;          // biClrImportant

    if (!(outfile = fopen(filename, "wb")))
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return 1;
    }

    //
    // Headers begin...
    // When printing ints and shorts, we write out 1 character at a time to avoid
    // endian issues.
    //

    fprintf(outfile, "BM");

    for (n = 0; n <= 5; n++)
    {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    // These next 4 characters are for the biPlanes and biBitCount fields.

    fprintf(outfile, "%c", 1);
    fprintf(outfile, "%c", 0);
    fprintf(outfile, "%c", 24);
    fprintf(outfile, "%c", 0);

    for (n = 7; n <= 12; n++)
    {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    //
    // Headers done, now write the data...
    //
    for (y = height - 1; y >= 0; y--) // BMP image format is written from bottom to top...
    {
        for (x = 0; x <= width - 1; x++)
        {
            red   = chanRGB[(y * width + x) * 3];
            green = chanRGB[(y * width + x) * 3 + 1];
            blue  = chanRGB[(y * width + x) * 3 + 2];

            if (red > 255)
                red = 255;
            if (red < 0)
                red = 0;
            if (green > 255)
                green = 255;
            if (green < 0)
                green = 0;
            if (blue > 255)
                blue = 255;
            if (blue < 0)
                blue = 0;
            // Also, it's written in (b,g,r) format...

            fprintf(outfile, "%c", blue);
            fprintf(outfile, "%c", green);
            fprintf(outfile, "%c", red);
        }
        if (extrabytes) // See above - BMP lines must be of lengths divisible by 4.
        {
            for (n = 1; n <= extrabytes; n++)
            {
                fprintf(outfile, "%c", 0);
            }
        }
    }

    fclose(outfile);
    return 0;
}

// *****************************************************************************
// parse parameters
// -----------------------------------------------------------------------------
int findParamIndex(const char **argv, int argc, const char *parm)
{
    int count = 0;
    int index = -1;

    for (int i = 0; i < argc; i++)
    {
        if (strncmp(argv[i], parm, 100) == 0)
        {
            index = i;
            count++;
        }
    }

    if (count == 0 || count == 1)
    {
        return index;
    }
    else
    {
        std::cout << "Error, parameter " << parm << " has been specified more than once, exiting\n" << std::endl;
        return -1;
    }

    return -1;
}

int NvDecode(std::string images_dir, int total_images, int batch_size, nvjpegOutputFormat_t outputFormat,
             uint8_t *gpuInput);

#endif
