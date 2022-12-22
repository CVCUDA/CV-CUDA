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

#include <common/TestUtils.h>
#include <cuda_runtime_api.h>
#include <getopt.h>
#include <math.h>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <fstream>
#include <iostream>
#include <numeric>

/**
 * @brief Utiltities for Image classification
 *
 */

#define TOPN 5

/**
 * @brief Utility to get the class names from the labels file
 *
 **/
std::vector<std::string> getClassLabels(const std::string &labelsFilePath)
{
    std::vector<std::string> classes;
    std::ifstream            labelsFile(labelsFilePath);
    if (!labelsFile.good())
    {
        throw std::runtime_error("ERROR: Invalid Labels File Path\n");
    }
    std::string classLabel;
    while (std::getline(labelsFile, classLabel))
    {
        classes.push_back(classLabel);
    }
    return classes;
}

/**
 * @brief Display the TopN classification results
 *
 **/
void DisplayResults(std::vector<std::vector<float>> &scores, std::vector<std::vector<int>> &indices,
                    std::string labelPath)
{
    auto classes = getClassLabels(labelPath);
    for (int i = 0; i < scores.size(); i++)
    {
        printf("\nClassification results for batch %d \n", i);
        for (int j = 0; j < TOPN; j++)
        {
            auto index = indices[i][j];
            printf("Class : %s , Score : %f\n", classes[index].c_str(), scores[i][index]);
        }
    }
}

/**
 * @brief Utility docs
 *
 **/
void showUsage()
{
    std::cout << "usage: ./nvcv_classification_app -e <tensorrt engine path> -i <image file path or  image directory "
                 "path> -l <labels file path> -b <batch size>"
              << std::endl;
}

/**
 * @brief Utility to parse the command line arguments
 *
 **/
int ParseArgs(int argc, char *argv[], std::string &modelPath, std::string &imagePath, std::string &labelPath,
              uint32_t &batchSize)
{
    static struct option long_options[] = {
        {     "help",       no_argument, 0, 'h'},
        {   "engine", required_argument, 0, 'e'},
        {"labelPath", required_argument, 0, 'l'},
        {"imagePath", required_argument, 0, 'i'},
        {    "batch", required_argument, 0, 'b'},
        {          0,                 0, 0,   0}
    };

    int long_index = 0;
    int opt        = 0;
    while ((opt = getopt_long(argc, argv, "he:l:i:b:", long_options, &long_index)) != -1)
    {
        switch (opt)
        {
        case 'h':
            showUsage();
            return -1;
            break;
        case 'e':
            modelPath = optarg;
            break;
        case 'l':
            labelPath = optarg;
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
    std::ifstream modelFile(modelPath);
    if (!modelFile.good())
    {
        showUsage();
        std::cerr << "Model path '" + modelPath + "' does not exist\n";
        return -1;
    }
    std::ifstream imageFile(imagePath);
    if (!imageFile.good())
    {
        showUsage();
        std::cerr << "Image path '" + modelPath + "' does not exist\n";
        return -1;
    }
    std::ifstream labelFile(labelPath);
    if (!labelFile.good())
    {
        showUsage();
        std::cerr << "Label path '" + modelPath + "' does not exist\n";
        return -1;
    }
    return 0;
}
