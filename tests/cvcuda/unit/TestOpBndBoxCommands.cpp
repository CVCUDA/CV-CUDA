/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Definitions.hpp"

#include <cvcuda/priv/OsdRenderer.hpp>
#include <nvcv/Tensor.hpp>

#include <array>
#include <cstring>
#include <vector>

namespace {

namespace osd = cvcuda::priv::osd;
using cvcuda::priv::NVCVBndBoxesImpl;
using cvcuda::priv::NVCVElement;
using cvcuda::priv::NVCVElementsImpl;

std::vector<osd::RectangleCommand> DrawRectangles(osd::cuOSDContext                           &context,
                                                  const std::vector<std::vector<NVCVBndBoxI>> &boxes, int width,
                                                  int height, bool useOSD)
{
    if (useOSD)
    {
        std::vector<std::vector<std::shared_ptr<NVCVElement>>> elements(boxes.size());
        for (size_t sample = 0; sample < boxes.size(); ++sample)
        {
            for (const auto &box : boxes[sample])
            {
                auto element = std::make_shared<NVCVElement>(NVCV_OSD_RECT, &box);
                elements[sample].push_back(element);
            }
        }
        NVCVElementsImpl batch(elements);
        osd::DrawElements(&context, width, height, &batch);
    }
    else
    {
        NVCVBndBoxesImpl batch(boxes);
        osd::DrawBndBoxes(&context, width, height, &batch);
        return context.bndbox_commands;
    }

    std::vector<osd::RectangleCommand> commands(context.commands.size());
    for (size_t commandIndex = 0; commandIndex < commands.size(); ++commandIndex)
    {
        auto &command = commands[commandIndex];
        std::memcpy(&command, static_cast<const osd::RectangleCommand *>(context.commands[commandIndex].get()),
                    sizeof(command));
        if (command.thickness == -1)
        {
            command.ax2 = command.ay2 = command.bx2 = command.by2 = 0;
            command.cx2 = command.cy2 = command.dx2 = command.dy2 = 0;
        }
    }
    return commands;
}

std::array<float, 8> OuterCorners(const osd::RectangleCommand &command)
{
    return {command.ax1, command.ay1, command.bx1, command.by1, command.cx1, command.cy1, command.dx1, command.dy1};
}

std::array<float, 8> InnerCorners(const osd::RectangleCommand &command)
{
    return {command.ax2, command.ay2, command.bx2, command.by2, command.cx2, command.cy2, command.dx2, command.dy2};
}

std::array<int, 4> Bounds(const osd::RectangleCommand &command)
{
    return {command.bounding_left, command.bounding_top, command.bounding_right, command.bounding_bottom};
}

std::array<unsigned char, 4> Color(const osd::RectangleCommand &command)
{
    return {command.c0, command.c1, command.c2, command.c3};
}

} // namespace

TEST(OpBndBoxCommands, FillPrecedesBorderWithIdenticalOSDGeometry)
{
    const std::vector<std::vector<NVCVBndBoxI>> boxes = {
        {},
        {{{2, 3, 8, 6}, 3, {11, 22, 33, 201}, {44, 55, 66, 123}}},
        {},
        {{{4, 5, 6, 7}, -1, {77, 88, 99, 0}, {101, 102, 103, 255}}},
    };
    for (bool useOSD : {false, true})
    {
        SCOPED_TRACE(useOSD);
        osd::cuOSDContext context;
        const auto        commands = DrawRectangles(context, boxes, 32, 32, useOSD);
        ASSERT_EQ(commands.size(), 3u);
        for (const auto &command : commands)
        {
            EXPECT_EQ(command.type, osd::CommandType::Rectangle);
            EXPECT_FALSE(command.interpolation);
            EXPECT_EQ(command.reserved, 0);
        }
        EXPECT_EQ(commands[0].batch_index, 1);
        EXPECT_EQ(commands[1].batch_index, 1);
        EXPECT_EQ(commands[2].batch_index, 3);
        EXPECT_EQ(commands[0].thickness, -1);
        EXPECT_EQ(commands[1].thickness, 3);
        EXPECT_EQ(commands[2].thickness, -1);
        EXPECT_EQ(Color(commands[0]), (std::array<unsigned char, 4>{44, 55, 66, 123}));
        EXPECT_EQ(Color(commands[1]), (std::array<unsigned char, 4>{11, 22, 33, 201}));
        EXPECT_EQ(Color(commands[2]), (std::array<unsigned char, 4>{77, 88, 99, 0}));
        EXPECT_EQ(OuterCorners(commands[0]), (std::array<float, 8>{2, 3, 2, 8, 9, 8, 9, 3}));
        EXPECT_EQ(OuterCorners(commands[1]), (std::array<float, 8>{0.5, 1.5, 0.5, 9.5, 10.5, 9.5, 10.5, 1.5}));
        EXPECT_EQ(InnerCorners(commands[1]), (std::array<float, 8>{3.5, 4.5, 3.5, 6.5, 7.5, 6.5, 7.5, 4.5}));
        EXPECT_EQ(OuterCorners(commands[2]), (std::array<float, 8>{4, 5, 4, 11, 9, 11, 9, 5}));
        EXPECT_EQ(Bounds(commands[0]), (std::array<int, 4>{2, 3, 9, 8}));
        EXPECT_EQ(Bounds(commands[1]), (std::array<int, 4>{0, 1, 11, 10}));
    }
}

TEST(OpBndBoxCommands, ClippingAndInvisibleBoxesMatchOSD)
{
    const std::vector<std::vector<NVCVBndBoxI>> boxes = {
        {
         {{2, 3, 0, 6}, 2, {1, 2, 3, 255}, {4, 5, 6, 255}},
         {{2, 3, 6, -1}, 2, {1, 2, 3, 255}, {4, 5, 6, 255}},
         {{2, 3, 1, 6}, 2, {1, 2, 3, 255}, {4, 5, 6, 255}},
         {{100, 3, 6, 6}, 2, {1, 2, 3, 255}, {4, 5, 6, 255}},
         {{2, 3, 6, 6}, 2, {1, 2, 3, 0}, {4, 5, 6, 0}},
         {{-4, -6, 8, 6}, 4, {1, 2, 3, 0}, {4, 5, 6, 99}},
         {{14, 12, 50, 50}, 0, {7, 8, 9, 127}, {0, 0, 0, 0}},
         }
    };
    for (bool useOSD : {false, true})
    {
        SCOPED_TRACE(useOSD);
        osd::cuOSDContext context;
        const auto        commands = DrawRectangles(context, boxes, 17, 15, useOSD);
        ASSERT_EQ(commands.size(), 2u);
        EXPECT_EQ(commands[0].thickness, -1);
        EXPECT_EQ(commands[1].thickness, 0);
        EXPECT_EQ(Color(commands[0]), (std::array<unsigned char, 4>{4, 5, 6, 99}));
        EXPECT_EQ(OuterCorners(commands[0]), (std::array<float, 8>{0, 0, 0, 5, 7, 5, 7, 0}));
        EXPECT_EQ(Bounds(commands[0]), (std::array<int, 4>{0, 0, 7, 5}));
        EXPECT_EQ(OuterCorners(commands[1]), (std::array<float, 8>{14, 12, 14, 14, 16, 14, 16, 12}));
        EXPECT_EQ(InnerCorners(commands[1]), OuterCorners(commands[1]));
    }
}

TEST(OpBndBoxCommands, AvoidsPerRectangleSharedAllocations)
{
    osd::cuOSDContext context;
    NVCVBndBoxesImpl  boxes({{{{2, 3, 8, 6}, 3, {11, 22, 33, 255}, {44, 55, 66, 123}}}});
    osd::DrawBndBoxes(&context, 32, 32, &boxes);
    EXPECT_TRUE(context.commands.empty());
    ASSERT_EQ(context.bndbox_commands.size(), 2u);
    const auto *storage  = context.bndbox_commands.data();
    const auto  capacity = context.bndbox_commands.capacity();
    context.bndbox_commands.clear();
    osd::DrawBndBoxes(&context, 32, 32, &boxes);
    EXPECT_EQ(context.bndbox_commands.data(), storage);
    EXPECT_EQ(context.bndbox_commands.capacity(), capacity);
}

TEST(OpBndBoxCommands, UploadBuffersRetainStorageAndRebuildBatchOffsets)
{
    constexpr int batchSize = 4;
    nvcv::Tensor  input(batchSize, {32, 32}, nvcv::FMT_RGB8);
    auto          inputData = input.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(inputData);
    auto inputAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inputData);
    ASSERT_TRUE(inputAccess);
    const auto   bufferSize = inputAccess->sampleStride() * batchSize;
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    osd::cuOSDContext            context;
    unsigned char               *commandStorage    = nullptr;
    int                         *offsetStorage     = nullptr;
    const osd::RectangleCommand *rectangleStorage  = nullptr;
    size_t                       rectangleCapacity = 0;
    const NVCVBndBoxI            box{
        { 2,  3,  8,   6},
        3,
        {11, 22, 33, 255},
        {44, 55, 66, 123}
    };
    for (int count : {8, 2, 0, -1, -2, 1, 8})
    {
        SCOPED_TRACE(count);
        std::vector<std::vector<NVCVBndBoxI>> boxes(batchSize);
        if (count >= 0)
        {
            boxes[1].assign(count, box);
            boxes[3].push_back(box);
        }
        else if (count == -2)
        {
            boxes[1].push_back({
                { 2,  3,  8, 6},
                3,
                {11, 22, 33, 0},
                {44, 55, 66, 0}
            });
            boxes[3].push_back({
                { 2,  3,  0,   6},
                3,
                {11, 22, 33, 255},
                {44, 55, 66, 255}
            });
        }
        NVCVBndBoxesImpl batch(boxes);
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(inputData->basePtr(), 0x35, bufferSize, stream));
        osd::DrawBndBoxes(&context, 32, 32, &batch);
        if (rectangleStorage == nullptr)
        {
            rectangleStorage  = context.bndbox_commands.data();
            rectangleCapacity = context.bndbox_commands.capacity();
        }
        EXPECT_EQ(context.bndbox_commands.data(), rectangleStorage);
        EXPECT_EQ(context.bndbox_commands.capacity(), rectangleCapacity);
        osd::Render(&context, *inputData, *inputData, batchSize, 3, 32, 32, stream);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        EXPECT_TRUE(context.commands.empty());
        EXPECT_TRUE(context.bndbox_commands.empty());
        ASSERT_TRUE(context.gpu_commands);
        ASSERT_TRUE(context.gpu_commands_offset);
        if (commandStorage == nullptr)
        {
            commandStorage = context.gpu_commands->host();
            offsetStorage  = context.gpu_commands_offset->host();
        }
        EXPECT_EQ(context.gpu_commands->host(), commandStorage);
        EXPECT_EQ(context.gpu_commands_offset->host(), offsetStorage);
        if (count < 0)
        {
            std::vector<unsigned char> pixels(bufferSize);
            ASSERT_EQ(cudaSuccess, cudaMemcpy(pixels.data(), inputData->basePtr(), bufferSize, cudaMemcpyDeviceToHost));
            EXPECT_EQ(pixels, std::vector<unsigned char>(bufferSize, 0x35));
            continue;
        }
        const int commandCount = (count + 1) * 2;
        EXPECT_EQ(context.gpu_commands->size(), commandCount * sizeof(osd::RectangleCommand));
        EXPECT_EQ(context.gpu_commands_offset->size(), commandCount + batchSize + 1);
        for (int command = 0; command < commandCount; ++command)
        {
            EXPECT_EQ(offsetStorage[command], command * sizeof(osd::RectangleCommand));
        }
        const std::vector<int> expectedBatchOffsets{0, 0, count * 2, count * 2, commandCount};
        EXPECT_EQ(std::vector<int>(offsetStorage + commandCount, offsetStorage + commandCount + batchSize + 1),
                  expectedBatchOffsets);
    }
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
