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

#include "Definitions.hpp"

#include <common/ValueTests.hpp>
#include <cvcuda/OpPillowResize.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Rect.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>

#include <cmath>
#include <iostream>
#include <random>

namespace test = nvcv::test;
namespace t    = ::testing;

using Vecf  = std::vector<float>;
using uchar = unsigned char;

#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

template<typename T>
class TestMat
{
public:
    TestMat(int rows_, int cols_, int channels_, nvcv::DataKind dkind_)
        : rows(rows_)
        , cols(cols_)
        , channels(channels_)
        , dkind(dkind_)
    {
        data = std::vector<T>();
        data.resize(rows * cols * channels);
    }

    TestMat(int rows_, int cols_, int channels_, nvcv::DataKind dkind_, std::vector<T> &data_)
        : rows(rows_)
        , cols(cols_)
        , channels(channels_)
        , dkind(dkind_)
    {
        data = std::vector<T>();
        data = data_;
    }

    TestMat(const TestMat &test_mat, NVCVRectI roi)
    {
        rows     = roi.height;
        cols     = roi.width;
        channels = test_mat.channels;
        dkind    = test_mat.dkind;
        if (roi.height == test_mat.rows && roi.width == test_mat.cols)
        {
            data = std::vector<T>();
            data = test_mat.data;
        }
        else
        {
            data = std::vector<T>();
            data.resize(roi.width * roi.height * test_mat.channels);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        data[i * cols * channels + j * channels + c]
                            = test_mat.data[(i + roi.y) * test_mat.cols * channels + (j + roi.x) * channels + c];
                    }
                }
            }
        }
    }

    TestMat(nvcv::DataKind dkind_)
        : dkind(dkind_)
    {
        rows     = 0;
        cols     = 0;
        channels = 0;
        data     = std::vector<T>();
    }

    bool empty()
    {
        return data.empty();
    }

    void create(int rows_, int cols_, int ch_)
    {
        data = std::vector<T>();
        data.resize(rows_ * cols_ * ch_);
        rows     = rows_;
        cols     = cols_;
        channels = ch_;
    }

    TestMat<T> t() const
    {
        TestMat<T> new_test_mat(cols, rows, channels, dkind);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                for (int c = 0; c < channels; c++)
                {
                    new_test_mat.data[j * rows * channels + i * channels + c]
                        = data[i * cols * channels + j * channels + c];
                }
            }
        }
        return new_test_mat;
    }

    T get(int row, int col, int ch) const
    {
        return data[row * cols * channels + col * channels + ch];
    }

    void set(int row, int col, int ch, T val)
    {
        data[row * cols * channels + col * channels + ch] = val;
    }

    void print() const
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                std::cout << "i,j = " << i << "," << j;
                for (int c = 0; c < channels; c++)
                {
                    std::cout << " " << (int)get(i, j, c);
                }
                std::cout << std::endl;
            }
        }
    }

    int            rows;
    int            cols;
    int            channels;
    std::vector<T> data;
    nvcv::DataKind dkind;
};

struct Rect2f
{
    float x;      //!< x coordinate of the top-left corner
    float y;      //!< y coordinate of the top-left corner
    float width;  //!< width of the rectangle
    float height; //!< height of the rectangle
};

class PillowResizeCPU
{
protected:
    /**
     * \brief precision_bits 8 bits for result. Filter can have negative areas.
     * In one case the sum of the coefficients will be negative,
     * in the other it will be more than 1.0. That is why we need
     * two extra bits for overflow and int type.
     */
    static constexpr unsigned int precision_bits = 32 - 8 - 2;

    /**
     * \brief Filter Abstract class to handle the filters used by
     * the different interpolation methods.
     */
    class Filter
    {
    private:
        double _support; /** Support size (length of resampling filter). */

    public:
        /**
         * \brief Construct a new Filter object.
         *
         * \param[in] support Support size (length of resampling filter).
         */
        explicit Filter(double support)
            : _support{support} {};

        /**
         * \brief filter Apply filter.
         *
         * \param[in] x Input value.
         *
         * \return Processed value by the filter.
         */
        [[nodiscard]] virtual double filter(double x) const = 0;

        /**
         * \brief support Get support size.
         *
         * \return support size.
         */
        [[nodiscard]] double support() const
        {
            return _support;
        };
    };

    static constexpr float bilinear_filter_support = 1.;

    class BilinearFilter : public Filter
    {
    public:
        BilinearFilter()
            : Filter(bilinear_filter_support){};
        [[nodiscard]] double filter(double x) const override;
    };

    /**
     * \brief _lut Generate lookup table.
     * \reference https://joelfilho.com/blog/2020/compile_time_lookup_tables_in_cpp/
     *
     * \tparam Length Number of table elements.
     * \param[in] f Functor called to generate each elements in the table.
     *
     * \return An array of length Length with type deduced from Generator output.
     */
    template<size_t Length, typename Generator>
    static constexpr auto _lut(Generator &&f)
    {
        using content_type = decltype(f(size_t{0}));
        std::array<content_type, Length> arr{};
        for (size_t i = 0; i < Length; ++i)
        {
            arr[i] = f(i);
        }
        return arr;
    }

    /**
     * \brief _clip8_lut Clip lookup table.
     *
     * \tparam Length Number of table elements.
     * \tparam min_val Value of the starting element.
     */
    template<size_t Length, intmax_t min_val>
    inline static constexpr auto _clip8_lut = _lut<Length>(
        [](size_t n) -> uchar
        {
            intmax_t saturate_val = static_cast<intmax_t>(n) + min_val;
            if (saturate_val < 0)
            {
                return 0;
            }
            if (saturate_val > UCHAR_MAX)
            {
                return UCHAR_MAX;
            }
            return static_cast<uchar>(saturate_val);
        });

    /**
     * \brief _clip8 Optimized clip function.
     *
     * \param[in] in input value.
     *
     * \return Clipped value.
     */
    [[nodiscard]] static uchar _clip8(double in)
    {
        // Lookup table to speed up clip method.
        // Handles values from -640 to 639.
        const uchar *clip8_lookups = &_clip8_lut<1280, -640>[640]; // NOLINT
        // NOLINTNEXTLINE
        return clip8_lookups[static_cast<unsigned int>(in) >> precision_bits];
    }

    /**
     * \brief _roundUp Round function.
     * The output value will be cast to type T.
     *
     * \param[in] f Input value.
     *
     * \return Rounded value.
     */
    template<typename T>
    [[nodiscard]] static T _roundUp(double f)
    {
        return static_cast<T>(std::round(f));
    }

    /**
     * \brief _getPixelType Return the type of a matrix element.
     * If the matrix has multiple channels, the function returns the
     * type of the element without the channels.
     * For instance, if the type is CV_16SC3 the function return CV_16S.
     *
     * \param[in] img Input image.
     *
     * \return Matrix element type.
     */
    template<typename T>
    [[nodiscard]] static nvcv::DataKind _getPixelType(const TestMat<T> &img)
    {
        return img.dkind; // NOLINT
    }

    /**
     * \brief _precomputeCoeffs Compute 1D interpolation coefficients.
     * If you have an image (or a 2D matrix), call the method twice to compute
     * the coefficients for row and column either.
     * The coefficients are computed for each element in range [0, out_size).
     *
     * \param[in] in_size Input size (e.g. image width or height).
     * \param[in] in0 Input starting index.
     * \param[in] in1 Input last index.
     * \param[in] out_size Output size.
     * \param[in] filterp Pointer to a Filter object.
     * \param[out] bounds Bounds vector. A bound is a pair of xmin and xmax.
     * \param[out] kk Coefficients vector. To each elements corresponds a number of
     * coefficients returned by the function.
     *
     * \return Size of the filter coefficients.
     */
    [[nodiscard]] static int _precomputeCoeffs(int in_size, double in0, double in1, int out_size,
                                               const std::shared_ptr<Filter> &filterp, std::vector<int> &bounds,
                                               std::vector<double> &kk);

    /**
     * \brief _normalizeCoeffs8bpc Normalize coefficients for 8 bit per pixel matrix.
     *
     * \param[in] prekk Filter coefficients.
     *
     * \return Filter coefficients normalized.
     */
    [[nodiscard]] static std::vector<double> _normalizeCoeffs8bpc(const std::vector<double> &prekk);

    /**
     * \brief _resampleHorizontal Apply resample along the horizontal axis.
     * It calls the _resampleHorizontal with the correct pixel type using
     * the value returned by nvcv::Mat::type().
     *
     * \param[in, out] im_out Output resized matrix.
     *                        The matrix has to be previously initialized with right size.
     * \param[in] im_in Input matrix.
     * \param[in] offset Vertical offset (first used row in the source image).
     * \param[in] ksize Interpolation filter size.
     * \param[in] bounds Interpolation filter bounds (value of the min and max column
     *                   to be considered by the filter).
     * \param[in] prekk Interpolation filter coefficients.
     */
    template<typename T>
    static void _resampleHorizontal(TestMat<T> &im_out, const TestMat<T> &im_in, int offset, int ksize,
                                    const std::vector<int> &bounds, const std::vector<double> &prekk);

    /**
     * \brief _resampleVertical Apply resample along the vertical axis.
     * It calls the _resampleVertical with the correct pixel type using
     * the value returned by nvcv::Mat::type().
     *
     * \param[in, out] im_out Output resized matrix.
     *                        The matrix has to be previously initialized with right size.
     * \param[in] im_in Input matrix.
     * \param[in] ksize Interpolation filter size.
     * \param[in] bounds Interpolation filter bounds (value of the min and max row
     *                   to be considered by the filter).
     * \param[in] prekk Interpolation filter coefficients.
     */
    template<typename T>
    static void _resampleVertical(TestMat<T> &im_out, const TestMat<T> &im_in, int ksize,
                                  const std::vector<int> &bounds, const std::vector<double> &prekk);

    using preprocessCoefficientsFn = std::vector<double> (*)(const std::vector<double> &);

    template<typename T>
    using outMapFn = T (*)(double);

    /**
     * \brief _resampleHorizontal Apply resample along the horizontal axis.
     *
     * \param[in, out] im_out Output resized matrix.
     *                       The matrix has to be previously initialized with right size.
     * \param[in] im_in Input matrix.
     * \param[in] offset Vertical offset (first used row in the source image).
     * \param[in] ksize Interpolation filter size.
     * \param[in] bounds Interpolation filter bounds (index of min and max pixel
     *                   to be considered by the filter).
     * \param[in] prekk Interpolation filter coefficients.
     * \param[in] preprocessCoefficients Function used to process the filter coefficients.
     * \param[in] init_buffer Initial value of pixel buffer (default: 0.0).
     * \param[in] outMap Function used to convert the value of the pixel after
     *                   the interpolation into the output pixel.
     */
    template<typename T, typename T2>
    static void _resampleHorizontal(TestMat<T> &im_out, const TestMat<T> &im_in, int offset, int ksize,
                                    const std::vector<int> &bounds, const std::vector<double> &prekk,
                                    preprocessCoefficientsFn preprocessCoefficients = nullptr, double init_buffer = 0.,
                                    outMapFn<T2> outMap = nullptr);

    /**
     * \brief _resample Resize a matrix using the specified interpolation method.
     *
     * \param[in] im_in Input matrix.
     * \param[in] x_size Desidered output width.
     * \param[in] y_size Desidered output height.
     * \param[in] filter_p Pointer to the interpolation filter.
     * \param[in] rect Input region that has to be resized.
     *                 Region is defined as a vector of 4 point x0,y0,x1,y1.
     *
     * \return Resized matrix. The type of the matrix will be the same of im_in.
     */
    template<typename T>
    [[nodiscard]] static TestMat<T> _resample(const TestMat<T> &im_in, int x_size, int y_size,
                                              const std::shared_ptr<Filter> &filter_p,
                                              const Vecf                    &rect); //NVCVRectI

public:
    /**
     * \brief InterpolationMethods Interpolation methods.
     *
     * \see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters.
     */
    enum InterpolationMethods
    {
        INTERPOLATION_NEAREST  = 0,
        INTERPOLATION_BOX      = 4,
        INTERPOLATION_BILINEAR = 2,
        INTERPOLATION_HAMMING  = 5,
        INTERPOLATION_BICUBIC  = 3,
        INTERPOLATION_LANCZOS  = 1,
    };

    /**
     * \brief resize Porting of Pillow resize method.
     *
     * \param[in] src Input matrix that has to be processed.
     * \param[in] out_size Output matrix size.
     * \param[in] filter Interpolation method code, see InterpolationMethods.
     * \param[in] box Input roi. Only the elements inside the box will be resized.
     *
     * \return Resized matrix.
     *
     * \throw std::runtime_error In case the box is invalid, the interpolation filter
     *        or the input matrix type are not supported.
     */
    template<typename T>
    [[nodiscard]] static TestMat<T> resize(const TestMat<T> &src, const nvcv::Size2D &out_size, int filter,
                                           const Rect2f &box);

    /**
     * \brief resize Porting of Pillow resize method.
     *
     * \param[in] src Input matrix that has to be processed.
     * \param[in] out_size Output matrix size.
     * \param[in] filter Interpolation method code, see interpolation enum.
     *
     * \return Resized matrix.
     *
     * \throw std::runtime_error In case the box is invalid, the interpolation filter
     *        or the input matrix type are not supported.
     */
    template<typename T>
    [[nodiscard]] static TestMat<T> resize(const TestMat<T> &src, const nvcv::Size2D &out_size, int filter);

    static InterpolationMethods getInterpolationMethods(NVCVInterpolationType inter)
    {
        switch (inter)
        {
        case NVCV_INTERP_LINEAR:
            return PillowResizeCPU::InterpolationMethods::INTERPOLATION_BILINEAR;
        default:
            return PillowResizeCPU::InterpolationMethods::INTERPOLATION_BILINEAR;
        }
    }
};

template<typename T, typename T2>
void PillowResizeCPU::_resampleHorizontal(TestMat<T> &im_out, const TestMat<T> &im_in, int offset, int ksize,
                                          const std::vector<int> &bounds, const std::vector<double> &prekk,
                                          preprocessCoefficientsFn preprocessCoefficients, double init_buffer,
                                          outMapFn<T2> outMap)
{
    std::vector<double> kk(prekk.begin(), prekk.end());
    // Preprocess coefficients if needed.
    if (preprocessCoefficients != nullptr)
    {
        kk = preprocessCoefficients(kk);
    }

    for (int yy = 0; yy < im_out.rows; ++yy)
    {
        for (int xx = 0; xx < im_out.cols; ++xx)
        {
            int     xmin = bounds[xx * 2 + 0];
            int     xmax = bounds[xx * 2 + 1];
            double *k    = &kk[xx * ksize];
            for (int c = 0; c < im_in.channels; ++c)
            {
                double ss = init_buffer;
                for (int x = 0; x < xmax; ++x)
                {
                    // NOLINTNEXTLINE
                    ss += (T)im_in.get(yy + offset, x + xmin, c) * k[x];
                }
                // NOLINTNEXTLINE
                im_out.set(yy, xx, c, (T)(outMap == nullptr ? ss : outMap(ss)));
            }
        }
    }
}

double PillowResizeCPU::BilinearFilter::filter(double x) const
{
    if (x < 0.0)
    {
        x = -x;
    }
    if (x < 1.0)
    {
        return 1.0 - x;
    }
    return 0.0;
}

int PillowResizeCPU::_precomputeCoeffs(int in_size, double in0, double in1, int out_size,
                                       const std::shared_ptr<Filter> &filterp, std::vector<int> &bounds,
                                       std::vector<double> &kk)
{
    // Prepare for horizontal stretch.
    double scale       = 0;
    double filterscale = 0;
    filterscale = scale = static_cast<double>(in1 - in0) / out_size;
    if (filterscale < 1.0)
    {
        filterscale = 1.0;
    }

    // Determine support size (length of resampling filter).
    double support = filterp->support() * filterscale;

    // Maximum number of coeffs.
    int k_size = static_cast<int>(ceil(support)) * 2 + 1;

    // Check for overflow
    if (out_size > INT_MAX / (k_size * static_cast<int>(sizeof(double))))
    {
        throw std::runtime_error("Memory error");
    }

    // Coefficient buffer.
    kk.resize(out_size * k_size);

    // Bounds vector.
    bounds.resize(out_size * 2);

    int    x      = 0;
    int    xmin   = 0;
    int    xmax   = 0;
    double center = 0;
    double ww     = 0;
    double ss     = 0;

    const double half_pixel = 0.5;
    for (int xx = 0; xx < out_size; ++xx)
    {
        center = in0 + (xx + half_pixel) * scale;
        ww     = 0.0;
        ss     = 1.0 / filterscale;
        // Round the value.
        xmin = static_cast<int>(center - support + half_pixel);
        if (xmin < 0)
        {
            xmin = 0;
        }
        // Round the value.
        xmax = static_cast<int>(center + support + half_pixel);
        if (xmax > in_size)
        {
            xmax = in_size;
        }
        xmax -= xmin;
        double *k = &kk[xx * k_size];
        for (x = 0; x < xmax; ++x)
        {
            double w = filterp->filter((x + xmin - center + half_pixel) * ss);
            k[x]     = w; // NOLINT
            ww += w;
        }
        for (x = 0; x < xmax; ++x)
        {
            if (ww != 0.0)
            {
                k[x] /= ww; // NOLINT
            }
        }
        // Remaining values should stay empty if they are used despite of xmax.
        for (; x < k_size; ++x)
        {
            k[x] = 0; // NOLINT
        }
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    return k_size;
}

std::vector<double> PillowResizeCPU::_normalizeCoeffs8bpc(const std::vector<double> &prekk)
{
    std::vector<double> kk;
    kk.reserve(prekk.size());

    const double half_pixel = 0.5;
    for (const auto &k : prekk)
    {
        if (k < 0)
        {
            kk.emplace_back(static_cast<int>(-half_pixel + k * (1U << precision_bits)));
        }
        else
        {
            kk.emplace_back(static_cast<int>(half_pixel + k * (1U << precision_bits)));
        }
    }
    return kk;
}

template<typename T>
TestMat<T> PillowResizeCPU::resize(const TestMat<T> &src, const nvcv::Size2D &out_size, int filter)
{
    Rect2f box(0.F, 0.F, static_cast<float>(src.cols), static_cast<float>(src.rows));
    return resize(src, out_size, filter, box);
}

template<typename T>
TestMat<T> PillowResizeCPU::resize(const TestMat<T> &src, const nvcv::Size2D &out_size, int filter, const Rect2f &box)
{
    Vecf rect{box.x, box.y, box.x + box.width, box.y + box.height};

    int x_size = out_size.w;
    int y_size = out_size.h;
    if (x_size < 1 || y_size < 1)
    {
        throw std::runtime_error("Height and width must be > 0");
    }

    if (rect[0] < 0.F || rect[1] < 0.F)
    {
        throw std::runtime_error("Box offset can't be negative");
    }

    if (static_cast<int>(rect[2]) > src.cols || static_cast<int>(rect[3]) > src.rows)
    {
        throw std::runtime_error("Box can't exceed original image size");
    }

    if (box.width < 0 || box.height < 0)
    {
        throw std::runtime_error("Box can't be empty");
    }

    // If box's coordinates are int and box size matches requested size
    if (static_cast<int>(box.width) == x_size && static_cast<int>(box.height) == y_size)
    {
        NVCVRectI roi(static_cast<int>(box.x), static_cast<int>(box.y), static_cast<int>(box.width),
                      static_cast<int>(box.height));
        return TestMat(src, roi);
    }

    std::shared_ptr<Filter> filter_p;

    // Check filter.
    switch (filter)
    {
    case INTERPOLATION_BILINEAR:
        filter_p = std::make_shared<BilinearFilter>(BilinearFilter());
        break;
    default:
        throw std::runtime_error("unsupported resampling filter");
    }

    return PillowResizeCPU::_resample(src, x_size, y_size, filter_p, rect);
}

template<typename T>
TestMat<T> PillowResizeCPU::_resample(const TestMat<T> &im_in, int x_size, int y_size,
                                      const std::shared_ptr<Filter> &filter_p, const Vecf &rect)
{
    TestMat<T> im_out(im_in.dkind);
    TestMat<T> im_temp(im_in.dkind);

    std::vector<int>    bounds_horiz;
    std::vector<int>    bounds_vert;
    std::vector<double> kk_horiz;
    std::vector<double> kk_vert;

    bool need_horizontal = x_size != im_in.cols || (rect[0] != 0.0F) || static_cast<int>(rect[2]) != x_size;
    bool need_vertical   = y_size != im_in.rows || (rect[1] != 0.0F) || static_cast<int>(rect[3]) != y_size;

    // Compute horizontal filter coefficients.
    int ksize_horiz = _precomputeCoeffs(im_in.cols, rect[0], rect[2], x_size, filter_p, bounds_horiz, kk_horiz);

    // Compute vertical filter coefficients.
    int ksize_vert = _precomputeCoeffs(im_in.rows, rect[1], rect[3], y_size, filter_p, bounds_vert, kk_vert);

    // First used row in the source image.
    int ybox_first = bounds_vert[0];
    // Last used row in the source image.
    int ybox_last = bounds_vert[y_size * 2 - 2] + bounds_vert[y_size * 2 - 1];

    // Two-pass resize, horizontal pass.
    if (need_horizontal)
    {
        // Shift bounds for vertical pass.
        for (int i = 0; i < y_size; ++i)
        {
            bounds_vert[i * 2] -= ybox_first;
        }

        // Create destination image with desired ouput width and same input pixel type.
        im_temp.create(ybox_last - ybox_first, x_size, im_in.channels);
        if (!im_temp.empty())
        {
            _resampleHorizontal(im_temp, im_in, ybox_first, ksize_horiz, bounds_horiz, kk_horiz);
        }
        else
        {
            return TestMat<T>(im_in.dkind);
        }
        im_out = im_temp;
    }

    // Vertical pass.
    if (need_vertical)
    {
        // Create destination image with desired ouput size and same input pixel type.

        im_out.create(y_size, im_temp.cols, im_in.channels);
        if (!im_out.empty())
        {
            if (im_temp.empty())
            {
                im_temp = im_in;
            }
            // Input can be the original image or horizontally resampled one.
            _resampleVertical(im_out, im_temp, ksize_vert, bounds_vert, kk_vert);
        }
        else
        {
            return TestMat<T>(im_in.dkind);
        }
    }

    // None of the previous steps are performed, copying.
    if (im_out.empty())
    {
        im_out = im_in;
    }

    return im_out;
}

template<typename T>
void PillowResizeCPU::_resampleHorizontal(TestMat<T> &im_out, const TestMat<T> &im_in, int offset, int ksize,
                                          const std::vector<int> &bounds, const std::vector<double> &prekk)
{
    // Check pixel type.
    switch (_getPixelType(im_in))
    {
    case nvcv::DataKind::UNSIGNED:
        return _resampleHorizontal<T, unsigned char>(im_out, im_in, offset, ksize, bounds, prekk, _normalizeCoeffs8bpc,
                                                     (1U << (precision_bits - 1)), _clip8);
    case nvcv::DataKind::FLOAT:
        return _resampleHorizontal<T, float>(im_out, im_in, offset, ksize, bounds, prekk);
    default:
        throw std::runtime_error("Pixel kind not supported");
    }
}

template<typename T>
void PillowResizeCPU::_resampleVertical(TestMat<T> &im_out, const TestMat<T> &im_in, int ksize,
                                        const std::vector<int> &bounds, const std::vector<double> &prekk)
{
    im_out = im_out.t();
    _resampleHorizontal(im_out, im_in.t(), 0, ksize, bounds, prekk);
    im_out = im_out.t();
}

// clang-format off

NVCV_TEST_SUITE_P(OpPillowResize, test::ValueList<int, int, int, int, NVCVInterpolationType, int, nvcv::ImageFormat>
{
    // srcWidth, srcHeight, dstWidth, dstHeight,       interpolation, numberImages, imageFormat
    {        5,          5,        5,         5,  NVCV_INTERP_LINEAR,           1, nvcv::FMT_RGB8},
    {        5,          5,        5,         5,  NVCV_INTERP_LINEAR,           1, nvcv::FMT_RGBf32},
    {        10,        10,        5,         5,  NVCV_INTERP_LINEAR,           1, nvcv::FMT_RGB8},
    {        42,        40,       21,        20,  NVCV_INTERP_LINEAR,           1, nvcv::FMT_RGB8},
    {        21,        21,       42,        42,  NVCV_INTERP_LINEAR,           1, nvcv::FMT_RGB8},
    {        42,        42,       21,        21,  NVCV_INTERP_LINEAR,           4, nvcv::FMT_RGB8},
    {        21,        21,       42,        42,  NVCV_INTERP_LINEAR,           5, nvcv::FMT_RGB8},
    {        42,        42,       21,        21,  NVCV_INTERP_LINEAR,           6, nvcv::FMT_RGBf32},
    {        21,        21,       42,        42,  NVCV_INTERP_LINEAR,           7, nvcv::FMT_RGBf32},
});

// clang-format on

template<typename T>
void StartTest(int srcWidth, int srcHeight, int dstWidth, int dstHeight, NVCVInterpolationType interpolation,
               int numberOfImages, nvcv::ImageFormat fmt)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // Generate input
    nvcv::Tensor imgSrc(numberOfImages, {srcWidth, srcHeight}, fmt);

    const auto *srcData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgSrc.exportData());

    ASSERT_NE(nullptr, srcData);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<T>> srcVec(numberOfImages);
    int                         srcVecRowStride = srcWidth * fmt.planePixelStrideBytes(0);

    std::default_random_engine randEng;

    for (int i = 0; i < numberOfImages; ++i)
    {
        srcVec[i].resize(srcHeight * srcVecRowStride);

        std::default_random_engine             randEng{0};
        std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
        if (std::is_same<T, float>::value)
            std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return srcRand(randEng) / 255.0f; });
        else
            std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return srcRand(randEng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowStride(), srcVec[i].data(), srcVecRowStride,
                               srcVecRowStride, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    nvcv::Tensor imgDst(numberOfImages, {dstWidth, dstHeight}, fmt);

    cvcuda::PillowResize pillowResizeOp(nvcv::Size2D{std::max(srcWidth, dstWidth), std::max(srcHeight, dstHeight)},
                                        numberOfImages, fmt);
    EXPECT_NO_THROW(pillowResizeOp(stream, imgSrc, imgDst, interpolation));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    const auto *dstData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgDst.exportData());
    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    int dstVecRowStride = dstWidth * fmt.planePixelStrideBytes(0);
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<T> testVec(dstHeight * dstVecRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstVecRowStride, dstAccess->sampleData(i), dstAccess->rowStride(),
                               dstVecRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        TestMat<T> test_in(srcHeight, srcWidth, fmt.planePixelStrideBytes(0), nvcv::DataKind::UNSIGNED, srcVec[i]);
        PillowResizeCPU::InterpolationMethods inter = PillowResizeCPU::getInterpolationMethods(interpolation);
        TestMat<T> test_out = PillowResizeCPU::resize(test_in, nvcv::Size2D(dstWidth, dstHeight), inter);

        // maximum absolute error
        std::vector<int> mae(testVec.size());
        for (size_t i = 0; i < mae.size(); ++i)
        {
            mae[i] = abs(static_cast<int>((test_out.data)[i]) - static_cast<int>(testVec[i]));
        }

        int maeThreshold = 2;

        EXPECT_THAT(mae, t::Each(t::Le(maeThreshold)));
    }
}

TEST_P(OpPillowResize, tensor_correct_output)
{
    int                   srcWidth       = GetParamValue<0>();
    int                   srcHeight      = GetParamValue<1>();
    int                   dstWidth       = GetParamValue<2>();
    int                   dstHeight      = GetParamValue<3>();
    NVCVInterpolationType interpolation  = GetParamValue<4>();
    int                   numberOfImages = GetParamValue<5>();
    nvcv::ImageFormat     fmt            = GetParamValue<6>();
    if (nvcv::FMT_RGB8 == fmt || nvcv::FMT_RGBA8 == fmt)
        StartTest<uint8_t>(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberOfImages, fmt);
    else if (nvcv::FMT_RGBf32 == fmt || nvcv::FMT_RGBAf32 == fmt)
        StartTest<float>(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberOfImages, fmt);
}

template<typename T>
void StartVarShapeTest(int srcWidthBase, int srcHeightBase, int dstWidthBase, int dstHeightBase,
                       NVCVInterpolationType interpolation, int numberOfImages, nvcv::ImageFormat fmt)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // Create input and output
    std::default_random_engine         randEng;
    std::uniform_int_distribution<int> rndSrcWidth(srcWidthBase * 0.8, srcWidthBase * 1.1);
    std::uniform_int_distribution<int> rndSrcHeight(srcHeightBase * 0.8, srcHeightBase * 1.1);

    std::uniform_int_distribution<int> rndDstWidth(dstWidthBase * 0.8, dstWidthBase * 1.1);
    std::uniform_int_distribution<int> rndDstHeight(dstHeightBase * 0.8, dstHeightBase * 1.1);

    std::vector<std::unique_ptr<nvcv::Image>> imgSrc, imgDst;
    for (int i = 0; i < numberOfImages; ++i)
    {
        imgSrc.emplace_back(
            std::make_unique<nvcv::Image>(nvcv::Size2D{rndSrcWidth(randEng), rndSrcHeight(randEng)}, fmt));
        imgDst.emplace_back(
            std::make_unique<nvcv::Image>(nvcv::Size2D{rndDstWidth(randEng), rndDstHeight(randEng)}, fmt));
    }

    nvcv::ImageBatchVarShape batchSrc(numberOfImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numberOfImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    std::vector<std::vector<T>> srcVec(numberOfImages);
    std::vector<int>            srcVecRowStride(numberOfImages);

    // Populate input
    for (int i = 0; i < numberOfImages; ++i)
    {
        const auto *srcData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        assert(srcData->numPlanes() == 1);

        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        int srcRowStride = srcWidth * fmt.planePixelStrideBytes(0);

        srcVecRowStride[i] = srcRowStride;

        std::default_random_engine             randEng{0};
        std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};

        srcVec[i].resize(srcHeight * srcRowStride);
        if (std::is_same<T, float>::value)
            std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return srcRand(randEng) / 255.0f; });
        else
            std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return srcRand(randEng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec[i].data(), srcRowStride,
                               srcRowStride, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    nvcv::Size2D maxSrcSize = batchSrc.maxSize();
    nvcv::Size2D maxDstSize = batchDst.maxSize();

    // Generate test result
    cvcuda::PillowResize pillowResizeOp(
        nvcv::Size2D{std::max(maxSrcSize.w, maxDstSize.w), std::max(maxSrcSize.h, maxDstSize.h)}, numberOfImages, fmt);
    EXPECT_NO_THROW(pillowResizeOp(stream, batchSrc, batchDst, interpolation));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto *srcData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        assert(srcData->numPlanes() == 1);
        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        const auto *dstData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgDst[i]->exportData());
        assert(dstData->numPlanes() == 1);

        int dstWidth  = dstData->plane(0).width;
        int dstHeight = dstData->plane(0).height;

        int dstRowStride = dstWidth * fmt.planePixelStrideBytes(0);

        std::vector<T> testVec(dstHeight * dstRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        TestMat<T> test_in(srcHeight, srcWidth, fmt.planePixelStrideBytes(0), nvcv::DataKind::UNSIGNED, srcVec[i]);
        PillowResizeCPU::InterpolationMethods inter = PillowResizeCPU::getInterpolationMethods(interpolation);
        TestMat<T> test_out = PillowResizeCPU::resize(test_in, nvcv::Size2D(dstWidth, dstHeight), inter);

        // maximum absolute error
        std::vector<int> mae(testVec.size());
        for (size_t i = 0; i < mae.size(); ++i)
        {
            mae[i] = abs(static_cast<int>((test_out.data)[i]) - static_cast<int>(testVec[i]));
        }

        int maeThreshold = 2;

        EXPECT_THAT(mae, t::Each(t::Le(maeThreshold)));
    }
}

TEST_P(OpPillowResize, varshape_correct_output)
{
    int                   srcWidth       = GetParamValue<0>();
    int                   srcHeight      = GetParamValue<1>();
    int                   dstWidth       = GetParamValue<2>();
    int                   dstHeight      = GetParamValue<3>();
    NVCVInterpolationType interpolation  = GetParamValue<4>();
    int                   numberOfImages = GetParamValue<5>();
    nvcv::ImageFormat     fmt            = GetParamValue<6>();
    if (nvcv::FMT_RGB8 == fmt || nvcv::FMT_RGBA8 == fmt)
        StartVarShapeTest<uint8_t>(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberOfImages, fmt);
    else if (nvcv::FMT_RGBf32 == fmt || nvcv::FMT_RGBAf32 == fmt)
        StartVarShapeTest<float>(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberOfImages, fmt);
}
