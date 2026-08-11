/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "PlanarParityUtils.hpp"

#include <common/ValueTests.hpp>
#include <cvcuda/OpPillowResize.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Rect.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace test = nvcv::test;
namespace t    = ::testing;

using Vecf  = std::vector<float>;
using uchar = unsigned char;

class PillowResizeTestError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

template<typename T>
class TestMat
{
public:
    TestMat(int rows_, int cols_, int channels_, nvcv::DataKind dkind_)
        : rows(rows_)
        , cols(cols_)
        , channels(channels_)
        , data(static_cast<size_t>(rows_) * static_cast<size_t>(cols_) * static_cast<size_t>(channels_))
        , dkind(dkind_)
    {
    }

    TestMat(int rows_, int cols_, int channels_, nvcv::DataKind dkind_, const std::vector<T> &data_)
        : rows(rows_)
        , cols(cols_)
        , channels(channels_)
        , data(data_)
        , dkind(dkind_)
    {
    }

    TestMat(const TestMat &test_mat, NVCVRectI roi)
        : rows(roi.height)
        , cols(roi.width)
        , channels(test_mat.channels)
        , dkind(test_mat.dkind)
    {
        if (roi.height == test_mat.rows && roi.width == test_mat.cols)
        {
            data = test_mat.data;
        }
        else
        {
            data.resize(static_cast<size_t>(roi.width) * static_cast<size_t>(roi.height)
                        * static_cast<size_t>(test_mat.channels));

            auto copyPixel = [this, &test_mat, roi](int row, int col)
            {
                for (int c = 0; c < channels; c++)
                {
                    data[row * cols * channels + col * channels + c]
                        = test_mat.data[(row + roi.y) * test_mat.cols * channels + (col + roi.x) * channels + c];
                }
            };

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    copyPixel(i, j);
                }
            }
        }
    }

    explicit TestMat(nvcv::DataKind dkind_)
        : rows(0)
        , cols(0)
        , channels(0)
        , data()
        , dkind(dkind_)
    {
    }

    bool empty() const
    {
        return data.empty();
    }

    void create(int rows_, int cols_, int ch_)
    {
        data.assign(static_cast<size_t>(rows_) * static_cast<size_t>(cols_) * static_cast<size_t>(ch_), T{});
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
                    std::cout << " " << static_cast<int>(get(i, j, c));
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

static int ScaleDimension(int value, double scale)
{
    return static_cast<int>(static_cast<double>(value) * scale);
}

template<typename T>
static void FillRandomBytes(std::vector<T> &values)
{
    std::default_random_engine    randEng{0};
    std::uniform_int_distribution srcRand{0, 255};

    for (T &value : values)
    {
        value = static_cast<T>(srcRand(randEng));
    }
}

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
        virtual ~Filter() = default;

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

    static constexpr float box_filter_support = 0.5;

    class BoxFilter : public Filter
    {
    public:
        BoxFilter()
            : Filter(box_filter_support){};
        ~BoxFilter() override = default;
        [[nodiscard]] double filter(double x) const override;
    };

    static constexpr float bilinear_filter_support = 1.;

    class BilinearFilter : public Filter
    {
    public:
        BilinearFilter()
            : Filter(bilinear_filter_support){};
        ~BilinearFilter() override = default;
        [[nodiscard]] double filter(double x) const override;
    };

    static constexpr float hamming_filter_support = 1.;

    class HammingFilter : public Filter
    {
    public:
        HammingFilter()
            : Filter(hamming_filter_support){};
        ~HammingFilter() override = default;
        [[nodiscard]] double filter(double x) const override;
    };

    static constexpr float bicubic_filter_support = 2.;

    class BicubicFilter : public Filter
    {
    public:
        BicubicFilter()
            : Filter(bicubic_filter_support){};
        ~BicubicFilter() override = default;
        [[nodiscard]] double filter(double x) const override;
    };

    static constexpr float lanczos_filter_support = 3.;

    class LanczosFilter : public Filter
    {
    protected:
        [[nodiscard]] static double _sincFilter(double x);

    public:
        LanczosFilter()
            : Filter(lanczos_filter_support){};
        ~LanczosFilter() override = default;
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
        return clip8_lookups[std::max(-640, std::min(639, (int)(static_cast<unsigned int>(in) >> precision_bits)))];
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
     * \throw PillowResizeTestError In case the box is invalid, the interpolation filter
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
     * \throw PillowResizeTestError In case the box is invalid, the interpolation filter
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
        case NVCV_INTERP_NEAREST:
            return PillowResizeCPU::InterpolationMethods::INTERPOLATION_NEAREST;
        case NVCV_INTERP_CUBIC:
            return PillowResizeCPU::InterpolationMethods::INTERPOLATION_BICUBIC;
        case NVCV_INTERP_BOX:
            return PillowResizeCPU::InterpolationMethods::INTERPOLATION_BOX;
        case NVCV_INTERP_LANCZOS:
            return PillowResizeCPU::InterpolationMethods::INTERPOLATION_LANCZOS;
        case NVCV_INTERP_HAMMING:
            return PillowResizeCPU::InterpolationMethods::INTERPOLATION_HAMMING;
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

    auto resampleChannel = [&im_in, &kk, offset, ksize, &bounds, init_buffer](int yy, int xx, int c)
    {
        int           xmin = bounds[xx * 2 + 0];
        int           xmax = bounds[xx * 2 + 1];
        const double *k    = &kk[xx * ksize];
        double        ss   = init_buffer;

        for (int x = 0; x < xmax; ++x)
        {
            // NOLINTNEXTLINE
            ss += (T)im_in.get(yy + offset, x + xmin, c) * k[x];
        }
        return ss;
    };

    for (int yy = 0; yy < im_out.rows; ++yy)
    {
        for (int xx = 0; xx < im_out.cols; ++xx)
        {
            for (int c = 0; c < im_in.channels; ++c)
            {
                double ss = resampleChannel(yy, xx, c);
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

double PillowResizeCPU::BoxFilter::filter(double x) const
{
    if (const double half_pixel = 0.5; x > -half_pixel && x <= half_pixel)
    {
        return 1.0;
    }
    return 0.0;
}

double PillowResizeCPU::HammingFilter::filter(double x) const
{
    if (x < 0.0)
    {
        x = -x;
    }
    if (x == 0.0)
    {
        return 1.0;
    }
    if (x >= 1.0)
    {
        return 0.0;
    }
    x = x * M_PI;
    return sin(x) / x * (0.54F + 0.46F * cos(x));
}

double PillowResizeCPU::BicubicFilter::filter(double x) const
{
    const double a = -0.5;
    if (x < 0.0)
    {
        x = -x;
    }
    if (x < 1.0)
    {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    }
    if (x < 2.0)
    {
        return (((x - 5) * x + 8) * x - 4) * a;
    }
    return 0.0;
}

double PillowResizeCPU::LanczosFilter::_sincFilter(double x)
{
    if (x == 0.0)
    {
        return 1.0;
    }
    x = x * M_PI;
    return sin(x) / x;
}

double PillowResizeCPU::LanczosFilter::filter(double x) const
{
    if (const double lanczos_a_param = 3.0; - lanczos_a_param <= x && x < lanczos_a_param)
    {
        return _sincFilter(x) * _sincFilter(x / lanczos_a_param);
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
    filterscale = scale = (in1 - in0) / out_size;
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
        throw PillowResizeTestError("Memory error");
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
            if (std::fabs(ww) > 1e-5)
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
    Rect2f box{0.F, 0.F, static_cast<float>(src.cols), static_cast<float>(src.rows)};
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
        throw PillowResizeTestError("Height and width must be > 0");
    }

    if (rect[0] < 0.F || rect[1] < 0.F)
    {
        throw PillowResizeTestError("Box offset can't be negative");
    }

    if (static_cast<int>(rect[2]) > src.cols || static_cast<int>(rect[3]) > src.rows)
    {
        throw PillowResizeTestError("Box can't exceed original image size");
    }

    if (box.width < 0 || box.height < 0)
    {
        throw PillowResizeTestError("Box can't be empty");
    }

    // If box's coordinates are int and box size matches requested size
    if (static_cast<int>(box.width) == x_size && static_cast<int>(box.height) == y_size)
    {
        NVCVRectI roi{static_cast<int>(box.x), static_cast<int>(box.y), static_cast<int>(box.width),
                      static_cast<int>(box.height)};
        return TestMat(src, roi);
    }

    std::shared_ptr<Filter> filter_p;

    // Check filter.
    switch (filter)
    {
    case INTERPOLATION_BOX:
        filter_p = std::make_shared<BoxFilter>(BoxFilter());
        break;
    case INTERPOLATION_BILINEAR:
        filter_p = std::make_shared<BilinearFilter>(BilinearFilter());
        break;
    case INTERPOLATION_HAMMING:
        filter_p = std::make_shared<HammingFilter>(HammingFilter());
        break;
    case INTERPOLATION_BICUBIC:
        filter_p = std::make_shared<BicubicFilter>(BicubicFilter());
        break;
    case INTERPOLATION_LANCZOS:
        filter_p = std::make_shared<LanczosFilter>(LanczosFilter());
        break;
    default:
        throw PillowResizeTestError("unsupported resampling filter");
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
        throw PillowResizeTestError("Pixel kind not supported");
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
    {        8,          8,       16,        16,  NVCV_INTERP_LINEAR,           1, nvcv::FMT_RGB8},
    {       16,         16,        8,         8,  NVCV_INTERP_LINEAR,           1, nvcv::FMT_U16},
    {       16,         16,        8,         8,  NVCV_INTERP_LINEAR,           1, nvcv::FMT_S16},
    {        5,          5,        5,         5,  NVCV_INTERP_LINEAR,           1, nvcv::FMT_RGBf32},
    {       16,         16,        8,         8,  NVCV_INTERP_LINEAR,           1, nvcv::FMT_2F32},
    {        10,        10,        5,         5,  NVCV_INTERP_LINEAR,           1, nvcv::FMT_RGB8},
    {        42,        40,       21,        20,  NVCV_INTERP_LINEAR,           1, nvcv::FMT_RGB8},
    {        21,        21,       42,        42,  NVCV_INTERP_LINEAR,           1, nvcv::FMT_RGB8},
    {        42,        42,       21,        21,  NVCV_INTERP_LINEAR,           4, nvcv::FMT_RGB8},
    {        21,        21,       42,        42,  NVCV_INTERP_LINEAR,           5, nvcv::FMT_RGB8},
    {        42,        42,       21,        21,  NVCV_INTERP_LINEAR,           6, nvcv::FMT_RGBf32},
    {        21,        21,       42,        42,  NVCV_INTERP_LINEAR,           7, nvcv::FMT_RGBf32},
    {        21,        21,       40,        40,  NVCV_INTERP_BOX,              3, nvcv::FMT_RGBf32},
    {        41,        41,       20,        20,  NVCV_INTERP_HAMMING,          3, nvcv::FMT_RGB8},
    {        21,        21,       40,        40,  NVCV_INTERP_HAMMING,          3, nvcv::FMT_RGBf32},
    {        41,        41,       20,        20,  NVCV_INTERP_CUBIC,            3, nvcv::FMT_RGB8},
    {        21,        21,       40,        40,  NVCV_INTERP_CUBIC,            3, nvcv::FMT_RGBf32},
    {        41,        41,       20,        20,  NVCV_INTERP_LANCZOS,          3, nvcv::FMT_RGB8},
    {        21,        21,       40,        40,  NVCV_INTERP_LANCZOS,          3, nvcv::FMT_RGBf32},
    {      1920,      1080,       40,        40,   NVCV_INTERP_LINEAR,         16, nvcv::FMT_RGBA8},
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

    auto srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, srcData);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<T>> srcVec(numberOfImages);
    int                         srcVecRowStride = srcWidth * fmt.planePixelStrideBytes(0);
    int                         num_channels    = fmt.numChannels();
    nvcv::DataKind              dkind = std::is_same_v<T, float> ? nvcv::DataKind::FLOAT : nvcv::DataKind::UNSIGNED;

    for (int i = 0; i < numberOfImages; ++i)
    {
        srcVec[i].resize(srcHeight * srcWidth * num_channels);

        FillRandomBytes(srcVec[i]);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowStride(), srcVec[i].data(), srcVecRowStride,
                               srcVecRowStride, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    nvcv::Tensor imgDst(numberOfImages, {dstWidth, dstHeight}, fmt);

    cvcuda::PillowResize pillowResizeOp;

    cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(
        pillowResizeOp.getWorkspaceRequirements(numberOfImages, {srcWidth, srcHeight}, {dstWidth, dstHeight}, fmt));
    EXPECT_NO_THROW(pillowResizeOp(stream, ws.get(), imgSrc, imgDst, interpolation));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    int dstVecRowStride = dstWidth * fmt.planePixelStrideBytes(0);
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<T> testVec(dstHeight * dstWidth * num_channels);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstVecRowStride, dstAccess->sampleData(i), dstAccess->rowStride(),
                               dstVecRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        TestMat<T>                            test_in(srcHeight, srcWidth, num_channels, dkind, srcVec[i]);
        PillowResizeCPU::InterpolationMethods inter = PillowResizeCPU::getInterpolationMethods(interpolation);
        TestMat<T> test_out = PillowResizeCPU::resize(test_in, nvcv::Size2D(dstWidth, dstHeight), inter);

        // maximum absolute error
        int              maeThreshold = 2;
        int              count        = 0;
        std::vector<int> mae(testVec.size());
        for (size_t idx = 0; idx < mae.size(); ++idx)
        {
            mae[idx] = abs(static_cast<int>((test_out.data)[idx]) - static_cast<int>(testVec[idx]));
            if (mae[idx] > maeThreshold)
                count++;
        }

        if (interpolation == NVCV_INTERP_BOX || interpolation == NVCV_INTERP_LANCZOS)
            ASSERT_LE(count, 0.05 * mae.size()); // discontinuous in the filter border
        else
            ASSERT_LE(count, 0.005 * mae.size());
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
    else if (nvcv::FMT_RGBf32 == fmt || nvcv::FMT_RGBAf32 == fmt || nvcv::FMT_2F32 == fmt)
        StartTest<float>(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberOfImages, fmt);
    else if (nvcv::FMT_S16 == fmt)
        StartTest<int16_t>(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberOfImages, fmt);
    else if (nvcv::FMT_U16 == fmt)
        StartTest<uint16_t>(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberOfImages, fmt);
}

template<typename T>
void StartVarShapeTest(int srcWidthBase, int srcHeightBase, int dstWidthBase, int dstHeightBase,
                       NVCVInterpolationType interpolation, int numberOfImages, nvcv::ImageFormat fmt)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // Create input and output
    std::default_random_engine    randEng;
    std::uniform_int_distribution rndSrcWidth(ScaleDimension(srcWidthBase, 0.8), ScaleDimension(srcWidthBase, 1.1));
    std::uniform_int_distribution rndSrcHeight(ScaleDimension(srcHeightBase, 0.8), ScaleDimension(srcHeightBase, 1.1));

    std::uniform_int_distribution rndDstWidth(ScaleDimension(dstWidthBase, 0.8), ScaleDimension(dstWidthBase, 1.1));
    std::uniform_int_distribution rndDstHeight(ScaleDimension(dstHeightBase, 0.8), ScaleDimension(dstHeightBase, 1.1));

    std::vector<nvcv::Image>  imgSrc;
    std::vector<nvcv::Image>  imgDst;
    std::vector<nvcv::Size2D> srcSizes;
    std::vector<nvcv::Size2D> dstSizes;
    for (int i = 0; i < numberOfImages; ++i)
    {
        if (i == 0)
        {
            imgSrc.emplace_back(nvcv::Size2D{srcWidthBase, srcHeightBase}, fmt);
            imgDst.emplace_back(nvcv::Size2D{dstWidthBase, dstHeightBase}, fmt);
        }
        else
        {
            imgSrc.emplace_back(nvcv::Size2D{rndSrcWidth(randEng), rndSrcHeight(randEng)}, fmt);
            imgDst.emplace_back(nvcv::Size2D{rndDstWidth(randEng), rndDstHeight(randEng)}, fmt);
        }
        srcSizes.emplace_back(imgSrc.back().size());
        dstSizes.emplace_back(imgDst.back().size());
    }

    nvcv::ImageBatchVarShape batchSrc(numberOfImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numberOfImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    std::vector<std::vector<T>> srcVec(numberOfImages);
    std::vector<int>            srcVecRowStride(numberOfImages);
    int                         num_channels = fmt.numChannels();
    nvcv::DataKind              dkind = std::is_same_v<T, float> ? nvcv::DataKind::FLOAT : nvcv::DataKind::UNSIGNED;
    // Populate input
    for (int i = 0; i < numberOfImages; ++i)
    {
        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);

        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        int srcRowStride = srcWidth * fmt.planePixelStrideBytes(0);

        srcVecRowStride[i] = srcRowStride;

        srcVec[i].resize(srcHeight * srcWidth * num_channels);
        FillRandomBytes(srcVec[i]);
        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec[i].data(), srcRowStride,
                               srcRowStride, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    cvcuda::PillowResize pillowResizeOp;

    cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(
        pillowResizeOp.getWorkspaceRequirements(numberOfImages, srcSizes.data(), dstSizes.data(), fmt));
    EXPECT_NO_THROW(pillowResizeOp(stream, ws.get(), batchSrc, batchDst, interpolation));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);
        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(dstData->numPlanes() == 1);

        int dstWidth  = dstData->plane(0).width;
        int dstHeight = dstData->plane(0).height;

        int dstRowStride = dstWidth * fmt.planePixelStrideBytes(0);

        std::vector<T> testVec(dstHeight * dstWidth * num_channels);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        TestMat<T>                            test_in(srcHeight, srcWidth, num_channels, dkind, srcVec[i]);
        PillowResizeCPU::InterpolationMethods inter = PillowResizeCPU::getInterpolationMethods(interpolation);
        TestMat<T> test_out = PillowResizeCPU::resize(test_in, nvcv::Size2D(dstWidth, dstHeight), inter);

        // maximum absolute error
        int              maeThreshold = 2;
        int              count        = 0;
        std::vector<int> mae(testVec.size());
        for (size_t idx = 0; idx < mae.size(); ++idx)
        {
            mae[idx] = abs(static_cast<int>((test_out.data)[idx]) - static_cast<int>(testVec[idx]));
            if (mae[idx] > maeThreshold)
                count++;
        }

        if (interpolation == NVCV_INTERP_BOX || interpolation == NVCV_INTERP_LANCZOS)
            ASSERT_LE(count, 0.05 * mae.size()); // discontinuous in the filter border
        else
            ASSERT_LE(count, 0.005 * mae.size());
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
    else if (nvcv::FMT_RGBf32 == fmt || nvcv::FMT_RGBAf32 == fmt || nvcv::FMT_2F32 == fmt)
        StartVarShapeTest<float>(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberOfImages, fmt);
    else if (nvcv::FMT_S16 == fmt)
        StartVarShapeTest<int16_t>(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberOfImages, fmt);
    else if (nvcv::FMT_U16 == fmt)
        StartVarShapeTest<uint16_t>(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberOfImages, fmt);
}

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// PillowResize resizes each channel independently, so a planar input is resized plane-by-plane and
// must produce exactly the same pixels as the interleaved path. These tests feed identical data in
// both layouts through cvcuda::PillowResize and require the (re-interleaved) planar output to match
// the interleaved output bit-for-bit, for every supported interpolation mode and dtype.
// =============================================================================

namespace {

const nvcv::ImageFormat FMT_RGBS16{nvcv::ColorModel::RGB,  nvcv::CSPEC_UNDEFINED, nvcv::MemLayout::PITCH_LINEAR,
                                   nvcv::DataKind::SIGNED, nvcv::Swizzle::S_XYZ1, nvcv::Packing::X16_Y16_Z16};
const nvcv::ImageFormat FMT_RGBS16p{nvcv::ColorModel::RGB,  nvcv::CSPEC_UNDEFINED, nvcv::MemLayout::PITCH_LINEAR,
                                    nvcv::DataKind::SIGNED, nvcv::Swizzle::S_XYZ0, nvcv::Packing::X16,
                                    nvcv::Packing::X16,     nvcv::Packing::X16};

// Resize identical data in interleaved and planar tensor layout; outputs must match bit-for-bit.
// The shared scaffolding (upload/run/download/compare) lives in PlanarParityUtils.hpp; here we only
// bind the PillowResize call, which needs a per-format workspace.
void RunPlanarParityTensorCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH,
                               int dstW, int dstH, NVCVInterpolationType interp, int numImages)
{
    test::planar::RunTensorParity(
        planarFmt, interleavedFmt, srcW, srcH, dstW, dstH, numImages,
        [numImages, srcW, srcH, dstW, dstH, interp](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                                    nvcv::ImageFormat fmt)
        {
            cvcuda::PillowResize    op;
            cvcuda::UniqueWorkspace ws
                = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(numImages, {srcW, srcH}, {dstW, dstH}, fmt));
            EXPECT_NO_THROW(op(s, ws.get(), src, dst, interp));
        });
}

// Var-shape counterpart of RunPlanarParityTensorCase.
void RunPlanarParityVarShapeCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH,
                                 int dstW, int dstH, NVCVInterpolationType interp, int numImages)
{
    std::vector<nvcv::Size2D> srcSizes(numImages, {srcW, srcH});
    std::vector<nvcv::Size2D> dstSizes(numImages, {dstW, dstH});
    test::planar::RunVarShapeParity(
        planarFmt, interleavedFmt, srcW, srcH, dstW, dstH, numImages,
        [numImages, &srcSizes, &dstSizes, interp](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                                  const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat fmt)
        {
            cvcuda::PillowResize    op;
            cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(
                op.getWorkspaceRequirements(numImages, srcSizes.data(), dstSizes.data(), fmt));
            EXPECT_NO_THROW(op(s, ws.get(), src, dst, interp));
        });
}

} // namespace

// Parameters: srcW, srcH, dstW, dstH, interpolation, numImages, planarFmt, interleavedFmt
NVCV_TEST_SUITE_P(OpPillowResizePlanar,
                  test::ValueList<int, int, int, int, NVCVInterpolationType, int, nvcv::ImageFormat, nvcv::ImageFormat>{
  // RGB8 (3 channel uint8): every supported interpolation, expand and contract.
                      { 64, 48, 128, 96,  NVCV_INTERP_LINEAR, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                      {128, 96,  64, 48,   NVCV_INTERP_CUBIC, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                      { 64, 48, 100, 72,     NVCV_INTERP_BOX, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                      {100, 72,  40, 30, NVCV_INTERP_HAMMING, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                      { 50, 40, 100, 80, NVCV_INTERP_LANCZOS, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
 // RGBA8 (4 channel uint8).
                      { 50, 40, 100, 80,  NVCV_INTERP_LINEAR, 2,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
                      {100, 80,  50, 40,   NVCV_INTERP_CUBIC, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
 // Float planar (3 and 4 channel).
                      { 64, 48,  96, 72,  NVCV_INTERP_LINEAR, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
                      { 96, 72,  48, 36,   NVCV_INTERP_CUBIC, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
 // Signed 16-bit exercises Pillow's round-to-nearest output path.
                      { 72, 54,  45, 35,  NVCV_INTERP_LINEAR, 2,        FMT_RGBS16p,        FMT_RGBS16},
});

TEST_P(OpPillowResizePlanar, tensor_matches_interleaved)
{
    RunPlanarParityTensorCase(GetParamValue<6>(), GetParamValue<7>(), GetParamValue<0>(), GetParamValue<1>(),
                              GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>());
}

TEST_P(OpPillowResizePlanar, varshape_matches_interleaved)
{
    RunPlanarParityVarShapeCase(GetParamValue<6>(), GetParamValue<7>(), GetParamValue<0>(), GetParamValue<1>(),
                                GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>());
}

static auto OpPillowResizeNegativeParams()
{
    test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, NVCVInterpolationType> params{
        {nvcv::FMT_RGB8p, nvcv::FMT_RGB8,  NVCV_INTERP_LINEAR}, // planar in, interleaved out: layout mismatch
        {  nvcv::FMT_F64,  nvcv::FMT_F64,  NVCV_INTERP_LINEAR},
        { nvcv::FMT_RGB8, nvcv::FMT_RGB8, NVCV_INTERP_NEAREST},
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(nvcv::FMT_RGB8, nvcv::FMT_RGB8, static_cast<NVCVInterpolationType>(255));
#endif
    return params;
}

NVCV_TEST_SUITE_P(OpPillowResize_Negative, OpPillowResizeNegativeParams());

TEST_P(OpPillowResize_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat     inputFmt      = GetParamValue<0>();
    nvcv::ImageFormat     outputFmt     = GetParamValue<1>();
    NVCVInterpolationType interpolation = GetParamValue<2>();

    int numberOfImages = 3;

    // Generate input and output
    nvcv::Tensor imgSrc(numberOfImages, {24, 24}, inputFmt);
    nvcv::Tensor imgDst(numberOfImages, {12, 12}, outputFmt);

    cvcuda::PillowResize pillowResizeOp;

    cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(
        pillowResizeOp.getWorkspaceRequirements(numberOfImages, {24, 24}, {12, 12}, inputFmt));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&pillowResizeOp, &stream, &ws, &imgSrc, &imgDst, &interpolation]
                                { pillowResizeOp(stream, ws.get(), imgSrc, imgDst, interpolation); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpPillowResize_Negative, varshape_op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat     inputFmt      = GetParamValue<0>();
    nvcv::ImageFormat     outputFmt     = GetParamValue<1>();
    NVCVInterpolationType interpolation = GetParamValue<2>();

    int numberOfImages = 3;
    int srcWidthBase   = 4;
    int srcHeightBase  = 4;
    int dstWidthBase   = 8;
    int dstHeightBase  = 8;

    // Create input and output
    std::default_random_engine    randEng;
    std::uniform_int_distribution rndSrcWidth(ScaleDimension(srcWidthBase, 0.8), ScaleDimension(srcWidthBase, 1.1));
    std::uniform_int_distribution rndSrcHeight(ScaleDimension(srcHeightBase, 0.8), ScaleDimension(srcHeightBase, 1.1));

    std::uniform_int_distribution rndDstWidth(ScaleDimension(dstWidthBase, 0.8), ScaleDimension(dstWidthBase, 1.1));
    std::uniform_int_distribution rndDstHeight(ScaleDimension(dstHeightBase, 0.8), ScaleDimension(dstHeightBase, 1.1));

    std::vector<nvcv::Image>  imgSrc;
    std::vector<nvcv::Image>  imgDst;
    std::vector<nvcv::Size2D> srcSizes;
    std::vector<nvcv::Size2D> dstSizes;
    for (int i = 0; i < numberOfImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{rndSrcWidth(randEng), rndSrcHeight(randEng)}, inputFmt);
        imgDst.emplace_back(nvcv::Size2D{rndDstWidth(randEng), rndDstHeight(randEng)}, outputFmt);
        srcSizes.emplace_back(imgSrc.back().size());
        dstSizes.emplace_back(imgDst.back().size());
    }

    nvcv::ImageBatchVarShape batchSrc(numberOfImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numberOfImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Generate test result
    cvcuda::PillowResize pillowResizeOp;

    cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(
        pillowResizeOp.getWorkspaceRequirements(numberOfImages, srcSizes.data(), dstSizes.data(), inputFmt));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&pillowResizeOp, &stream, &ws, &batchSrc, &batchDst, &interpolation]
                                { pillowResizeOp(stream, ws.get(), batchSrc, batchDst, interpolation); }));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpPillowResize_Negative, varshape_hasDifferentFormat)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat     fmt           = nvcv::FMT_RGB8;
    NVCVInterpolationType interpolation = NVCV_INTERP_LINEAR;

    int numberOfImages = 3;
    int srcWidthBase   = 4;
    int srcHeightBase  = 4;
    int dstWidthBase   = 8;
    int dstHeightBase  = 8;

    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_RGBA8,             fmt},
        {            fmt, nvcv::FMT_RGBA8}
    };

    for (const auto &[inputFmtExtra, outputFmtExtra] : testSet)
    {
        // Create input and output
        std::default_random_engine    randEng;
        std::uniform_int_distribution rndSrcWidth(ScaleDimension(srcWidthBase, 0.8), ScaleDimension(srcWidthBase, 1.1));
        std::uniform_int_distribution rndSrcHeight(ScaleDimension(srcHeightBase, 0.8),
                                                   ScaleDimension(srcHeightBase, 1.1));
        std::uniform_int_distribution rndDstWidth(ScaleDimension(dstWidthBase, 0.8), ScaleDimension(dstWidthBase, 1.1));
        std::uniform_int_distribution rndDstHeight(ScaleDimension(dstHeightBase, 0.8),
                                                   ScaleDimension(dstHeightBase, 1.1));

        std::vector<nvcv::Image>  imgSrc;
        std::vector<nvcv::Image>  imgDst;
        std::vector<nvcv::Size2D> srcSizes;
        std::vector<nvcv::Size2D> dstSizes;

        // Create n-1 images with standard format
        for (int i = 0; i < numberOfImages - 1; ++i)
        {
            int tmpSrcWidth  = i == 0 ? srcWidthBase : rndSrcWidth(randEng);
            int tmpSrcHeight = i == 0 ? srcHeightBase : rndSrcHeight(randEng);
            int tmpDstWidth  = i == 0 ? dstWidthBase : rndDstWidth(randEng);
            int tmpDstHeight = i == 0 ? dstHeightBase : rndDstHeight(randEng);

            imgSrc.emplace_back(nvcv::Size2D{tmpSrcWidth, tmpSrcHeight}, fmt);
            imgDst.emplace_back(nvcv::Size2D{tmpDstWidth, tmpDstHeight}, fmt);
            srcSizes.emplace_back(imgSrc.back().size());
            dstSizes.emplace_back(imgDst.back().size());
        }

        // Add the last image with different format
        imgSrc.emplace_back(nvcv::Size2D{srcWidthBase, srcHeightBase}, inputFmtExtra);
        imgDst.emplace_back(nvcv::Size2D{dstWidthBase, dstHeightBase}, outputFmtExtra);
        srcSizes.emplace_back(imgSrc.back().size());
        dstSizes.emplace_back(imgDst.back().size());

        nvcv::ImageBatchVarShape batchSrc(numberOfImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

        nvcv::ImageBatchVarShape batchDst(numberOfImages);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        // Generate test result
        cvcuda::PillowResize pillowResizeOp;

        cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(
            pillowResizeOp.getWorkspaceRequirements(numberOfImages, srcSizes.data(), dstSizes.data(), inputFmtExtra));

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&pillowResizeOp, &stream, &ws, &batchSrc, &batchDst, &interpolation]
                                    { pillowResizeOp(stream, ws.get(), batchSrc, batchDst, interpolation); }));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// 2-channel planar (NCHW/CHW) is rejected: nvcv defines no 2-plane planar format, and PillowResize
// follows the Resize/Normalize convention of disallowing 2-channel planar (see .agents/guidance/PLANAR_GUIDELINES.md).
// The tensor is built by raw (N, C, H, W) shape because no 2-channel image format exists to construct
// it from. The var-shape 2-channel planar guard is unreachable from any constructable input (no
// 2-channel format), so only the tensor path is exercised here.
TEST(OpPillowResize_Negative, planar_two_channel_rejected)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src(
        {
            {1, 2, 24, 24},
            "NCHW"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor dst(
        {
            {1, 2, 12, 12},
            "NCHW"
    },
        nvcv::TYPE_U8);

    cvcuda::PillowResize    pillowResizeOp;
    // Workspace sizing only needs a valid format; the op rejects the 2-channel planar tensor before the
    // workspace is touched.
    cvcuda::UniqueWorkspace ws
        = cvcuda::AllocateWorkspace(pillowResizeOp.getWorkspaceRequirements(1, {24, 24}, {12, 12}, nvcv::FMT_RGB8));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&pillowResizeOp, &stream, &ws, &src, &dst]
                                { pillowResizeOp(stream, ws.get(), src, dst, NVCV_INTERP_LINEAR); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpPillowResize_Negative, invalidGetWorkSpaceReq)
{
    NVCVOperatorHandle pillowResizeHandle;
    ASSERT_EQ(NVCV_SUCCESS, cvcudaPillowResizeCreate(&pillowResizeHandle));
    std::array<NVCVSize2D, 1> inputSizesWH{{{224, 224}}};
    std::array<NVCVSize2D, 1> outputSizesWH{{{112, 112}}};
    NVCVWorkspaceRequirements req{};

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaPillowResizeVarShapeGetWorkspaceRequirements(pillowResizeHandle, 1, inputSizesWH.data(),
                                                                 outputSizesWH.data(), NVCV_IMAGE_FORMAT_U8, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaPillowResizeVarShapeGetWorkspaceRequirements(pillowResizeHandle, 1, nullptr, outputSizesWH.data(),
                                                                 NVCV_IMAGE_FORMAT_U8, &req));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaPillowResizeVarShapeGetWorkspaceRequirements(pillowResizeHandle, 1, inputSizesWH.data(), nullptr,
                                                                 NVCV_IMAGE_FORMAT_U8, &req));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaPillowResizeGetWorkspaceRequirements(
                                               pillowResizeHandle, 1, 24, 24, 24, 24, NVCV_IMAGE_FORMAT_U8, nullptr));

    nvcvOperatorDestroy(pillowResizeHandle);
}

TEST(OpPillowResize_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaPillowResizeCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}

TEST(OpPillowResize_Negative, null_workspace)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::PillowResize op;

    nvcv::Tensor imgSrc(1, {4, 4}, nvcv::FMT_RGB8);
    nvcv::Tensor imgDst(1, {4, 4}, nvcv::FMT_RGB8);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaPillowResizeSubmit(op.handle(), stream, nullptr, imgSrc.handle(),
                                                                    imgDst.handle(), NVCV_INTERP_LINEAR));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpPillowResize_Negative, null_workspace_varshape)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::PillowResize op;

    const int                numImages = 2;
    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{4, 4}, nvcv::FMT_RGB8);
        imgDst.emplace_back(nvcv::Size2D{4, 4}, nvcv::FMT_RGB8);
    }

    nvcv::ImageBatchVarShape batchSrc(numImages);
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaPillowResizeVarShapeSubmit(op.handle(), stream, nullptr, batchSrc.handle(), batchDst.handle(),
                                               NVCV_INTERP_LINEAR));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpPillowResize_Negative, invalid_interpolation)
{
    NVCVOperatorHandle op;
    ASSERT_EQ(NVCV_SUCCESS, cvcudaPillowResizeCreate(&op));

    // NVCV_INTERP_NEAREST is valid for other ops but not supported by PillowResize
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaPillowResizeSubmit(op, nullptr, nullptr, nullptr, nullptr, NVCV_INTERP_NEAREST));
    // Completely out-of-range enum value
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaPillowResizeSubmit(op, nullptr, nullptr, nullptr, nullptr, static_cast<NVCVInterpolationType>(99)));

    EXPECT_NO_THROW(nvcvOperatorDestroy(op));
}

TEST(OpPillowResize_Negative, invalid_interpolation_varshape)
{
    NVCVOperatorHandle op;
    ASSERT_EQ(NVCV_SUCCESS, cvcudaPillowResizeCreate(&op));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaPillowResizeVarShapeSubmit(op, nullptr, nullptr, nullptr, nullptr, NVCV_INTERP_NEAREST));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaPillowResizeVarShapeSubmit(op, nullptr, nullptr, nullptr, nullptr,
                                                                            static_cast<NVCVInterpolationType>(99)));

    EXPECT_NO_THROW(nvcvOperatorDestroy(op));
}

// The legacy kernels compute per-sample offsets as 32-bit (sample * imgStride), so any tensor whose
// byte extent exceeds INT32_MAX overflows the addressing and corrupts memory. The operator must
// reject such tensors instead of launching.
TEST(OpPillowResize_Negative, oversized_tensor_rejected)
{
    size_t freeMem  = 0;
    size_t totalMem = 0;
    ASSERT_EQ(cudaSuccess, cudaMemGetInfo(&freeMem, &totalMem));
    // src (1.2 GB) + dst (4.8 GB) + workspace intermediate (2.4 GB) plus slack.
    if (freeMem < 10ULL << 30)
    {
        GTEST_SKIP() << "needs ~10 GB free device memory, have " << (freeMem >> 20) << " MiB";
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat fmt = nvcv::FMT_RGBf32;
    // 48 x 2160 x 3840 x 3 floats = 4.6 GiB > INT32_MAX bytes: sample offsets overflow 32-bit.
    nvcv::Tensor      imgSrc(48, {1920, 1080}, fmt);
    nvcv::Tensor      imgDst(48, {3840, 2160}, fmt);

    cvcuda::PillowResize pillowResizeOp;

    cvcuda::UniqueWorkspace ws
        = cvcuda::AllocateWorkspace(pillowResizeOp.getWorkspaceRequirements(48, {1920, 1080}, {3840, 2160}, fmt));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { pillowResizeOp(stream, ws.get(), imgSrc, imgDst, NVCV_INTERP_LINEAR); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Var-shape sibling of oversized_tensor_rejected: the horizontally-resized intermediate is one dense
// elem-typed image slot per batch entry (numImages x maxInH x maxOutW x C) addressed with 32-bit
// products, so batches whose intermediate exceeds INT32_MAX bytes must be rejected.
TEST(OpPillowResize_Negative, oversized_varshape_rejected)
{
    size_t freeMem  = 0;
    size_t totalMem = 0;
    ASSERT_EQ(cudaSuccess, cudaMemGetInfo(&freeMem, &totalMem));
    if (freeMem < 10ULL << 30)
    {
        GTEST_SKIP() << "needs ~10 GB free device memory, have " << (freeMem >> 20) << " MiB";
    }

    try
    {
        nvcv::ImageFormat fmt            = nvcv::FMT_RGB8;
        int               numberOfImages = 192;
        nvcv::Size2D      srcSize{1920, 1080};
        nvcv::Size2D      dstSize{3840, 2160};

        std::vector<nvcv::Image>  imgSrc;
        std::vector<nvcv::Image>  imgDst;
        std::vector<nvcv::Size2D> srcSizes;
        std::vector<nvcv::Size2D> dstSizes;
        for (int i = 0; i < numberOfImages; ++i)
        {
            imgSrc.emplace_back(srcSize, fmt);
            imgDst.emplace_back(dstSize, fmt);
            srcSizes.push_back(srcSize);
            dstSizes.push_back(dstSize);
        }

        nvcv::ImageBatchVarShape batchSrc(numberOfImages);
        nvcv::ImageBatchVarShape batchDst(numberOfImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        cvcuda::PillowResize    pillowResizeOp;
        cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(
            pillowResizeOp.getWorkspaceRequirements(numberOfImages, srcSizes.data(), dstSizes.data(), fmt));

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { pillowResizeOp(stream, ws.get(), batchSrc, batchDst, NVCV_INTERP_LINEAR); }));

        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }
    catch (const nvcv::Exception &e)
    {
        if (e.code() == nvcv::Status::ERROR_OUT_OF_MEMORY)
        {
            GTEST_SKIP() << "insufficient device memory for oversized var-shape input: " << e.what();
        }
        throw;
    }
}
