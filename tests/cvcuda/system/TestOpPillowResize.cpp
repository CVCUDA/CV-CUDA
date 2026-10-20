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

#include <common/DeviceBuffer.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpPillowResize.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Rect.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
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
const nvcv::ImageFormat FMT_RGBS8{nvcv::ColorModel::RGB,  nvcv::CSPEC_UNDEFINED, nvcv::MemLayout::PITCH_LINEAR,
                                  nvcv::DataKind::SIGNED, nvcv::Swizzle::S_XYZ1, nvcv::Packing::X8_Y8_Z8};
const nvcv::ImageFormat FMT_RGBS8p{nvcv::ColorModel::RGB,  nvcv::CSPEC_UNDEFINED, nvcv::MemLayout::PITCH_LINEAR,
                                   nvcv::DataKind::SIGNED, nvcv::Swizzle::S_XYZ0, nvcv::Packing::X8,
                                   nvcv::Packing::X8,      nvcv::Packing::X8};

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
                      { 96, 72,  48, 36,  NVCV_INTERP_LINEAR, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
                      { 64, 48,  96, 72,  NVCV_INTERP_LINEAR, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
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

TEST(OpPillowResizePlanar, unaligned_vector_tail_matches_interleaved)
{
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 40, 32, 65, 49, NVCV_INTERP_LINEAR, 1);
}

TEST(OpPillowResizePlanar, rgba_float_tensor_contract_linear_matches_interleaved)
{
    // Four-channel FLOAT downscale reaches the planar fused downscale path while retaining an exact
    // interleaved-layout parity check through the shared helper.
    RunPlanarParityTensorCase(nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32, 96, 72, 48, 36, NVCV_INTERP_LINEAR, 1);
}

TEST(OpPillowResizePlanar, rgb_float_tensor_all_filters_match_interleaved)
{
    constexpr std::array<NVCVInterpolationType, 5> interpolations{
        NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC, NVCV_INTERP_BOX, NVCV_INTERP_HAMMING, NVCV_INTERP_LANCZOS};
    for (const auto interpolation : interpolations)
    {
        SCOPED_TRACE(static_cast<int>(interpolation));
        RunPlanarParityTensorCase(nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, 50, 40, 100, 80, interpolation, 2);
    }
}

TEST(OpPillowResizePlanar, signed8_tensor_contract_and_expand_match_interleaved)
{
    // Signed 8-bit contract exercises the paired interleaved horizontal round-to-nearest path.
    RunPlanarParityTensorCase(FMT_RGBS8p, FMT_RGBS8, 72, 54, 44, 34, NVCV_INTERP_LINEAR, 2);
    // Signed 8-bit expand exercises the paired planar vertical round-to-nearest path.
    RunPlanarParityTensorCase(FMT_RGBS8p, FMT_RGBS8, 48, 35, 80, 65, NVCV_INTERP_LINEAR, 1);
}

// Byte-exact reference for the optimized U8 paths. PillowResizeCPU above uses double/fixed-point
// coefficients and permits pixel error; this independent scalar oracle preserves float tap order,
// FMA accumulation and per-pass quantization without reading operator internals.

namespace {

// Pillow's bilinear/bicubic filter kernels. The literal types are load-bearing: the |x| < 1 cubic
// lobe has always evaluated in double precision, and narrowing it early moves results across a
// half-integer rounding boundary.
float PillowGoldBilinearFilter(float value)
{
    if (const float x = std::fabs(value); x < 1.f)
    {
        return static_cast<float>(1.0 - static_cast<double>(x));
    }
    return 0.f;
}

float PillowGoldBicubicFilter(float value)
{
    const float x = std::fabs(value);
    if (x < 1.f)
    {
        const auto   xd     = static_cast<double>(x);
        const double linear = std::fma(1.5, xd, -2.5);
        return static_cast<float>(std::fma(linear * xd, xd, 1.));
    }
    if (x < 2.f)
    {
        // This lobe is evaluated entirely in single precision by the operator.
        const float cubic = std::fma(std::fma(x - 5.f, x, 8.f), x, -4.f);
        return cubic * -0.5f;
    }
    return 0.f;
}

float PillowGoldFilter(NVCVInterpolationType interp, float value)
{
    return interp == NVCV_INTERP_CUBIC ? PillowGoldBicubicFilter(value) : PillowGoldBilinearFilter(value);
}

float PillowGoldFilterSupport(NVCVInterpolationType interp)
{
    return interp == NVCV_INTERP_CUBIC ? 2.f : 1.f;
}

// One axis of resampling weights: `bounds[2*i]` is the first contributing source index of output
// index i, `bounds[2*i+1]` the number of taps, and `kk[i*ksize + t]` the normalized weight.
struct PillowGoldCoeffs
{
    int                ksize = 0;
    std::vector<int>   bounds;
    std::vector<float> kk;
};

PillowGoldCoeffs PillowGoldPrecomputeCoeffs(int inSize, int outSize, NVCVInterpolationType interp)
{
    if (interp != NVCV_INTERP_LINEAR && interp != NVCV_INTERP_CUBIC)
        throw std::invalid_argument("Pillow strict oracle supports only linear and cubic interpolation");

    const float scale       = static_cast<float>(inSize) / static_cast<float>(outSize);
    float       filterscale = scale;
    if (filterscale < 1.0f)
    {
        filterscale = 1.0f;
    }
    const float support = PillowGoldFilterSupport(interp) * filterscale;

    PillowGoldCoeffs out;
    out.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    out.bounds.assign(static_cast<size_t>(outSize) * 2, 0);
    out.kk.assign(static_cast<size_t>(outSize) * out.ksize, 0.f);

    const float halfPixel = 0.5f;
    // The operator derives the reciprocal in double precision before narrowing; reproducing that
    // exactly matters because the result scales every tap distance.
    const auto  ss = static_cast<float>(1.0 / static_cast<double>(filterscale));

    for (int xx = 0; xx < outSize; ++xx)
    {
        const float center = (static_cast<float>(xx) + halfPixel) * scale;

        auto xmin = static_cast<int>(center - support + halfPixel);
        if (xmin < 0)
        {
            xmin = 0;
        }
        auto xmax = static_cast<int>(center + support + halfPixel);
        if (xmax > inSize)
        {
            xmax = inSize;
        }
        xmax -= xmin;

        float *k  = out.kk.data() + static_cast<size_t>(xx) * out.ksize;
        float  ww = 0.f;
        for (int x = 0; x < xmax; ++x)
        {
            const float w = PillowGoldFilter(interp, (static_cast<float>(x + xmin) - center + halfPixel) * ss);
            k[x]          = w;
            ww += w;
        }
        if (std::fabs(ww) > 1e-5f)
        {
            for (int x = 0; x < xmax; ++x)
            {
                k[x] /= ww;
            }
        }

        out.bounds[static_cast<size_t>(xx) * 2]     = xmin;
        out.bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
    }
    return out;
}

// Quantize each pass with round-to-nearest-even and unsigned-byte saturation.
uint8_t PillowGoldQuantize(float value)
{
    const float rounded = std::nearbyint(value);
    if (!(rounded > 0.f))
    {
        return 0;
    }
    if (rounded >= 255.f)
    {
        return 255;
    }
    return static_cast<uint8_t>(rounded);
}

// Full two-pass separable resample of one HWC sample. The horizontal pass is quantized into the
// intermediate exactly as the operator's intermediate buffer is, which is what makes an integer
// result reproducible from the host.
std::vector<uint8_t> PillowGoldResampleSample(const std::vector<uint8_t> &src, int srcW, int srcH, int dstW, int dstH,
                                              int channels, NVCVInterpolationType interp)
{
    const PillowGoldCoeffs hc = PillowGoldPrecomputeCoeffs(srcW, dstW, interp);
    const PillowGoldCoeffs vc = PillowGoldPrecomputeCoeffs(srcH, dstH, interp);

    std::vector<uint8_t> inter(static_cast<size_t>(srcH) * dstW * channels);
    for (int y = 0; y < srcH; ++y)
    {
        for (int xc = 0; xc < dstW * channels; ++xc)
        {
            const int    x    = xc / channels;
            const int    c    = xc % channels;
            const int    xmin = hc.bounds[static_cast<size_t>(x) * 2];
            const int    taps = hc.bounds[static_cast<size_t>(x) * 2 + 1];
            const float *k    = hc.kk.data() + static_cast<size_t>(x) * hc.ksize;
            float        sum  = 0.f;
            for (int t = 0; t < taps; ++t)
            {
                const size_t idx = (static_cast<size_t>(y) * srcW + xmin + t) * channels + c;
                sum              = std::fma(k[t], static_cast<float>(src[idx]), sum);
            }
            inter[static_cast<size_t>(y) * dstW * channels + xc] = PillowGoldQuantize(sum);
        }
    }

    std::vector<uint8_t> dst(static_cast<size_t>(dstH) * dstW * channels);
    for (int y = 0; y < dstH; ++y)
    {
        const int    ymin = vc.bounds[static_cast<size_t>(y) * 2];
        const int    taps = vc.bounds[static_cast<size_t>(y) * 2 + 1];
        const float *k    = vc.kk.data() + static_cast<size_t>(y) * vc.ksize;
        for (int xc = 0; xc < dstW * channels; ++xc)
        {
            float sum = 0.f;
            for (int t = 0; t < taps; ++t)
            {
                const size_t idx = static_cast<size_t>(ymin + t) * dstW * channels + xc;
                sum              = std::fma(k[t], static_cast<float>(inter[idx]), sum);
            }
            dst[static_cast<size_t>(y) * dstW * channels + xc] = PillowGoldQuantize(sum);
        }
    }
    return dst;
}

nvcv::ImageFormat PillowGoldFormat(int channels, bool planar)
{
    switch (channels)
    {
    case 1:
        return nvcv::FMT_U8;
    case 3:
        return planar ? nvcv::FMT_RGB8p : nvcv::FMT_RGB8;
    default:
        return planar ? nvcv::FMT_RGBA8p : nvcv::FMT_RGBA8;
    }
}

void PillowGoldFillSample(std::vector<uint8_t> &values, int sample)
{
    size_t i = 0;
    for (uint8_t &value : values)
    {
        const auto raw = static_cast<unsigned>((i * 2654435761u + sample * 40503u) >> 13) & 0xFFu;
        ++i;
        value = static_cast<uint8_t>(raw);
    }
}

std::vector<uint8_t> PillowPlanarByteTailSample(nvcv::DataKind kind, int sample, int width, int height, int numChannels)
{
    std::vector<uint8_t> values(static_cast<size_t>(width) * height * numChannels);

    const uint8_t low  = kind == nvcv::DataKind::SIGNED ? 0x80 : 0;
    const uint8_t high = kind == nvcv::DataKind::SIGNED ? 0x7F : 0xFF;
    for (int row = 0; row < height; ++row)
    {
        for (int column = 0; column < width; ++column)
        {
            for (int channel = 0; channel < numChannels; ++channel)
            {
                values[(static_cast<size_t>(row) * width + column) * numChannels + channel]
                    = ((column / 5 + row / 3 + channel + sample) % 2) ? high : low;
            }
        }
    }
    return values;
}

void PillowGoldCopyRows(uint8_t *dst, const uint8_t *src, int height, int64_t rowStride, size_t rowBytes)
{
    for (int row = 0; row < height; ++row)
    {
        std::memcpy(dst + row * rowStride, src + row * rowBytes, rowBytes);
    }
}

// Optional row/sample padding exercises the packed-access alignment gates.
struct PillowGoldPitchedTensor
{
    nvcv::Tensor tensor;
    NVCVByte    *allocation      = nullptr;
    size_t       allocationBytes = 0;
};

PillowGoldPitchedTensor PillowGoldMakeTensor(int numImages, int width, int height, int channels, bool planar,
                                             int rowPad, int samplePad, int planePad = 0, bool chw = false)
{
    PillowGoldPitchedTensor out;
    nvcv::TensorShape       shape{
        {numImages, height, width, channels},
        "NHWC"
    };
    if (planar)
    {
        shape = chw ? nvcv::TensorShape{{channels, height, width}, "CHW"}
                    : nvcv::TensorShape{{numImages, channels, height, width}, "NCHW"};
    }
    if (rowPad == 0 && samplePad == 0 && planePad == 0)
    {
        out.tensor = planar ? nvcv::Tensor(shape, nvcv::TYPE_U8)
                            : nvcv::util::CreateTensor(numImages, width, height, PillowGoldFormat(channels, planar));
        return out;
    }

    const size_t rowBytes     = static_cast<size_t>(width) * (planar ? 1 : channels);
    const size_t rowStride    = rowBytes + rowPad;
    const size_t planeStride  = rowStride * height + planePad;
    const size_t sampleStride = planeStride * (planar ? channels : 1) + samplePad;
    out.allocationBytes       = sampleStride * numImages;

    EXPECT_EQ(cudaSuccess, cudaMalloc(&out.allocation, out.allocationBytes));
    EXPECT_EQ(cudaSuccess, cudaMemset(out.allocation, 0xA5, out.allocationBytes));

    nvcv::TensorDataStridedCuda::Buffer buffer{};
    buffer.basePtr = out.allocation;
    if (planar)
    {
        const int channelDim           = chw ? 0 : 1;
        buffer.strides[0]              = static_cast<int64_t>(sampleStride);
        buffer.strides[channelDim]     = static_cast<int64_t>(planeStride);
        buffer.strides[channelDim + 1] = static_cast<int64_t>(rowStride);
        buffer.strides[channelDim + 2] = 1;
    }
    else
    {
        buffer.strides[0] = static_cast<int64_t>(sampleStride);
        buffer.strides[1] = static_cast<int64_t>(rowStride);
        buffer.strides[2] = static_cast<int64_t>(channels);
        buffer.strides[3] = 1;
    }
    out.tensor = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, buffer});
    return out;
}

void RunPillowBitExactCase(int srcW, int srcH, int dstW, int dstH, int channels, NVCVInterpolationType interp,
                           int numImages, bool planar, int srcRowPad = 0, int srcSamplePad = 0, int dstRowPad = 0,
                           int srcPlanePad = 0, int dstPlanePad = 0, int dstSamplePad = 0, bool chw = false)
{
    const nvcv::ImageFormat fmt = PillowGoldFormat(channels, planar);
    ASSERT_TRUE(!chw || (planar && numImages == 1));
    ASSERT_TRUE(planar || (srcPlanePad == 0 && dstPlanePad == 0));

    auto pitchedSrc
        = PillowGoldMakeTensor(numImages, srcW, srcH, channels, planar, srcRowPad, srcSamplePad, srcPlanePad, chw);
    auto pitchedDst
        = PillowGoldMakeTensor(numImages, dstW, dstH, channels, planar, dstRowPad, dstSamplePad, dstPlanePad, chw);
    const nvcv::Tensor &src = pitchedSrc.tensor;
    const nvcv::Tensor &dst = pitchedDst.tensor;

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, srcData);
    ASSERT_NE(nullptr, dstData);
    if (planar)
    {
        ASSERT_EQ(chw ? nvcv::TENSOR_CHW : nvcv::TENSOR_NCHW, srcData->layout());
        ASSERT_EQ(chw ? nvcv::TENSOR_CHW : nvcv::TENSOR_NCHW, dstData->layout());
    }
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(srcAccess);
    ASSERT_TRUE(dstAccess);

    const int    dstRowStride    = dstW * channels;
    const size_t srcSampleValues = static_cast<size_t>(srcH) * srcW * channels;

    std::vector<std::vector<uint8_t>> goldHWC(numImages);

    for (int i = 0; i < numImages; ++i)
    {
        std::vector<uint8_t> srcHWC(srcSampleValues);
        PillowGoldFillSample(srcHWC, i);
        goldHWC[i] = PillowGoldResampleSample(srcHWC, srcW, srcH, dstW, dstH, channels, interp);

        if (planar)
        {
            test::planar::UploadPlanarSample(*srcAccess, i,
                                             test::planar::DeinterleaveToPlanes(srcHWC, srcW, srcH, channels, 1), srcW,
                                             srcH, channels, 1);
        }
        else
        {
            test::planar::UploadInterleavedSample(*srcAccess, i, srcHWC, srcW, srcH, srcW * channels);
        }
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::PillowResize    op;
    cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(
        op.getWorkspaceRequirements(numImages, nvcv::Size2D{srcW, srcH}, nvcv::Size2D{dstW, dstH}, fmt));
    ASSERT_NO_THROW(op(stream, ws.get(), src, dst, interp));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        std::vector<uint8_t> bytes
            = planar ? test::planar::InterleaveFromPlanes(
                  test::planar::DownloadPlanarSample(*dstAccess, i, dstW, dstH, channels, 1), dstW, dstH, channels, 1)
                     : test::planar::DownloadInterleavedSample(*dstAccess, i, dstW, dstH, dstRowStride);

        ASSERT_EQ(goldHWC[i], bytes);
    }

    if (pitchedSrc.allocation != nullptr)
    {
        EXPECT_EQ(cudaSuccess, cudaFree(pitchedSrc.allocation));
    }
    if (pitchedDst.allocation != nullptr)
    {
        const int            planes   = planar ? channels : 1;
        const size_t         rowBytes = static_cast<size_t>(dstW) * (planar ? 1 : channels);
        std::vector<uint8_t> expected(pitchedDst.allocationBytes, 0xA5);
        std::vector<uint8_t> actual(pitchedDst.allocationBytes);
        for (int sample = 0; sample < numImages; ++sample)
        {
            const auto packed = planar ? test::planar::DeinterleaveToPlanes(goldHWC[sample], dstW, dstH, channels, 1)
                                       : goldHWC[sample];
            const auto sampleOffset = dstAccess->sampleData(sample) - dstData->basePtr();
            for (int plane = 0; plane < planes; ++plane)
            {
                PillowGoldCopyRows(expected.data() + sampleOffset + plane * dstAccess->chStride(),
                                   packed.data() + static_cast<size_t>(plane) * dstH * rowBytes, dstH,
                                   dstAccess->rowStride(), rowBytes);
            }
        }
        ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), pitchedDst.allocation, actual.size(), cudaMemcpyDeviceToHost));
        EXPECT_EQ(expected, actual);
        EXPECT_EQ(cudaSuccess, cudaFree(pitchedDst.allocation));
    }
}

struct PillowGoldStridedImage
{
    nvcv::Image image;
    size_t      allocationBytes = 0;
};

PillowGoldStridedImage PillowGoldMakeImage(nvcv::Size2D size, nvcv::ImageFormat format, int rowPadding)
{
    PillowGoldStridedImage             out;
    nvcv::ImageDataStridedCuda::Buffer buffer{};
    buffer.numPlanes = format.numPlanes();
    for (int planeIndex = 0; planeIndex < buffer.numPlanes; ++planeIndex)
    {
        auto &plane     = buffer.planes[planeIndex];
        plane.width     = size.w;
        plane.height    = size.h;
        plane.rowStride = size.w * format.planePixelStrideBytes(planeIndex) + rowPadding + planeIndex * 3;
        out.allocationBytes += static_cast<size_t>(plane.rowStride) * size.h + 17;
    }

    NVCVByte *allocation = nullptr;
    auto      status     = cudaMalloc(&allocation, out.allocationBytes);
    if (status != cudaSuccess)
    {
        throw PillowResizeTestError(cudaGetErrorString(status));
    }
    auto cleanup = [](NVCVByte *buffer)
    {
        EXPECT_EQ(cudaSuccess, cudaFree(buffer));
    };
    std::unique_ptr<NVCVByte, decltype(cleanup)> owner(allocation, cleanup);
    status = cudaMemset(allocation, 0xA5, out.allocationBytes);
    if (status != cudaSuccess)
    {
        throw PillowResizeTestError(cudaGetErrorString(status));
    }
    size_t offset = 0;
    for (int planeIndex = 0; planeIndex < buffer.numPlanes; ++planeIndex)
    {
        auto &plane   = buffer.planes[planeIndex];
        plane.basePtr = allocation + offset;
        offset += static_cast<size_t>(plane.rowStride) * size.h + 17;
    }
    out.image = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{format, buffer},
                                    nvcv::ImageDataCleanupCallback{[allocation](const nvcv::ImageData &)
                                                                   {
                                                                       EXPECT_EQ(cudaSuccess, cudaFree(allocation));
                                                                   }});
    owner.release();
    return out;
}

std::vector<uint8_t> PillowGoldPackImage(const PillowGoldStridedImage &image, const std::vector<uint8_t> &values,
                                         bool planar)
{
    const auto   data     = image.image.exportData<nvcv::ImageDataStridedCuda>();
    const auto   size     = image.image.size();
    const int    channels = image.image.format().numChannels();
    const auto   packed   = planar ? test::planar::DeinterleaveToPlanes(values, size.w, size.h, channels, 1) : values;
    const size_t rowBytes = static_cast<size_t>(size.w) * (planar ? 1 : channels);
    std::vector<uint8_t> expected(image.allocationBytes, 0xA5);
    for (int planeIndex = 0; planeIndex < data->numPlanes(); ++planeIndex)
    {
        const auto &plane  = data->plane(planeIndex);
        const auto  offset = static_cast<size_t>(plane.basePtr - data->plane(0).basePtr);
        PillowGoldCopyRows(expected.data() + offset,
                           packed.data() + static_cast<size_t>(planeIndex) * size.h * rowBytes, size.h, plane.rowStride,
                           rowBytes);
    }
    return expected;
}

struct PillowGoldVarShapeSizes
{
    std::array<nvcv::Size2D, 4> source;
    std::array<nvcv::Size2D, 4> destination;
};

PillowGoldVarShapeSizes PillowGoldVarShapeScenario(int scenario)
{
    const std::array<std::array<nvcv::Size2D, 4>, 8> sourceShapes{
        {
         {{{1920, 24}, {1024, 48}, {130, 34}, {18, 6}}},
         {{{1920, 25}, {131, 39}, {65, 17}, {7, 3}}},
         {{{49, 27}, {17, 9}, {1, 1}, {65, 31}}},
         {{{1920, 9}, {1537, 5}, {257, 3}, {1, 1}}},
         {{{1920, 24}, {1024, 17}, {130, 9}, {18, 3}}},
         {{{1920, 25}, {131, 39}, {65, 17}, {7, 3}}},
         {{{1920, 25}, {131, 48}, {65, 17}, {7, 3}}},
         {{{1920, 25}, {131, 48}, {65, 17}, {8, 3}}},
         }
    };
    const std::array<std::array<nvcv::Size2D, 4>, 8> destinationShapes{
        {
         {{{960, 12}, {512, 24}, {65, 17}, {9, 3}}},
         {{{864, 11}, {57, 21}, {33, 5}, {1, 1}}},
         {{{101, 55}, {35, 19}, {2, 3}, {33, 63}}},
         {{{9, 3}, {7, 2}, {1, 1}, {1, 1}}},
         {{{960, 24}, {512, 17}, {65, 9}, {9, 3}}},
         {{{864, 25}, {57, 39}, {33, 17}, {1, 3}}},
         {{{833, 11}, {57, 21}, {33, 5}, {2, 1}}},
         {{{833, 11}, {57, 21}, {33, 5}, {2, 1}}},
         }
    };
    return {sourceShapes.at(scenario), destinationShapes.at(scenario)};
}

void PillowGoldFillVarShapeSample(std::vector<uint8_t> &input, nvcv::Size2D size, int channels, int sampleIndex,
                                  int pattern)
{
    PillowGoldFillSample(input, sampleIndex);
    if (pattern == 0)
    {
        return;
    }
    for (int row = 0; row < size.h; ++row)
    {
        for (int column = 0; column < size.w; ++column)
        {
            for (int channel = 0; channel < channels; ++channel)
            {
                const size_t index = (static_cast<size_t>(row) * size.w + column) * channels + channel;
                input[index]
                    = pattern == 1
                        ? static_cast<uint8_t>(((row / 2 + column / 3 + channel + sampleIndex) % 2) * 255)
                        : static_cast<uint8_t>((row * 23 + column * 5 + channel * 113 + sampleIndex * 67) & 255);
            }
        }
    }
}

void RunPillowVarShapeBitExactBatch(const cvcuda::PillowResize &op, const cvcuda::Workspace &workspace,
                                    cudaStream_t stream, bool planar, NVCVInterpolationType interpolation,
                                    const PillowGoldVarShapeSizes &sizes, int pattern)
{
    ASSERT_EQ(FE_TONEAREST, std::fegetround());
    SCOPED_TRACE(t::Message() << "planar=" << planar << " interpolation=" << interpolation << " pattern=" << pattern);
    const auto                         &sourceSizes      = sizes.source;
    const auto                         &destinationSizes = sizes.destination;
    constexpr int                       channels         = 3;
    constexpr int                       numImages        = 4;
    const auto                          format           = PillowGoldFormat(channels, planar);
    std::vector<PillowGoldStridedImage> sources;
    std::vector<PillowGoldStridedImage> destinations;
    std::vector<std::vector<uint8_t>>   sourceBytes;
    std::vector<std::vector<uint8_t>>   expectedBytes;
    nvcv::ImageBatchVarShape            sourceBatch(numImages);
    nvcv::ImageBatchVarShape            destinationBatch(numImages);

    for (int sampleIndex = 0; sampleIndex < numImages; ++sampleIndex)
    {
        const auto sourceSize      = sourceSizes[sampleIndex];
        const auto destinationSize = destinationSizes[sampleIndex];
        sources.push_back(PillowGoldMakeImage(sourceSize, format, 5 + sampleIndex * 2));
        destinations.push_back(PillowGoldMakeImage(destinationSize, format, 7 + sampleIndex * 2));
        sourceBatch.pushBack(sources.back().image);
        destinationBatch.pushBack(destinations.back().image);
        std::vector<uint8_t> input(static_cast<size_t>(sourceSize.w) * sourceSize.h * channels);
        PillowGoldFillVarShapeSample(input, sourceSize, channels, sampleIndex, pattern);
        sourceBytes.push_back(PillowGoldPackImage(sources.back(), input, planar));
        const auto reference = PillowGoldResampleSample(input, sourceSize.w, sourceSize.h, destinationSize.w,
                                                        destinationSize.h, channels, interpolation);
        expectedBytes.push_back(PillowGoldPackImage(destinations.back(), reference, planar));
        const auto sourceData = sources.back().image.exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(cudaSuccess, cudaMemcpy(sourceData->plane(0).basePtr, sourceBytes.back().data(),
                                          sourceBytes.back().size(), cudaMemcpyHostToDevice));
        const auto destinationData = destinations.back().image.exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(destinationData->plane(0).basePtr, 0xA5,
                                               destinations.back().allocationBytes, stream));
    }

    ASSERT_NO_THROW(op(stream, workspace, sourceBatch, destinationBatch, interpolation));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int sampleIndex = 0; sampleIndex < numImages; ++sampleIndex)
    {
        SCOPED_TRACE(sampleIndex);
        const auto           destinationData = destinations[sampleIndex].image.exportData<nvcv::ImageDataStridedCuda>();
        std::vector<uint8_t> actual(expectedBytes[sampleIndex].size());
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(actual.data(), destinationData->plane(0).basePtr, actual.size(), cudaMemcpyDeviceToHost));
        EXPECT_EQ(expectedBytes[sampleIndex], actual);
        const auto sourceData = sources[sampleIndex].image.exportData<nvcv::ImageDataStridedCuda>();
        actual.resize(sourceBytes[sampleIndex].size());
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(actual.data(), sourceData->plane(0).basePtr, actual.size(), cudaMemcpyDeviceToHost));
        EXPECT_EQ(sourceBytes[sampleIndex], actual);
    }
}

void RunPillowVarShapeBitExactCase(bool planar, NVCVInterpolationType interpolation, int scenario, int pattern)
{
    SCOPED_TRACE(scenario);
    const auto        sizes = PillowGoldVarShapeScenario(scenario);
    test::StreamGuard stream;
    ASSERT_NE(nullptr, stream.get());
    cvcuda::PillowResize op;
    auto                 workspace = cvcuda::AllocateWorkspace(
                        op.getWorkspaceRequirements(static_cast<int>(sizes.source.size()), sizes.source.data(),
                                                    sizes.destination.data(), PillowGoldFormat(3, planar)));
    ASSERT_NO_FATAL_FAILURE(
        RunPillowVarShapeBitExactBatch(op, workspace.get(), stream.get(), planar, interpolation, sizes, pattern));
}

void RunPillowVarShapeWorkspaceReuse(bool planar, NVCVInterpolationType interpolation)
{
    std::array<PillowGoldVarShapeSizes, 3> batches{PillowGoldVarShapeScenario(4), PillowGoldVarShapeScenario(3),
                                                   PillowGoldVarShapeScenario(5)};
    std::rotate(batches[1].source.begin(), batches[1].source.begin() + 1, batches[1].source.end() - 1);
    std::rotate(batches[1].destination.begin(), batches[1].destination.begin() + 1, batches[1].destination.end() - 1);
    std::swap(batches[2].source[0], batches[2].source[2]);
    std::swap(batches[2].destination[0], batches[2].destination[2]);

    test::StreamGuard stream;
    ASSERT_NE(nullptr, stream.get());
    cvcuda::PillowResize op;
    const auto           format       = PillowGoldFormat(3, planar);
    auto                 requirements = op.getWorkspaceRequirements(static_cast<int>(batches[0].source.size()),
                                                                    batches[0].source.data(), batches[0].destination.data(), format);
    for (size_t batchIndex = 1; batchIndex < batches.size(); ++batchIndex)
    {
        requirements = cvcuda::MaxWorkspaceReq(
            requirements, op.getWorkspaceRequirements(static_cast<int>(batches[batchIndex].source.size()),
                                                      batches[batchIndex].source.data(),
                                                      batches[batchIndex].destination.data(), format));
    }
    auto                        workspace   = cvcuda::AllocateWorkspace(requirements);
    const auto                 *hostStorage = workspace.get().hostMem.data;
    const auto                 *cudaStorage = workspace.get().cudaMem.data;
    const std::array<size_t, 4> sequence{0, 1, 2, 0};
    for (size_t submission = 0; submission < sequence.size(); ++submission)
    {
        SCOPED_TRACE(submission);
        ASSERT_NO_FATAL_FAILURE(RunPillowVarShapeBitExactBatch(op, workspace.get(), stream.get(), planar, interpolation,
                                                               batches[sequence[submission]], submission % 3));
        ASSERT_EQ(hostStorage, workspace.get().hostMem.data);
        ASSERT_EQ(cudaStorage, workspace.get().cudaMem.data);
    }
}

} // namespace

NVCV_TEST_SUITE_P(OpPillowResizeVarShapeBitExact,
                  test::ValueList<bool, NVCVInterpolationType, int>{
                      { true,  NVCV_INTERP_CUBIC, 0},
                      { true,  NVCV_INTERP_CUBIC, 1},
                      { true,  NVCV_INTERP_CUBIC, 2},
                      { true,  NVCV_INTERP_CUBIC, 3},
                      { true,  NVCV_INTERP_CUBIC, 4},
                      { true,  NVCV_INTERP_CUBIC, 5},
                      {false,  NVCV_INTERP_CUBIC, 0},
                      {false,  NVCV_INTERP_CUBIC, 1},
                      {false,  NVCV_INTERP_CUBIC, 2},
                      {false,  NVCV_INTERP_CUBIC, 3},
                      {false,  NVCV_INTERP_CUBIC, 4},
                      {false,  NVCV_INTERP_CUBIC, 5},
                      {false,  NVCV_INTERP_CUBIC, 6},
                      {false,  NVCV_INTERP_CUBIC, 7},
                      { true, NVCV_INTERP_LINEAR, 0},
                      { true, NVCV_INTERP_LINEAR, 1},
                      { true, NVCV_INTERP_LINEAR, 2},
                      { true, NVCV_INTERP_LINEAR, 3},
                      { true, NVCV_INTERP_LINEAR, 4},
                      { true, NVCV_INTERP_LINEAR, 5},
                      {false, NVCV_INTERP_LINEAR, 0},
                      {false, NVCV_INTERP_LINEAR, 1},
                      {false, NVCV_INTERP_LINEAR, 2},
                      {false, NVCV_INTERP_LINEAR, 3},
                      {false, NVCV_INTERP_LINEAR, 4},
                      {false, NVCV_INTERP_LINEAR, 5},
});

TEST_P(OpPillowResizeVarShapeBitExact, rgb8_matches_scalar_oracle_and_preserves_padding)
{
    for (int pattern = 0; pattern < 3; ++pattern)
    {
        SCOPED_TRACE(pattern);
        RunPillowVarShapeBitExactCase(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), pattern);
    }
}

NVCV_TEST_SUITE_P(OpPillowResizeVarShapeBitExactReuse, test::ValueList<bool, NVCVInterpolationType>{
                                                           { true,  NVCV_INTERP_CUBIC},
                                                           {false,  NVCV_INTERP_CUBIC},
                                                           { true, NVCV_INTERP_LINEAR},
                                                           {false, NVCV_INTERP_LINEAR},
});

TEST_P(OpPillowResizeVarShapeBitExactReuse, shared_global_shared_workspace_matches_scalar_oracle)
{
    RunPillowVarShapeWorkspaceReuse(GetParamValue<0>(), GetParamValue<1>());
}

// srcW, srcH, dstW, dstH, channels, interpolation, numImages, planar
NVCV_TEST_SUITE_P(OpPillowResizeBitExact, test::ValueList<int, int, int, int, int, NVCVInterpolationType, int, bool>{
  // Exact 2x: tiny boundaries, odd paired tail and multiple samples.
                                              {   2,   2,    4,  4, 3, NVCV_INTERP_LINEAR, 1, false},
                                              {   1,   1,    2,  2, 3, NVCV_INTERP_LINEAR, 1, false},
                                              { 257,  33,  514, 66, 3, NVCV_INTERP_LINEAR, 2, false},
 // Narrow/scalar paths, both filters, odd extents and anisotropic ratios.
                                              {  64,  48,  128, 96, 3,  NVCV_INTERP_CUBIC, 2, false},
                                              { 128,  96,   64, 48, 1, NVCV_INTERP_LINEAR, 2, false},
                                              {  64,  48,  128, 96, 1,  NVCV_INTERP_CUBIC, 1, false},
                                              { 129,  97,   65, 49, 3,  NVCV_INTERP_CUBIC, 1, false},
                                              {  65,  49,  131, 99, 3, NVCV_INTERP_LINEAR, 1, false},
                                              { 129,  97,   65, 49, 1,  NVCV_INTERP_CUBIC, 1, false},
                                              { 131,  99,   66, 50, 4, NVCV_INTERP_LINEAR, 1, false},
                                              { 192, 108,   48, 96, 3,  NVCV_INTERP_CUBIC, 1, false},
                                              {  48,  96,  192, 54, 3, NVCV_INTERP_LINEAR, 1, false},
                                              { 128,  96,   64, 48, 3, NVCV_INTERP_LINEAR, 2,  true},
 // Planar scalar paths and anisotropic contraction.
                                              {  64,  48,  128, 96, 3,  NVCV_INTERP_CUBIC, 1,  true},
                                              { 129,  97,   65, 49, 3,  NVCV_INTERP_CUBIC, 1,  true},
                                              {  65,  49,  131, 99, 3, NVCV_INTERP_LINEAR, 1,  true},
                                              { 192, 108,   48, 96, 4,  NVCV_INTERP_CUBIC, 1,  true},
 // Wide spans, including the C4 cubic contraction fallback.
                                              {1024,  48,  512, 24, 1,  NVCV_INTERP_CUBIC, 1, false},
                                              { 512,  24, 1024, 48, 3,  NVCV_INTERP_CUBIC, 1, false},
                                              {1024,  48,  512, 24, 3, NVCV_INTERP_LINEAR, 1, false},
                                              { 512,  24, 1024, 48, 4, NVCV_INTERP_LINEAR, 1, false},
                                              {1024,  48,  512, 24, 4,  NVCV_INTERP_CUBIC, 1, false},
                                              { 512,  24, 1024, 48, 3, NVCV_INTERP_LINEAR, 1,  true},
                                              {1024,  48,  512, 24, 3,  NVCV_INTERP_CUBIC, 1,  true},
                                              { 512,  24, 1024, 48, 4, NVCV_INTERP_LINEAR, 1,  true},
                                              {1024,  48,  512, 24, 4,  NVCV_INTERP_CUBIC, 1,  true},
 // Partial horizontal groups for each scalar/vector and layout family.
                                              { 520,  24, 1040, 48, 1, NVCV_INTERP_LINEAR, 1, false},
                                              { 522,  24, 1044, 48, 4,  NVCV_INTERP_CUBIC, 1, false},
                                              { 522,  24, 1044, 48, 3, NVCV_INTERP_LINEAR, 1,  true},
                                              { 522,  24, 1044, 48, 4,  NVCV_INTERP_CUBIC, 1,  true},
                                              {  63,   7,   31,  7, 1, NVCV_INTERP_LINEAR, 2,  true},
                                              {  33,   9,   65,  5, 1, NVCV_INTERP_LINEAR, 2,  true},
                                              {  17,   7,   33,  7, 1,  NVCV_INTERP_CUBIC, 2,  true},
                                              { 257,   5,  128,  9, 1,  NVCV_INTERP_CUBIC, 2,  true},
                                              {  65,   7,   32,  7, 3, NVCV_INTERP_LINEAR, 2,  true},
                                              {  49,   9,   97,  5, 3, NVCV_INTERP_LINEAR, 2,  true},
                                              {  33,   7,   65,  7, 3,  NVCV_INTERP_CUBIC, 2,  true},
                                              { 259,   5,  129,  9, 3,  NVCV_INTERP_CUBIC, 2,  true},
                                              { 195,   7,   97,  7, 4, NVCV_INTERP_LINEAR, 2,  true},
                                              {  63,   9,  127,  5, 4, NVCV_INTERP_LINEAR, 2,  true},
                                              {  65,   7,  129,  7, 4,  NVCV_INTERP_CUBIC, 2,  true},
                                              {  67,   5,   33,  9, 4,  NVCV_INTERP_CUBIC, 2,  true},
});

TEST_P(OpPillowResizeBitExact, u8_matches_scalar_oracle)
{
    RunPillowBitExactCase(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(),
                          GetParamValue<4>(), GetParamValue<5>(), GetParamValue<6>(), GetParamValue<7>());
}

namespace {

void StorePillowRgbPixels(std::vector<uint8_t> &storage, const std::vector<uint8_t> &pixels, nvcv::Size2D size,
                          bool planar, const nvcv::ImageDataStridedCuda::Buffer &buffer,
                          const std::array<size_t, 3> &offsets)
{
    const int pixelBytes = planar ? 1 : 3;
    for (int row = 0; row < size.h; ++row)
        for (int column = 0; column < size.w; ++column)
            for (int channel = 0; channel < 3; ++channel)
            {
                const int plane = planar ? channel : 0;
                storage[offsets[plane] + row * buffer.planes[plane].rowStride + column * pixelBytes
                        + (planar ? 0 : channel)]
                    = pixels[(static_cast<size_t>(row) * size.w + column) * 3 + channel];
            }
}

void RunPillowVarShapeBitExactCase(bool planar, bool largeSupport, NVCVInterpolationType interpolation)
{
    std::vector<nvcv::Size2D> inputSizes{
        {257, 129},
        {133,  99},
        { 31,  17},
        {129,  65},
        {128,  64}
    };
    std::vector<nvcv::Size2D> outputSizes{
        {129, 65},
        { 65, 49},
        { 65, 33},
        { 63, 31},
        { 64, 32}
    };
    if (largeSupport)
    {
        inputSizes.push_back({65, 257});
        outputSizes.push_back({33, 17});
    }

    const int                                                   count  = static_cast<int>(inputSizes.size());
    const auto                                                  format = planar ? nvcv::FMT_RGB8p : nvcv::FMT_RGB8;
    std::vector<std::unique_ptr<NVCVByte, decltype(&cudaFree)>> allocations;
    std::vector<std::vector<uint8_t>>                           expected;
    std::vector<NVCVByte *>                                     outputAllocations;
    nvcv::ImageBatchVarShape                                    source(count);
    nvcv::ImageBatchVarShape                                    destination(count);

    auto makeImage = [&](nvcv::Size2D size, const std::vector<uint8_t> &pixels, int sample, bool output)
    {
        nvcv::ImageDataStridedCuda::Buffer buffer{};
        buffer.numPlanes = planar ? 3 : 1;
        std::array<size_t, 3> offsets{};
        size_t                bytes = 3 + sample;
        for (int plane = 0; plane < buffer.numPlanes; ++plane)
        {
            offsets[plane]       = bytes;
            auto &descriptor     = buffer.planes[plane];
            descriptor.width     = size.w;
            descriptor.height    = size.h;
            descriptor.rowStride = size.w * (planar ? 1 : 3) + 7 + sample + plane * 3;
            bytes += descriptor.rowStride * size.h + 19 + plane;
        }
        std::vector<uint8_t> storage(bytes, 0xA5);
        StorePillowRgbPixels(storage, pixels, size, planar, buffer, offsets);
        NVCVByte *allocation = nullptr;
        EXPECT_EQ(cudaSuccess, cudaMalloc(&allocation, bytes));
        if (allocation == nullptr)
            throw PillowResizeTestError("PillowResize pitched-image allocation failed");
        allocations.emplace_back(allocation, &cudaFree);
        for (int plane = 0; plane < buffer.numPlanes; ++plane)
            buffer.planes[plane].basePtr = allocation + offsets[plane];
        if (output)
        {
            expected.push_back(storage);
            outputAllocations.push_back(allocation);
            EXPECT_EQ(cudaSuccess, cudaMemset(allocation, 0xA5, bytes));
        }
        else
        {
            EXPECT_EQ(cudaSuccess, cudaMemcpy(allocation, storage.data(), bytes, cudaMemcpyHostToDevice));
        }
        return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{format, buffer});
    };

    for (int sample = 0; sample < count; ++sample)
    {
        const auto           input  = inputSizes[sample];
        const auto           output = outputSizes[sample];
        std::vector<uint8_t> pixels(static_cast<size_t>(input.w) * input.h * 3);
        PillowGoldFillSample(pixels, sample);
        auto gold = PillowGoldResampleSample(pixels, input.w, input.h, output.w, output.h, 3, interpolation);
        source.pushBack(makeImage(input, pixels, sample, false));
        destination.pushBack(makeImage(output, gold, sample, true));
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::PillowResize op;
    auto                 workspace
        = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(count, inputSizes.data(), outputSizes.data(), format));
    EXPECT_NO_THROW(op(stream, workspace.get(), source, destination, interpolation));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    for (int sample = 0; sample < count; ++sample)
    {
        SCOPED_TRACE(sample);
        std::vector<uint8_t> actual(expected[sample].size());
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(actual.data(), outputAllocations[sample], actual.size(), cudaMemcpyDeviceToHost));
        EXPECT_EQ(expected[sample], actual);
    }
}

} // namespace

TEST(OpPillowResizeBitExact, varshape_rgb_heterogeneous_pitched_tiles)
{
    for (bool planar : {false, true})
        for (auto interpolation : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
        {
            SCOPED_TRACE(::testing::Message() << "planar=" << planar << " interpolation=" << interpolation);
            RunPillowVarShapeBitExactCase(planar, false, interpolation);
        }
}

TEST(OpPillowResizeBitExact, varshape_rgb_heterogeneous_large_support_fallback)
{
    for (bool planar : {false, true})
        for (auto interpolation : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
        {
            SCOPED_TRACE(::testing::Message() << "planar=" << planar << " interpolation=" << interpolation);
            RunPillowVarShapeBitExactCase(planar, true, interpolation);
        }
}

namespace {

nvcv::ImageDataStridedCuda::Buffer MakePillowAliasOutput(const nvcv::ImageDataStridedCuda::Buffer &target, int sample,
                                                         int aliasMode, int shiftBytes)
{
    auto output = target;
    for (int plane = 0; plane < target.numPlanes; ++plane)
    {
        const int targetPlane = aliasMode == 3 && sample < 2 ? (plane + 1) % target.numPlanes : plane;
        output.planes[plane]  = target.planes[targetPlane];
        if (sample < 2)
            output.planes[plane].basePtr += shiftBytes;
    }
    return output;
}

void RunPillowVarShapeAliasCase(bool planar, NVCVInterpolationType interpolation, nvcv::Size2D size, int aliasMode)
{
    const nvcv::ImageFormat           format     = planar ? nvcv::FMT_RGB8p : nvcv::FMT_RGB8;
    const int                         planes     = planar ? 3 : 1;
    const int                         pixelBytes = planar ? 1 : 3;
    const int                         shiftBytes = aliasMode == 2 ? 64 * pixelBytes : 0;
    const std::array<nvcv::Size2D, 3> inputSizes{
        size, size, nvcv::Size2D{31, 17}
    };
    const std::array<nvcv::Size2D, 3> outputSizes{
        size, size, nvcv::Size2D{65, 33}
    };
    std::array<nvcv::ImageDataStridedCuda::Buffer, 4>           buffers{};
    std::array<std::array<size_t, 3>, 4>                        offsets{};
    std::array<std::vector<uint8_t>, 4>                         expected;
    std::array<std::vector<uint8_t>, 3>                         snapshots;
    std::vector<std::unique_ptr<NVCVByte, decltype(&cudaFree)>> allocations;

    for (int storage = 0; storage < 4; ++storage)
    {
        const auto dimensions = storage == 3 ? outputSizes[2] : inputSizes[storage];
        auto      &buffer     = buffers[storage];
        buffer.numPlanes      = planes;
        size_t bytes          = 5 + storage;
        for (int plane = 0; plane < planes; ++plane)
        {
            offsets[storage][plane] = bytes;
            auto &descriptor        = buffer.planes[plane];
            descriptor.width        = dimensions.w;
            descriptor.height       = dimensions.h;
            descriptor.rowStride    = dimensions.w * pixelBytes + shiftBytes + 17 + plane * 3 + storage;
            bytes += descriptor.rowStride * dimensions.h + 23;
        }
        expected[storage].assign(bytes, 0xA5);
        if (storage < 3)
        {
            auto &pixels = snapshots[storage];
            pixels.resize(static_cast<size_t>(dimensions.w) * dimensions.h * 3);
            PillowGoldFillSample(pixels, storage + 7);
            StorePillowRgbPixels(expected[storage], pixels, dimensions, planar, buffer, offsets[storage]);
        }
        NVCVByte *allocation = nullptr;
        ASSERT_EQ(cudaSuccess, cudaMalloc(&allocation, bytes));
        allocations.emplace_back(allocation, &cudaFree);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(allocation, expected[storage].data(), bytes, cudaMemcpyHostToDevice));
        for (int plane = 0; plane < planes; ++plane)
            buffer.planes[plane].basePtr = allocation + offsets[storage][plane];
    }

    nvcv::ImageBatchVarShape source(3);
    nvcv::ImageBatchVarShape destination(3);
    for (int sample = 0; sample < 3; ++sample)
    {
        source.pushBack(nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{format, buffers[sample]}));
        int target = sample;
        if (sample == 2)
            target = 3;
        else if (aliasMode == 0)
            target = 1 - sample;
        auto output = MakePillowAliasOutput(buffers[target], sample, aliasMode, shiftBytes);
        auto gold   = PillowGoldResampleSample(snapshots[sample], inputSizes[sample].w, inputSizes[sample].h,
                                               outputSizes[sample].w, outputSizes[sample].h, 3, interpolation);
        std::array<size_t, 3> outputOffsets{};
        for (int plane = 0; plane < planes; ++plane)
            outputOffsets[plane] = output.planes[plane].basePtr - allocations[target].get();
        StorePillowRgbPixels(expected[target], gold, outputSizes[sample], planar, output, outputOffsets);
        destination.pushBack(nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{format, output}));
    }

    cudaStream_t rawStream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&rawStream, cudaStreamNonBlocking));
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)> stream(rawStream,
                                                                                              &cudaStreamDestroy);
    cvcuda::PillowResize                                                               op;
    auto                                                                               workspace
        = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(3, inputSizes.data(), outputSizes.data(), format));
    ASSERT_NO_THROW(op(stream.get(), workspace.get(), source, destination, interpolation));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.get()));
    for (size_t storage = 0; storage < expected.size(); ++storage)
    {
        SCOPED_TRACE(storage);
        std::vector<uint8_t> actual(expected[storage].size());
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(actual.data(), allocations[storage].get(), actual.size(), cudaMemcpyDeviceToHost));
        EXPECT_EQ(expected[storage], actual);
    }
}

} // namespace

TEST(OpPillowResizeBitExact, varshape_rgb_cross_sample_alias_snapshot)
{
    for (bool planar : {false, true})
        for (auto interpolation : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
            for (auto size : {
                     nvcv::Size2D{  1,  1},
                     nvcv::Size2D{129, 65}
            })
            {
                SCOPED_TRACE(::testing::Message() << planar << ":" << interpolation << ":" << size.w);
                RunPillowVarShapeAliasCase(planar, interpolation, size, 0);
            }
}

TEST(OpPillowResizeBitExact, varshape_rgb_same_view_alias_snapshot)
{
    for (bool planar : {false, true})
        for (auto interpolation : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
            RunPillowVarShapeAliasCase(planar, interpolation, {129, 65}, 1);
}

TEST(OpPillowResizeBitExact, varshape_rgb_shifted_alias_mixed_batch_snapshot)
{
    for (bool planar : {false, true})
        for (auto interpolation : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
            RunPillowVarShapeAliasCase(planar, interpolation, {129, 65}, 2);
}

TEST(OpPillowResizeBitExact, varshape_rgb_cross_plane_alias_snapshot)
{
    for (auto interpolation : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
        RunPillowVarShapeAliasCase(true, interpolation, {129, 65}, 3);
}

TEST(OpPillowResizeBitExact, linear_2x_large_coordinate)
{
    // At output2^23, float(outX)+0.5 loses its fraction. Constant quarter taps
    // must fall back to the established coefficient calculation for this valid width.
    constexpr int srcX      = 1 << 22;
    constexpr int srcW      = srcX + 2;
    constexpr int outX      = 2 * srcX;
    nvcv::Tensor  src       = nvcv::util::CreateTensor(1, srcW, 1, nvcv::FMT_RGB8);
    nvcv::Tensor  dst       = nvcv::util::CreateTensor(1, 2 * srcW, 2, nvcv::FMT_RGB8);
    auto          srcData   = src.exportData<nvcv::TensorDataStridedCuda>();
    auto          dstData   = dst.exportData<nvcv::TensorDataStridedCuda>();
    auto          srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto          dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(srcAccess && dstAccess);
    ASSERT_EQ(cudaSuccess, cudaMemset(srcAccess->sampleData(0), 0, srcAccess->sampleStride()));
    const std::array<uint8_t, 3> white{255, 255, 255};
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(srcAccess->sampleData(0) + 3 * srcX, white.data(), white.size(), cudaMemcpyHostToDevice));
    cvcuda::PillowResize    op;
    cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(
        op.getWorkspaceRequirements(1, nvcv::Size2D{srcW, 1}, nvcv::Size2D{2 * srcW, 2}, nvcv::FMT_RGB8));
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    ASSERT_NO_THROW(op(stream, ws.get(), src, dst, NVCV_INTERP_LINEAR));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    std::array<uint8_t, 3> actual;
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(actual.data(), dstAccess->sampleData(0) + 3 * outX, actual.size(), cudaMemcpyDeviceToHost));
    const float   center = (static_cast<float>(outX) + 0.5f) * 0.5f;
    const uint8_t gold   = PillowGoldQuantize(255.0f * PillowGoldBilinearFilter(srcX - center + 0.5f));
    for (uint8_t value : actual) EXPECT_EQ(value, gold);
}

TEST(OpPillowResizeBitExact, misaligned_destination)
{
    RunPillowBitExactCase(256, 8, 512, 16, 3, NVCV_INTERP_LINEAR, 2, false, 0, 0, 1);
}

TEST(OpPillowResizeBitExact, planar_single_channel_grouping)
{
    RunPillowBitExactCase(512, 24, 1024, 48, 1, NVCV_INTERP_LINEAR, 1, true);
    RunPillowBitExactCase(65, 49, 131, 99, 1, NVCV_INTERP_CUBIC, 1, true);
    RunPillowBitExactCase(55, 49, 109, 99, 1, NVCV_INTERP_CUBIC, 1, true);
}

TEST(OpPillowResizeBitExact, planar_rgb_cubic_expand_wide_rows)
{
    RunPillowBitExactCase(1920, 9, 3840, 18, 3, NVCV_INTERP_CUBIC, 2, true);
    RunPillowBitExactCase(1922, 9, 3844, 18, 3, NVCV_INTERP_CUBIC, 2, true);
    RunPillowBitExactCase(1921, 9, 3842, 18, 3, NVCV_INTERP_CUBIC, 2, true);
}

NVCV_TEST_SUITE_P(OpPillowResizePlanarCubicContract, test::ValueList<int, int, int, int>{
                                                         {1920, 27, 960, 13},
                                                         {1921, 27, 961, 13},
                                                         {1920, 27, 865, 11},
                                                         { 224, 17,  64,  9},
                                                         { 225, 17,  64,  9},
                                                         { 448, 17, 128,  9},
                                                         { 449, 17, 128,  9},
                                                         { 480, 17,  64,  9},
                                                         { 481, 17,  64,  9},
                                                         { 160, 17,  52,  9},
                                                         { 160, 17,  53,  9},
                                                         { 320, 17, 108,  9},
                                                         { 320, 17, 109,  9},
                                                         { 128, 31,  64,  1},
                                                         { 128, 32,  64,  1},
                                                         {   3,  3,   1,  1},
                                                         {   4,  3,   1,  1},
});

TEST_P(OpPillowResizePlanarCubicContract, aligned_rgb8_matches_scalar_oracle)
{
    RunPillowBitExactCase(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(), 3,
                          NVCV_INTERP_CUBIC, 2, true);
}

TEST_P(OpPillowResizePlanarCubicContract, padded_rgb8_matches_scalar_oracle)
{
    RunPillowBitExactCase(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(), 3,
                          NVCV_INTERP_CUBIC, 2, true, 3, 7, 7, 5, 11, 13);
}

TEST(OpPillowResizeBitExact, planar_rgb_cubic_contract_padded_chw)
{
    RunPillowBitExactCase(1921, 27, 961, 13, 3, NVCV_INTERP_CUBIC, 1, true, 3, 7, 7, 5, 11, 13, true);
}

TEST(OpPillowResizeBitExact, planar_rgb_cubic_contract_aligned_chw)
{
    RunPillowBitExactCase(1921, 27, 961, 13, 3, NVCV_INTERP_CUBIC, 1, true, 0, 0, 0, 0, 0, 0, true);
}

TEST(OpPillowResizeBitExact, planar_rgb_cubic_contract_two_byte_aligned)
{
    RunPillowBitExactCase(1920, 27, 960, 13, 3, NVCV_INTERP_CUBIC, 2, true, 2);
}

TEST(OpPillowResizeBitExact, planar_rgb_cubic_contract_odd_source_base)
{
    constexpr int           numImages = 2;
    constexpr nvcv::Size2D  srcSize{128, 18};
    constexpr nvcv::Size2D  dstSize{64, 9};
    const nvcv::TensorShape srcShape{
        {numImages, 3, srcSize.h, srcSize.w},
        "NCHW"
    };
    const nvcv::TensorShape dstShape{
        {numImages, 3, dstSize.h, dstSize.w},
        "NCHW"
    };

    nvcv::TensorDataStridedCuda::Buffer buffer{};
    buffer.strides[3] = 1;
    buffer.strides[2] = srcSize.w;
    buffer.strides[1] = buffer.strides[2] * srcSize.h;
    buffer.strides[0] = buffer.strides[1] * 3;
    test::DeviceBuffer<NVCVByte> allocation(buffer.strides[0] * numImages + 1);
    ASSERT_NE(nullptr, allocation.get());
    buffer.basePtr = allocation.get() + 1;

    nvcv::Tensor src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{srcShape, nvcv::TYPE_U8, buffer});
    nvcv::Tensor dst(dstShape, nvcv::TYPE_U8);
    auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto         dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, srcData);
    ASSERT_NE(nullptr, dstData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(srcAccess && dstAccess);
    ASSERT_EQ(1U, reinterpret_cast<std::uintptr_t>(srcAccess->sampleData(0)) % 2);
    ASSERT_EQ(0, srcAccess->rowStride() % 2);
    ASSERT_EQ(0, srcAccess->chStride() % 2);
    ASSERT_EQ(0, srcAccess->sampleStride() % 2);

    std::vector<std::vector<uint8_t>> goldHWC(numImages);
    for (int sample = 0; sample < numImages; ++sample)
    {
        std::vector<uint8_t> srcHWC(srcSize.h * srcSize.w * 3);
        PillowGoldFillSample(srcHWC, sample);
        goldHWC[sample]
            = PillowGoldResampleSample(srcHWC, srcSize.w, srcSize.h, dstSize.w, dstSize.h, 3, NVCV_INTERP_CUBIC);
        test::planar::UploadPlanarSample(*srcAccess, sample,
                                         test::planar::DeinterleaveToPlanes(srcHWC, srcSize.w, srcSize.h, 3, 1),
                                         srcSize.w, srcSize.h, 3, 1);
    }

    test::StreamGuard stream;
    ASSERT_NE(nullptr, stream.get());
    cvcuda::PillowResize op;
    auto                 workspace = cvcuda::AllocateWorkspace(
                        op.getWorkspaceRequirements(numImages, srcSize, dstSize, PillowGoldFormat(3, true)));
    ASSERT_NO_THROW(op(stream.get(), workspace.get(), src, dst, NVCV_INTERP_CUBIC));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.get()));

    for (int sample = 0; sample < numImages; ++sample)
    {
        SCOPED_TRACE(sample);
        const auto actual = test::planar::InterleaveFromPlanes(
            test::planar::DownloadPlanarSample(*dstAccess, sample, dstSize.w, dstSize.h, 3, 1), dstSize.w, dstSize.h, 3,
            1);
        ASSERT_EQ(goldHWC[sample], actual);
    }
}

TEST(OpPillowResizeBitExact, planar_cubic_tall_vertical_support)
{
    RunPillowBitExactCase(64, 128, 128, 4, 3, NVCV_INTERP_CUBIC, 2, true);
}

TEST(OpPillowResizeBitExact, planar_padded_nchw_and_chw)
{
    constexpr int srcRowPad    = 3;
    constexpr int srcPlanePad  = 5;
    constexpr int srcSamplePad = 7;
    constexpr int dstRowPad    = 7;
    constexpr int dstPlanePad  = 11;
    constexpr int dstSamplePad = 13;
    for (const int channels : {1, 3, 4})
    {
        SCOPED_TRACE(channels);
        RunPillowBitExactCase(33, 9, 65, 5, channels, NVCV_INTERP_LINEAR, 2, true, srcRowPad, srcSamplePad, dstRowPad,
                              srcPlanePad, dstPlanePad, dstSamplePad);
        RunPillowBitExactCase(33, 7, 65, 11, channels, NVCV_INTERP_CUBIC, 1, true, srcRowPad, srcSamplePad, dstRowPad,
                              srcPlanePad, dstPlanePad, dstSamplePad, true);
        RunPillowBitExactCase(55, 9, 109, 5, channels, NVCV_INTERP_LINEAR, 2, true, srcRowPad, srcSamplePad, dstRowPad,
                              srcPlanePad, dstPlanePad, dstSamplePad);
        RunPillowBitExactCase(55, 7, 109, 11, channels, NVCV_INTERP_CUBIC, 1, true, srcRowPad, srcSamplePad, dstRowPad,
                              srcPlanePad, dstPlanePad, dstSamplePad, true);
    }
}

NVCV_TEST_SUITE_P(OpPillowResizePlanarByteTails, test::ValueList<int, int, int, int, int, nvcv::DataKind>{
                                                     {17, 7,  33, 7, 1, nvcv::DataKind::UNSIGNED},
                                                     {49, 5,  97, 9, 1,   nvcv::DataKind::SIGNED},
                                                     {33, 9,  65, 5, 3, nvcv::DataKind::UNSIGNED},
                                                     {49, 9,  97, 5, 3,   nvcv::DataKind::SIGNED},
                                                     {65, 9, 129, 5, 4, nvcv::DataKind::UNSIGNED},
                                                     {33, 7,  65, 7, 4,   nvcv::DataKind::SIGNED},
                                                     {55, 7, 109, 7, 1, nvcv::DataKind::UNSIGNED},
                                                     {55, 5, 109, 9, 1,   nvcv::DataKind::SIGNED},
                                                     {55, 9, 109, 5, 3, nvcv::DataKind::UNSIGNED},
                                                     {55, 9, 109, 5, 3,   nvcv::DataKind::SIGNED},
                                                     {55, 9, 109, 5, 4, nvcv::DataKind::UNSIGNED},
                                                     {55, 7, 109, 7, 4,   nvcv::DataKind::SIGNED},
});

TEST_P(OpPillowResizePlanarByteTails, all_filters_match_interleaved)
{
    const int               srcW           = GetParamValue<0>();
    const int               srcH           = GetParamValue<1>();
    const int               dstW           = GetParamValue<2>();
    const int               dstH           = GetParamValue<3>();
    const int               channels       = GetParamValue<4>();
    const nvcv::DataKind    kind           = GetParamValue<5>();
    const nvcv::ImageFormat planarFmt      = PillowGoldFormat(channels, true).dataKind(kind);
    const nvcv::ImageFormat interleavedFmt = PillowGoldFormat(channels, false).dataKind(kind);

    constexpr std::array<NVCVInterpolationType, 5> interpolations{
        NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC, NVCV_INTERP_BOX, NVCV_INTERP_HAMMING, NVCV_INTERP_LANCZOS};
    for (const auto interpolation : interpolations)
    {
        SCOPED_TRACE(static_cast<int>(interpolation));
        auto invoke = [srcW, srcH, dstW, dstH, interpolation](cudaStream_t stream, const nvcv::Tensor &src,
                                                              const nvcv::Tensor &dst, nvcv::ImageFormat fmt)
        {
            cvcuda::PillowResize    op;
            cvcuda::UniqueWorkspace ws
                = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(2, {srcW, srcH}, {dstW, dstH}, fmt));
            EXPECT_NO_THROW(op(stream, ws.get(), src, dst, interpolation));
        };
        if (channels == 1)
        {
            test::planar::RunTensorSingleChannelLayoutParity(
                srcW, srcH, dstW, dstH, 2, planarFmt.planeDataType(0),
                [&invoke, &planarFmt](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst)
                { invoke(stream, src, dst, planarFmt); });
            continue;
        }
        test::planar::RunTensorParity(
            planarFmt, interleavedFmt, srcW, srcH, dstW, dstH, 2, invoke,
            [kind](int sample, int width, int height, int numChannels)
            { return PillowPlanarByteTailSample(kind, sample, width, height, numChannels); },
            [](const std::vector<uint8_t> &interleaved, const std::vector<uint8_t> &planar)
            { EXPECT_EQ(interleaved, planar); });
    }
}

// srcW, srcH, dstW, dstH, channels, interpolation, srcRowPad, srcSamplePad
//
// Padded row and sample strides drive the same oracle through the launch sites' alignment gates.
// Independently padding both proves neither row nor sample addressing is inferred from the logical
// image extent. Four-byte pads keep the widened and packed accesses eligible; one- and two-byte
// pads defeat them and select the narrow and scalar fallbacks instead.
NVCV_TEST_SUITE_P(OpPillowResizeBitExactPitched,
                  test::ValueList<int, int, int, int, int, NVCVInterpolationType, int, int>{
  // The profiled 1920 -> 864 horizontal contraction, at packed, aligned-pitched and
  // misaligned-pitched strides. Equal source and destination heights make the vertical pass an
  // identity, isolating the horizontal contraction.
                      {1920, 9,  864,  9, 3,  NVCV_INTERP_CUBIC, 0,  0},
                      {1920, 9,  864,  9, 3,  NVCV_INTERP_CUBIC, 8, 16},
                      {1920, 9,  864,  9, 3,  NVCV_INTERP_CUBIC, 1, 16},
                      {1920, 9,  960,  9, 3,  NVCV_INTERP_CUBIC, 8, 16},
                      {1920, 9,  960,  9, 3,  NVCV_INTERP_CUBIC, 8,  2},
 // An odd output width leaves a partial pair for the paired horizontal pass and a partial run
  // for the widened one.
                      {1920, 9,  865,  9, 3,  NVCV_INTERP_CUBIC, 0,  0},
                      {1920, 9,  833,  9, 3,  NVCV_INTERP_CUBIC, 1, 16},
 // Byte channel counts that take the widened vertical span and the per-thread horizontal run.
                      { 512, 8, 1024, 16, 1, NVCV_INTERP_LINEAR, 4,  8},
                      { 512, 8, 1024, 16, 4, NVCV_INTERP_LINEAR, 4,  8},
                      { 512, 8, 1024, 16, 4,  NVCV_INTERP_CUBIC, 1,  8},
 // Exact2x RGB8 source pitch need not be aligned; destination fallback is checked separately.
                      { 256, 8,  512, 16, 3, NVCV_INTERP_LINEAR, 1,  0},
});

TEST_P(OpPillowResizeBitExactPitched, u8_matches_scalar_oracle)
{
    RunPillowBitExactCase(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(),
                          GetParamValue<4>(), GetParamValue<5>(), 2, false, GetParamValue<6>(), GetParamValue<7>());
}

TEST(OpPillowResizePlanar, signed32_single_channel_tensor_layouts_match)
{
    constexpr int srcW = 37;
    constexpr int srcH = 29;
    constexpr int dstW = 23;
    constexpr int dstH = 17;

    test::planar::RunTensorSingleChannelLayoutParity(
        srcW, srcH, dstW, dstH, 1, nvcv::TYPE_S32,
        [](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst)
        {
            cvcuda::PillowResize    op;
            cvcuda::UniqueWorkspace ws
                = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(1, {srcW, srcH}, {dstW, dstH}, nvcv::FMT_S32));
            EXPECT_NO_THROW(op(stream, ws.get(), src, dst, NVCV_INTERP_LINEAR));
        });
}

TEST(OpPillowResizePlanar, s32_single_channel_tensor_filters_match_interleaved)
{
    // S32 is tensor-only; the odd output width also exercises the scalar vertical tail.
    constexpr std::array<NVCVInterpolationType, 5> interpolations{
        NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC, NVCV_INTERP_BOX, NVCV_INTERP_HAMMING, NVCV_INTERP_LANCZOS};
    for (const auto interpolation : interpolations)
    {
        SCOPED_TRACE(static_cast<int>(interpolation));
        test::planar::RunTensorSingleChannelLayoutParity(
            17, 13, 9, 7, 2, nvcv::TYPE_S32,
            [interpolation](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst)
            {
                cvcuda::PillowResize    op;
                cvcuda::UniqueWorkspace ws
                    = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(2, {17, 13}, {9, 7}, nvcv::FMT_S32));
                EXPECT_NO_THROW(op(stream, ws.get(), src, dst, interpolation));
            });
    }
}

TEST(OpPillowResizePlanar, rgba_float_varshape_all_filters_match_interleaved)
{
    constexpr std::array<NVCVInterpolationType, 5> interpolations{
        NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC, NVCV_INTERP_BOX, NVCV_INTERP_HAMMING, NVCV_INTERP_LANCZOS};
    for (const auto interpolation : interpolations)
    {
        SCOPED_TRACE(static_cast<int>(interpolation));
        RunPlanarParityVarShapeCase(nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32, 50, 40, 100, 80, interpolation, 2);
    }
}

TEST(OpPillowResizePlanar, rgb_float_varshape_odd_width_all_filters_match_interleaved)
{
    constexpr std::array<NVCVInterpolationType, 5> interpolations{
        NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC, NVCV_INTERP_BOX, NVCV_INTERP_HAMMING, NVCV_INTERP_LANCZOS};
    for (const auto interpolation : interpolations)
    {
        SCOPED_TRACE(static_cast<int>(interpolation));
        // The odd RGB output width makes the non-fused VarShape vertical vector launch visit its row tail.
        RunPlanarParityVarShapeCase(nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, 49, 37, 101, 83, interpolation, 2);
    }
}

TEST(OpPillowResizePlanar, rgba_float_tensor_all_filters_match_interleaved)
{
    constexpr std::array<NVCVInterpolationType, 5> interpolations{
        NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC, NVCV_INTERP_BOX, NVCV_INTERP_HAMMING, NVCV_INTERP_LANCZOS};
    for (const auto interpolation : interpolations)
    {
        SCOPED_TRACE(static_cast<int>(interpolation));
        RunPlanarParityTensorCase(nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32, 50, 40, 100, 80, interpolation, 2);
    }
}

TEST(OpPillowResizePlanar, filter_boundary_paths_match_interleaved)
{
    // These ratios make the coefficient precompute visit negative, interior, and out-of-support
    // arguments in each filter while the shared parity helper exercises both pointer layouts.
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 2, 2, 6, 6, NVCV_INTERP_LINEAR, 1);
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 2, 2, 41, 41, NVCV_INTERP_BOX, 1);
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 2, 2, 6, 6, NVCV_INTERP_HAMMING, 1);
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 3, 3, 9, 9, NVCV_INTERP_CUBIC, 1);
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 4, 4, 12, 12, NVCV_INTERP_LANCZOS, 1);
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
// follows the Resize/Normalize convention of disallowing 2-channel planar.
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

TEST(OpPillowResize_Negative, planar_channel_mismatch_rejected)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src(
        {
            {1, 3, 24, 24},
            "NCHW"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor dst(
        {
            {1, 4, 12, 12},
            "NCHW"
    },
        nvcv::TYPE_U8);

    cvcuda::PillowResize    op;
    cvcuda::UniqueWorkspace ws
        = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(1, {24, 24}, {12, 12}, nvcv::FMT_RGBA8));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(stream, ws.get(), src, dst, NVCV_INTERP_LINEAR); }));

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

TEST(OpPillowResize_Negative, rejects_non_image_tensor)
{
    cvcuda::PillowResize op;
    nvcv::Tensor         inTensor{
        {{16}, "N"},
        nvcv::TYPE_U8
    };
    nvcv::Tensor outTensor{
        {{16}, "N"},
        nvcv::TYPE_U8
    };

    cvcuda::UniqueWorkspace ws
        = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(1, {1, 1}, {1, 1}, nvcv::FMT_U8));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(nullptr, ws.get(), inTensor, outTensor, NVCV_INTERP_LINEAR); }));
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
    catch (const std::exception &e)
    {
        // AllocateWorkspace calls the default allocator callback directly, whose CUDA failure escapes as the
        // private nvcv::priv::Exception rather than the public nvcv::Exception.
        if (std::strstr(e.what(), "NVCV_ERROR_OUT_OF_MEMORY") != nullptr)
        {
            GTEST_SKIP() << "insufficient device memory for oversized var-shape input: " << e.what();
        }
        throw;
    }
}

// The interleaved var-shape kernel uses grid.z = numImages (pass_z = batch) — the same as the
// planar kernel, which was already guarded. Before the fix the guard was conditioned on
// isPlanar; an interleaved batch of > 65535 images would reach the kernel launch and fail with
// cudaErrorInvalidConfiguration rather than a clean diagnostic. The fix removes the isPlanar
// condition so the guard covers both layouts.
//
// This test exercises the interleaved path (FMT_RGB8). The same check now also rejects planar
// batches, but the original planar guard covered that case before the fix too.
//
// Construction: one backing image (1 × 1 RGB8) is shared across 65 536 logical batch entries
// via handle copies. This avoids 65 536 separate CUDA allocations while giving the batch the
// required numImages() == 65536. Because the guard fires synchronously at operator() call
// time — before any kernel launch or workspace touch — the shared backing image does not
// affect correctness of the test.
TEST(OpPillowResize_Negative, varshape_batch_over_65535_rejected)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int kNumImages = 65536; // one over the CUDA grid-z limit of 65535

    nvcv::Image              oneImage(nvcv::Size2D{1, 1}, nvcv::FMT_RGB8);
    std::vector<nvcv::Image> images(kNumImages, oneImage);

    nvcv::ImageBatchVarShape batchSrc(kNumImages);
    nvcv::ImageBatchVarShape batchDst(kNumImages);
    batchSrc.pushBack(images.begin(), images.end());
    batchDst.pushBack(images.begin(), images.end());

    std::vector<nvcv::Size2D> sizes(kNumImages, nvcv::Size2D{1, 1});

    cvcuda::PillowResize    pillowResizeOp;
    cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(
        pillowResizeOp.getWorkspaceRequirements(kNumImages, sizes.data(), sizes.data(), nvcv::FMT_RGB8));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { pillowResizeOp(stream, ws.get(), batchSrc, batchDst, NVCV_INTERP_LINEAR); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// The tensor kernel uses grid.z = numSamples (in_N); a batch exceeding the CUDA grid-z limit
// of 65535 would previously reach the kernel launch and fail with
// cudaErrorInvalidConfiguration. Before the fix no guard existed for the tensor path; the
// var-shape guard existed only for the planar layout. This test verifies that the tensor guard
// added by the fix fires at submission time and reports NVCV_ERROR_INVALID_ARGUMENT.
//
// A 65 536 × 1 × 1 × RGB8 tensor = 192 KiB; the workspace for a uniform 1 × 1 → 1 × 1
// resize is a few MB of coefficient and intermediate buffers, well within device limits.
TEST(OpPillowResize_Negative, tensor_batch_over_65535_rejected)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int kNumSamples = 65536; // one over the CUDA grid-z limit of 65535
    nvcv::Tensor  src(kNumSamples, {1, 1}, nvcv::FMT_RGB8);
    nvcv::Tensor  dst(kNumSamples, {1, 1}, nvcv::FMT_RGB8);

    cvcuda::PillowResize    pillowResizeOp;
    cvcuda::UniqueWorkspace ws = cvcuda::AllocateWorkspace(
        pillowResizeOp.getWorkspaceRequirements(kNumSamples, {1, 1}, {1, 1}, nvcv::FMT_RGB8));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { pillowResizeOp(stream, ws.get(), src, dst, NVCV_INTERP_LINEAR); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Hardening: batch comes from the input and is used as the launch z extent against the output, so a
// shorter output batch used to have a missing destination image indexed.
TEST(OpPillowResize_Negative, rejects_varshape_image_count_mismatch)
{
    cvcuda::PillowResize op;

    nvcv::ImageBatchVarShape src(3);
    nvcv::ImageBatchVarShape dst(2);
    for (int i = 0; i < 3; ++i)
    {
        src.pushBack(nvcv::Image(nvcv::Size2D{16, 12}, nvcv::FMT_RGB8));
    }
    for (int i = 0; i < 2; ++i)
    {
        dst.pushBack(nvcv::Image(nvcv::Size2D{8, 6}, nvcv::FMT_RGB8));
    }

    cvcuda::UniqueWorkspace ws
        = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(3, {16, 12}, {16, 12}, nvcv::FMT_RGB8));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(nullptr, ws.get(), src, dst, NVCV_INTERP_LINEAR); }));
}
