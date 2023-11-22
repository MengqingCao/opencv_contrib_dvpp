// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/cann_interface.hpp"

namespace opencv_test
{
namespace
{

#define ARITHM_MAT_DEPTH Values(CV_32S, CV_32SC3)
#define TYPICAL_ASCEND_MAT_SIZES \
    Values(::perf::sz1080p, ::perf::sz2K, ::perf::sz2160p, ::perf::sz4320p)
#define DEF_PARAM_TEST(name, ...) \
    typedef ::perf::TestBaseWithParam<testing::tuple<__VA_ARGS__>> name

DEF_PARAM_TEST(NPU, Size, int);
DEF_PARAM_TEST(CPU, Size, int);

PERF_TEST_P(NPU, MAT_ADD_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    AscendMat dst, mask;
    AscendMat src1, src2;
    src1.upload(mat1);
    src2.upload(mat2);
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::add(src1, src2, dst, mask, -1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_ADD_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::add(mat1, mat2, dst, noArray(), -1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MAT_SUB_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    AscendMat dst, mask;
    AscendMat src1, src2;
    src1.upload(mat1);
    src2.upload(mat2);
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::subtract(src1, src2, dst, mask, -1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_SUB_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::subtract(mat1, mat2, dst, noArray(), -1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MAT_MUL_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    AscendMat dst;
    AscendMat src1, src2;
    src1.upload(mat1);
    src2.upload(mat2);
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::multiply(src1, src2, dst, 1, -1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_MUL_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::multiply(mat1, mat2, dst, 1, -1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MAT_DIV_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    AscendMat dst;
    AscendMat src1, src2;
    src1.upload(mat1);
    src2.upload(mat2);
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::divide(src1, src2, dst, 1, -1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_DIV_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::divide(mat1, mat2, dst, 1, -1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MAT_BITWISE_AND_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    AscendMat dst;
    AscendMat src1, src2;
    src1.upload(mat1);
    src2.upload(mat2);
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::bitwise_and(src1, src2, dst); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_BITWISE_AND_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::bitwise_and(mat1, mat2, dst, noArray()); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MAT_BITWISE_OR_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    AscendMat dst;
    AscendMat src1, src2;
    src1.upload(mat1);
    src2.upload(mat2);
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::bitwise_or(src1, src2, dst); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_BITWISE_OR_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::bitwise_or(mat1, mat2, dst, noArray()); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MAT_BITWISE_XOR_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    AscendMat dst;
    AscendMat src1, src2;
    src1.upload(mat1);
    src2.upload(mat2);
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::bitwise_xor(src1, src2, dst); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_BITWISE_XOR_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat1(GET_PARAM(0), GET_PARAM(1));
    Mat mat2(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat1, WARMUP_RNG);
    declare.in(mat2, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::bitwise_xor(mat1, mat2, dst, noArray()); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MAT_BITWISE_NOT_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat(GET_PARAM(0), GET_PARAM(1));
    AscendMat dst, mask;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::bitwise_not(src, dst, mask); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MAT_BITWISE_NOT_MAT, testing::Combine(TYPICAL_ASCEND_MAT_SIZES, ARITHM_MAT_DEPTH))
{
    Mat mat(GET_PARAM(0), GET_PARAM(1));
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::bitwise_not(mat, dst, noArray()); }
    SANITY_CHECK_NOTHING();
}

} // namespace
} // namespace opencv_test
