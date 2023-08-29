// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/cann_interface.hpp"

namespace opencv_test
{
namespace
{
#define TYPICAL_NPU_MAT_SIZES \
    Values(::perf::sz1080p, ::perf::sz2K, ::perf::sz2160p, ::perf::sz4320p)
#define DEF_PARAM_TEST(name, ...) \
    typedef ::perf::TestBaseWithParam<testing::tuple<__VA_ARGS__>> name

DEF_PARAM_TEST(NPU, Size);
DEF_PARAM_TEST(CPU, Size);

PERF_TEST_P(NPU, MERGE, TYPICAL_NPU_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC1);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    NpuMat npuMat[3];
    npuMat[0].upload(mat);
    npuMat[1].upload(mat);
    npuMat[2].upload(mat);

    TEST_CYCLE() { cv::cann::merge(&npuMat[0], 3, dst); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MERGE, TYPICAL_NPU_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC1);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    Mat mats[3] = {mat, mat, mat};
    TEST_CYCLE() { cv::merge(&mats[0], 3, dst); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, SPLIT, TYPICAL_NPU_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    NpuMat npuMat[3];

    TEST_CYCLE() { cv::cann::split(mat, &npuMat[0]); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, SPLIT, TYPICAL_NPU_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    declare.in(mat, WARMUP_RNG);
    Mat mats[3] = {mat, mat, mat};
    TEST_CYCLE() { cv::split(mat, &mats[0]); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, TRANSPOSE, TYPICAL_NPU_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::transpose(mat, dst); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, TRANSPOSE, TYPICAL_NPU_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE() { cv::transpose(mat, dst); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, FLIP, TYPICAL_NPU_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::flip(mat, dst, -1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, FLIP, TYPICAL_NPU_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE() { cv::flip(mat, dst, -1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, ROTATE, TYPICAL_NPU_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { cv::cann::rotate(mat, dst, 1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, ROTATE, TYPICAL_NPU_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE() { cv::rotate(mat, dst, 1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, CROP, TYPICAL_NPU_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    Rect b(1, 2, 4, 4);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE() { NpuMat cropped_cann(mat, b); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, CROP, TYPICAL_NPU_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    Rect b(1, 2, 4, 4);
    TEST_CYCLE() { Mat cropped_cv(mat, b); }
    SANITY_CHECK_NOTHING();
}

} // namespace
} // namespace opencv_test
