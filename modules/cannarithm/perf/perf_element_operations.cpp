// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/cann_arithm.hpp"

namespace opencv_test
{
namespace
{

#define ARITHM_MAT_DEPTH Values(CV_32S, CV_32SC3)
#define TYPICAL_ACL_MAT_SIZES ::perf::sz1080p, ::perf::sz2K, ::perf::sz2160p, ::perf::sz4320p
#define DEVICE_ID 0
#define DEF_PARAM_TEST(name, ...) \
    typedef ::perf::TestBaseWithParam<testing::tuple<__VA_ARGS__>> name

// NPU Perf Test
DEF_PARAM_TEST(NPU, cv::Size, perf::MatDepth);
#define TEST_NPU_OP_MAT(idx, op, ...)                                                       \
    PERF_TEST_P(NPU, MAT_##op##_MAT_##idx,                                                  \
                testing::Combine(testing::Values(TYPICAL_ACL_MAT_SIZES), ARITHM_MAT_DEPTH)) \
    {                                                                                       \
        Size size = GET_PARAM(0);                                                           \
        int depth = GET_PARAM(1);                                                           \
                                                                                            \
        Mat src1(size, depth), src2(size, depth);                                           \
        declare.in(src1, WARMUP_RNG);                                                       \
        declare.in(src2, WARMUP_RNG);                                                       \
        cv::cann::setDevice(DEVICE_ID);                                                     \
                                                                                            \
        AclMat npu_src1, npu_src2, dst;                                                     \
        npu_src1.upload(src1);                                                              \
        npu_src2.upload(src2);                                                              \
        AclStream stream;                                                                   \
        TEST_CYCLE() { cv::cann::op(npu_src1, npu_src2, dst, __VA_ARGS__); }                \
        SANITY_CHECK_NOTHING();                                                             \
        cv::cann::resetDevice();                                                            \
    }

// CPU Perf Test
DEF_PARAM_TEST(CPU, cv::Size, perf::MatDepth);
#define TEST_CPU_OP_MAT(idx, op, ...)                                                       \
    PERF_TEST_P(CPU, MAT_##op##_MAT_##idx,                                                  \
                testing::Combine(testing::Values(TYPICAL_ACL_MAT_SIZES), ARITHM_MAT_DEPTH)) \
    {                                                                                       \
        Size size = GET_PARAM(0);                                                           \
        int depth = GET_PARAM(1);                                                           \
                                                                                            \
        Mat src1(size, depth), src2(size, depth), dst(size, depth);                         \
        declare.in(src1, WARMUP_RNG);                                                       \
        declare.in(src2, WARMUP_RNG);                                                       \
                                                                                            \
        TEST_CYCLE() cv::op(src1, src2, dst, __VA_ARGS__);                                  \
        SANITY_CHECK_NOTHING();                                                             \
    }

TEST_NPU_OP_MAT(1, add, noArray(), -1);
TEST_CPU_OP_MAT(1, add, noArray(), -1);

TEST_NPU_OP_MAT(1, subtract, noArray(), -1);
TEST_CPU_OP_MAT(1, subtract, noArray(), -1);

TEST_NPU_OP_MAT(1, multiply, 1, -1);
TEST_CPU_OP_MAT(1, multiply, 1, -1);

TEST_NPU_OP_MAT(1, divide, 1, -1);
TEST_CPU_OP_MAT(1, divide, 1, -1);

TEST_NPU_OP_MAT(1, bitwise_and, noArray());
TEST_CPU_OP_MAT(1, bitwise_and, noArray());

TEST_NPU_OP_MAT(1, bitwise_or, noArray());
TEST_CPU_OP_MAT(1, bitwise_or, noArray());

TEST_NPU_OP_MAT(1, bitwise_xor, noArray());
TEST_CPU_OP_MAT(1, bitwise_xor, noArray());

} // namespace
} // namespace opencv_test
