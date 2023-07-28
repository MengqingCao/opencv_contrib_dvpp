// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/ts/cuda_test.hpp"
#include "opencv2/cann_arithm.hpp"

namespace opencv_test
{
namespace
{
// Random Generator
Mat randomMat(int w, int h, int dtype)
{
    Mat rnMat(w, h, dtype);
    RNG rng;
    rng.fill(rnMat, RNG::UNIFORM, 0.f, 1.f);
    return rnMat;
}
cv::Scalar randomScalar()
{
    RNG rng;
    return Scalar(rng, rng.next(), rng.next(), rng.next());
}
float randomNum()
{
    RNG rng;
    float rdnNum = float(rng.uniform(0.3, 3.0));
    return rdnNum;
}
Mat genMask()
{
    Mat mask = Mat::zeros(Size(10, 10), CV_8UC1);
    rectangle(mask, cv::Rect(5, 5, 3, 3), Scalar(255), -1);
    return mask;
}

#define DEVICE_ID 0

/****************TEST CASE***************/
// MAT & Mat
#define TEST_MAT_OP_MAT(idx, op, ...)                        \
    TEST(ELEMENTWISE_OP, MAT_##op##_MAT_##idx)               \
    {                                                        \
        cv::cann::setDevice(DEVICE_ID);                      \
                                                             \
        Mat cpuMat1 = randomMat(10, 10, CV_32SC3);           \
        Mat cpuMat2 = randomMat(10, 10, CV_32SC3);           \
        Mat cpuDst;                                          \
        cv::op(cpuMat1, cpuMat2, cpuDst, __VA_ARGS__);       \
                                                             \
        AclMat mat1, mat2;                                   \
        mat1.upload(cpuMat1);                                \
        mat2.upload(cpuMat2);                                \
        AclMat dst, dstS;                                    \
        cv::cann::op(mat1, mat2, dst, __VA_ARGS__);          \
        Mat npuDst, npuDstS;                                 \
        dst.download(npuDst);                                \
        AclStream stream;                                    \
        cv::cann::op(mat1, mat2, dstS, __VA_ARGS__, stream); \
        stream.waitForCompletion();                          \
        dstS.download(npuDstS);                              \
                                                             \
        EXPECT_MAT_NEAR(npuDst, cpuDst, 0.0);                \
        EXPECT_MAT_NEAR(npuDst, npuDstS, 0.0);               \
        cv::cann::resetDevice();                             \
    }

TEST_MAT_OP_MAT(1, add, noArray(), -1);
TEST_MAT_OP_MAT(1, subtract, noArray(), -1);
TEST_MAT_OP_MAT(1, multiply, 1, -1);
TEST_MAT_OP_MAT(1, divide, 1, -1);
TEST_MAT_OP_MAT(1, bitwise_and, noArray());
TEST_MAT_OP_MAT(1, bitwise_or, noArray());
TEST_MAT_OP_MAT(1, bitwise_xor, noArray());

TEST_MAT_OP_MAT(2, add, genMask(), CV_32SC3);
TEST_MAT_OP_MAT(2, subtract, genMask(), CV_32SC3);
TEST_MAT_OP_MAT(2, multiply, randomNum(), -1);
TEST_MAT_OP_MAT(2, divide, randomNum(), -1);
TEST_MAT_OP_MAT(2, bitwise_and, genMask());
TEST_MAT_OP_MAT(2, bitwise_or, genMask());
TEST_MAT_OP_MAT(2, bitwise_xor, genMask());

// SCALAR & MAT
#define TEST_MAT_OP_SCALAR(idx, op, ...)                           \
    TEST(ELEMENTWISE_OP, MAT_##op##_SCALAR_##idx)                  \
    {                                                              \
        Scalar cpuS1 = randomScalar();                             \
        Scalar cpuS2 = randomScalar();                             \
        Mat cpuMatS1(10, 10, CV_32SC3, cpuS1);                     \
        Mat cpuMatS2(10, 10, CV_32SC3, cpuS2);                     \
        Mat cpuDst, cpuDstC;                                       \
        cv::op(cpuMatS1, cpuMatS2, cpuDst, __VA_ARGS__);           \
        cv::op(cpuMatS2, cpuMatS1, cpuDstC, __VA_ARGS__);          \
        cv::cann::setDevice(DEVICE_ID);                            \
                                                                   \
        AclMat mat;                                                \
        mat.upload(cpuMatS2);                                      \
        AclMat dst, dstS, dstC, dstCS;                             \
        cv::cann::op(cpuS1, cpuMatS2, dst, __VA_ARGS__);           \
        cv::cann::op(cpuMatS2, cpuS1, dstC, __VA_ARGS__);          \
        Mat npuDst, npuDstS, npuDstC, npuDstCS;                    \
        dst.download(npuDst);                                      \
        dstC.download(npuDstC);                                    \
        AclStream stream;                                          \
        cv::cann::op(cpuS1, cpuMatS2, dstS, __VA_ARGS__, stream);  \
        cv::cann::op(cpuMatS2, cpuS1, dstCS, __VA_ARGS__, stream); \
        stream.waitForCompletion();                                \
        dstS.download(npuDstS);                                    \
        dstCS.download(npuDstCS);                                  \
                                                                   \
        EXPECT_MAT_NEAR(npuDst, npuDstS, 0.0);                     \
        EXPECT_MAT_NEAR(npuDst, cpuDst, 0.0);                      \
        EXPECT_MAT_NEAR(npuDstC, npuDstCS, 0.0);                   \
        EXPECT_MAT_NEAR(npuDstC, cpuDstC, 0.0);                    \
                                                                   \
        cv::cann::resetDevice();                                   \
    }
TEST_MAT_OP_SCALAR(1, add, noArray(), -1);
TEST_MAT_OP_SCALAR(1, subtract, noArray(), -1);
TEST_MAT_OP_SCALAR(1, multiply, 1, -1);
TEST_MAT_OP_SCALAR(1, divide, 1, -1);
TEST_MAT_OP_SCALAR(1, bitwise_and, noArray());
TEST_MAT_OP_SCALAR(1, bitwise_or, noArray());
TEST_MAT_OP_SCALAR(1, bitwise_xor, noArray());

TEST_MAT_OP_SCALAR(2, add, genMask(), CV_32SC3);
TEST_MAT_OP_SCALAR(2, subtract, genMask(), CV_32SC3);
TEST_MAT_OP_SCALAR(2, bitwise_and, genMask());
TEST_MAT_OP_SCALAR(2, bitwise_or, genMask());
TEST_MAT_OP_SCALAR(2, bitwise_xor, genMask());
TEST_MAT_OP_SCALAR(2, multiply, randomNum(), -1);
TEST_MAT_OP_SCALAR(2, divide, randomNum(), -1);
} // namespace
} // namespace opencv_test
