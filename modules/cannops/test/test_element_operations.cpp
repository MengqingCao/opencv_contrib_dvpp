// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <iostream>

namespace opencv_test
{
namespace
{
template <typename FCV, typename FCANN, typename... PARAMS>
void testMatOpMat(FCV cvFunc, FCANN cannFunc, PARAMS... param)
{
    cv::cann::setDevice(DEVICE_ID);
    Mat mat1 = randomMat(10, 10, CV_32SC3);
    Mat mat2 = randomMat(10, 10, CV_32SC3);
    Mat cpuDst, check;

    cvFunc(mat1, mat2, cpuDst, param...);
    cannFunc(mat1, mat2, check, param..., AscendStream::Null());
    EXPECT_MAT_NEAR(cpuDst, check, 0.0);

    AscendStream stream;
    cannFunc(mat1, mat2, check, param..., stream);
    stream.waitForCompletion();
    EXPECT_MAT_NEAR(cpuDst, check, 0.0);

    cv::cann::resetDevice();
}

TEST(ELEMENTWISE_OP, MAT_ADD_MAT) { testMatOpMat(cv::add, cv::cann::add, noArray(), -1); }

TEST(ELEMENTWISE_OP, MAT_SUB_MAT) { testMatOpMat(cv::subtract, cv::cann::subtract, noArray(), -1); }

TEST(ELEMENTWISE_OP, MAT_MUL_MAT) { testMatOpMat(cv::multiply, cv::cann::multiply, 1, -1); }

/*
 * TODO cv::divide will round each element by cvRound while Ascend DIV op will floor each element.
 * In order to pass the testcase, using interger for all matrix and scalar, fixme after Ascend
 * support round element.
 */
/*
TEST(ELEMENTWISE_OP, MAT_DIV_MAT)
{

    testMatOpMat([](const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst, double scale, int dtype)
                 { cv::divide(src1, src2, dst, scale, dtype); },
                 cv::cann::divide, 1, -1);
}
*/

TEST(ELEMENTWISE_OP, MAT_BITWISE_AND_MAT)
{
    testMatOpMat(cv::bitwise_and, cv::cann::bitwise_and, noArray());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_OR_MAT)
{
    testMatOpMat(cv::bitwise_or, cv::cann::bitwise_or, noArray());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_XOR_MAT)
{
    testMatOpMat(cv::bitwise_xor, cv::cann::bitwise_xor, noArray());
}

TEST(ELEMENTWISE_OP, MAT_ADD_MAT_WITH_MASK_AND_DTYPE)
{
    testMatOpMat(cv::add, cv::cann::add, genMask(), CV_32SC3);
}

TEST(ELEMENTWISE_OP, MAT_SUB_MAT_WITH_MASK_AND_DTYPE)
{
    testMatOpMat(cv::subtract, cv::cann::subtract, genMask(), CV_32SC3);
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_AND_MAT_WITH_MASK)
{
    testMatOpMat(cv::bitwise_and, cv::cann::bitwise_and, genMask());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_OR_MAT_WITH_MASK)
{
    testMatOpMat(cv::bitwise_or, cv::cann::bitwise_or, genMask());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_XOR_MAT_WITH_MASK)
{
    testMatOpMat(cv::bitwise_xor, cv::cann::bitwise_xor, genMask());
}

/* Ascend Mul will case scale to interger first if matrix dtype is interger.
 * Result is not match, fixme after Ascend Op updated.
 */
float randomScale = randomInterger();
TEST(ELEMENTWISE_OP, MAT_MUL_MAT_WITH_SCALE)
{
    testMatOpMat(cv::multiply, cv::cann::multiply, randomScale, -1);
}

/*
TEST(ELEMENTWISE_OP, MAT_DIV_MAT_WITH_SCALE)
{
    testMatOpMat([](const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst, double scale, int dtype)
                 { cv::divide(src1, src2, dst, scale, dtype); },
                 cv::cann::divide, randomScale, -1);
}
*/

template <typename FCV, typename FCANN, typename... PARAMS>
void testMatOpScalar(FCV cvFunc, FCANN cannFunc, PARAMS... param)
{
    Scalar scalar = randomScalar();
    Mat mat(10, 10, CV_32SC3, randomScalar());
    Mat cpuDst1, cpuDst2, checker1, checker2;

    cvFunc(Mat(10, 10, CV_32SC3, scalar), mat, cpuDst1, param...);
    cvFunc(mat, Mat(10, 10, CV_32SC3, scalar), cpuDst2, param...);
    cv::cann::setDevice(DEVICE_ID);

    cannFunc(scalar, mat, checker1, param..., AscendStream::Null());
    cannFunc(mat, scalar, checker2, param..., AscendStream::Null());
    EXPECT_MAT_NEAR(cpuDst1, checker1, 0.0);
    EXPECT_MAT_NEAR(cpuDst2, checker2, 0.0);

    AscendStream stream;
    cannFunc(scalar, mat, checker1, param..., stream);
    cannFunc(mat, scalar, checker2, param..., stream);
    stream.waitForCompletion();
    EXPECT_MAT_NEAR(cpuDst1, checker1, 0.0);
    EXPECT_MAT_NEAR(cpuDst2, checker2, 0.0);

    cv::cann::resetDevice();
}

TEST(ELEMENTWISE_OP, MAT_ADD_SCALAR) { testMatOpScalar(cv::add, cv::cann::add, noArray(), -1); }

TEST(ELEMENTWISE_OP, MAT_SUB_SCALAR)
{
    testMatOpScalar(cv::subtract, cv::cann::subtract, noArray(), -1);
}

TEST(ELEMENTWISE_OP, MAT_MUL_SCALAR) { testMatOpScalar(cv::multiply, cv::cann::multiply, 1, -1); }

/*
TEST(ELEMENTWISE_OP, MAT_DIV_SCALAR)
{
    testMatOpScalar([](const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst, double scale,
                       int dtype) { cv::divide(src1, src2, dst, scale, dtype); },
                    cv::cann::divide, 1, -1);
}
*/

TEST(ELEMENTWISE_OP, MAT_BITWISE_AND_SCALAR)
{
    testMatOpScalar(cv::bitwise_and, cv::cann::bitwise_and, noArray());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_OR_SCALAR)
{
    testMatOpScalar(cv::bitwise_or, cv::cann::bitwise_or, noArray());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_XOR_SCALAR)
{
    testMatOpScalar(cv::bitwise_xor, cv::cann::bitwise_xor, noArray());
}

TEST(ELEMENTWISE_OP, MAT_ADD_SCALAR_WITH_MASK_AND_DETYPE)
{
    testMatOpScalar(cv::add, cv::cann::add, genMask(), CV_32SC3);
}

TEST(ELEMENTWISE_OP, MAT_SUB_SCALAR_WITH_MASK_AND_DETYPE)
{
    testMatOpScalar(cv::subtract, cv::cann::subtract, genMask(), CV_32SC3);
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_AND_SCALAR_WITH_MASK)
{
    testMatOpScalar(cv::bitwise_and, cv::cann::bitwise_and, genMask());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_OR_SCALAR_WITH_MASK)
{
    testMatOpScalar(cv::bitwise_or, cv::cann::bitwise_or, genMask());
}

TEST(ELEMENTWISE_OP, MAT_BITWISE_XOR_SCALAR_WITH_MASK)
{
    testMatOpScalar(cv::bitwise_xor, cv::cann::bitwise_xor, genMask());
}

TEST(ELEMENTWISE_OP, MAT_MUL_SCALAR_WITH_SCALE)
{
    testMatOpScalar(cv::multiply, cv::cann::multiply, randomScale, -1);
}

/*
TEST(ELEMENTWISE_OP, MAT_DIV_SCALAR_WITH_SCALE)
{
    testMatOpScalar([](const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dst, double scale,
                       int dtype) { cv::divide(src1, src2, dst, scale, dtype); },
                    cv::cann::divide, randomScale, -1);
}
*/

TEST(ELEMENTWISE_OP, MAT_BITWISE_NOT_1)
{
    Mat cpuOpRet, checker, cpuMat = randomMat(10, 10, CV_32SC3);

    cv::cann::setDevice(DEVICE_ID);

    cv::bitwise_not(cpuMat, cpuOpRet);
    cv::cann::bitwise_not(cpuMat, checker);
    EXPECT_MAT_NEAR(cpuOpRet, checker, 0.0);

    cv::cann::resetDevice();
}

// TODO random test matrix
TEST(ELEMENTWISE_OP, MAT_ADD_WEIGHTED_1)
{
    Mat cpuOpRet, checker, cpuMat1 = Mat::ones(5, 5, CV_32S), cpuMat2 = Mat::ones(5, 5, CV_32S);

    cv::cann::setDevice(DEVICE_ID);

    cv::addWeighted(cpuMat1, 2, cpuMat2, 3, 5, cpuOpRet);
    cv::cann::addWeighted(cpuMat1, 2, cpuMat2, 3, 5, checker);
    EXPECT_MAT_NEAR(cpuOpRet, checker, 0.0);

    cv::cann::resetDevice();
}

TEST(ELEMENTWISE_OP, MAT_THRESHOLD_1)
{
    Mat cpuOpRet, checker, cpuMat = randomMat(10, 10, CV_16SC3, 0.0, 255.0);

    NpuMat npuMat, npuMat16F, aclOpRet, aclOpRet16S;
    cv::cann::setDevice(DEVICE_ID);
    npuMat.upload(cpuMat);
    npuMat.convertTo(npuMat16F, CV_16F);

    for (int i = 0; i <= 4; i++)
    {
        cv::threshold(cpuMat, cpuOpRet, 128, 250, i);
        cv::cann::threshold(npuMat16F, aclOpRet, 128, 250, i);
        aclOpRet.convertTo(aclOpRet16S, CV_16S);
        aclOpRet16S.download(checker);

        EXPECT_MAT_NEAR(cpuOpRet, checker, 1e-10);
    }

    cv::cann::resetDevice();
}

} // namespace
} // namespace opencv_test
