// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/cann_interface.hpp"

namespace opencv_test
{
namespace
{
#define TYPICAL_ASCEND_MAT_SIZES \
    Values(::perf::sz1080p, ::perf::sz2K, ::perf::sz2160p, ::perf::sz4320p)
// #define DVPP_ASCEND_MAT_SIZES                                                               \
//     Values(::perf::sz1080p, ::perf::sz2160p, ::perf::sz1440p, ::perf::sz3MP, ::perf::sz5MP, \
//            ::perf::sz2K)
#define DVPP_ASCEND_MAT_SIZES \
    Values(::perf::sz1080p, ::perf::sz2K, ::perf::sz2160p, ::perf::sz4320p)
#define DEF_PARAM_TEST(name, ...) \
    typedef ::perf::TestBaseWithParam<testing::tuple<__VA_ARGS__>> name

DEF_PARAM_TEST(NPU, Size);
DEF_PARAM_TEST(CPU, Size);

PERF_TEST_P(NPU, MERGE_WARMUP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC1);
    AscendMat dst;
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    AscendMat ascendMat[3];
    ascendMat[0].upload(mat);
    ascendMat[1].upload(mat);
    ascendMat[2].upload(mat);

    TEST_CYCLE_N(10) { cv::cann::merge(&ascendMat[0], 3, dst); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, MERGE, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC1);
    AscendMat dst;
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    AscendMat ascendMat[3];
    ascendMat[0].upload(mat);
    ascendMat[1].upload(mat);
    ascendMat[2].upload(mat);

    TEST_CYCLE_N(10) { cv::cann::merge(&ascendMat[0], 3, dst); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, MERGE, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC1);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    Mat mats[3] = {mat, mat, mat};
    TEST_CYCLE_N(10) { cv::merge(&mats[0], 3, dst); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, SPLIT_WARMUP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    AscendMat ascendMat[3];
    AscendMat src;
    src.upload(mat);

    TEST_CYCLE_N(10) { cv::cann::split(src, &ascendMat[0]); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, SPLIT, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    AscendMat ascendMat[3];
    AscendMat src;
    src.upload(mat);

    TEST_CYCLE_N(10) { cv::cann::split(src, &ascendMat[0]); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, SPLIT, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    declare.in(mat, WARMUP_RNG);
    Mat mats[3] = {mat, mat, mat};
    TEST_CYCLE_N(10) { cv::split(mat, &mats[0]); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, TRANSPOSE_WARMUP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC4);
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::transpose(src, dst); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, TRANSPOSE, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC4);
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::transpose(src, dst); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, TRANSPOSE, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC4);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::transpose(mat, dst); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, FLIP_WARMUP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::flip(src, dst, -1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, FLIP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::flip(src, dst, -1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, FLIP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::flip(mat, dst, -1); }
    SANITY_CHECK_NOTHING();
}
PERF_TEST_P(NPU, ROTATE_WARMUP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::rotate(src, dst, 1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, ROTATE, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::rotate(src, dst, 1); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, ROTATE, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::rotate(mat, dst, 1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, CROP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    Rect b(1, 2, 64, 64);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { AscendMat cropped_cann(mat, b); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, CROP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    Rect b(1, 2, 64, 64);
    TEST_CYCLE_N(10) { Mat cropped_cv(mat, b); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, CROP_OVERLOAD_WARMUP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    Rect b(1, 2, 64, 64);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::crop(src, b); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, CROP_OVERLOAD, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    Rect b(1, 2, 64, 64);
    cv::cann::setDevice(DEVICE_ID);
    TEST_CYCLE_N(10) { cv::cann::crop(src, b); }
    cv::cann::resetDevice();
    SANITY_CHECK_NOTHING();
}
PERF_TEST_P(CPU, RESIZE, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    Size dsize = Size(256, 256);
    TEST_CYCLE_N(10) { cv::resize(mat, dst, dsize, 0, 0, 1); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, RESIZE_WARMUP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_32FC3);
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    Size dsize = Size(256, 256);
    TEST_CYCLE_N(10) { cv::cann::resize(src, dst, dsize, 0, 0, 2); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, RESIZE, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_32FC3);
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    Size dsize = Size(256, 256);
    TEST_CYCLE_N(10) { cv::cann::resize(src, dst, dsize, 0, 0, 2); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, THRESHOLD_WARMUP, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_32FC3);
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::cann::threshold(src, dst, 100.0, 255.0, cv::THRESH_BINARY); }
    SANITY_CHECK_NOTHING();
}
PERF_TEST_P(NPU, THRESHOLD, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_32FC3);
    AscendMat dst;
    AscendMat src;
    src.upload(mat);
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::cann::threshold(src, dst, 100.0, 255.0, cv::THRESH_BINARY); }
    SANITY_CHECK_NOTHING();
}
PERF_TEST_P(CPU, THRESHOLD, TYPICAL_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_32FC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    TEST_CYCLE_N(10) { cv::threshold(mat, dst, 100.0, 255.0, cv::THRESH_BINARY); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, RESIZE_DVPP_MAT, DVPP_ASCEND_MAT_SIZES)
{
    Mat mat(GET_PARAM(0), CV_8UC3);
    Mat dst;
    declare.in(mat, WARMUP_RNG);
    Size dsize = Size(256, 256);
    TEST_CYCLE_N(10) { cv::cann::resize(mat, dst, dsize, 0, 0, 0); }
    SANITY_CHECK_NOTHING();
}
// PERF_TEST_P(NPU, RESIZE_DVPP_ASCENDMAT, DVPP_ASCEND_MAT_SIZES)
// {
//     Mat mat(GET_PARAM(0), CV_8UC3);
//     // Mat dst;
//     AscendMat dst;
//     AscendMat src;
//     src.upload(mat);
//     declare.in(mat, WARMUP_RNG);
//     Size dsize = Size(256, 256);
//     TEST_CYCLE_N(10) { cv::cann::resize(src, dst, dsize, 0, 0, 0); }
//     SANITY_CHECK_NOTHING();
// }

PERF_TEST_P(NPU, COPY_MAKE_BORDER_DVPP, DVPP_ASCEND_MAT_SIZES)
{
    Mat resized_cv, checker, cpuOpRet, cpuMat(GET_PARAM(0), CV_8UC3);
    declare.in(cpuMat, WARMUP_RNG);
    int top, bottom, left, right;
    top = (int)(20);
    bottom = top;
    left = (int)(20);
    right = left;
    int borderType = 1;
    float scalarV[3] = {0, 0, 255};
    Scalar value = {scalarV[0], scalarV[1], scalarV[2]};

    TEST_CYCLE_N(10)
    {
        cv::cann::copyMakeBorder(cpuMat, checker, top, bottom, left, right, borderType, value);
    }

    SANITY_CHECK_NOTHING();
}
PERF_TEST_P(CPU, COPY_MAKE_BORDER_DVPP, DVPP_ASCEND_MAT_SIZES)
{
    Mat resized_cv, checker, cpuOpRet, cpuMat(GET_PARAM(0), CV_8UC3);
    declare.in(cpuMat, WARMUP_RNG);
    int top, bottom, left, right;
    top = (int)(20);
    bottom = top;
    left = (int)(20);
    right = left;
    int borderType = 1;
    float scalarV[3] = {0, 0, 255};
    Scalar value = {scalarV[0], scalarV[1], scalarV[2]};

    TEST_CYCLE_N(10)
    {
        cv::copyMakeBorder(cpuMat, checker, top, bottom, left, right, borderType, value);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, CROP_RESIZE_BORDER_DVPP, DVPP_ASCEND_MAT_SIZES)
{
    Size size = GET_PARAM(0);
    Mat resized_cv, checker, cpuOpRet, cpuMat(size, CV_8UC3);
    declare.in(cpuMat, WARMUP_RNG);

    const Rect b(1, 0, size.width / 2, size.height);
    Size dsize = Size(size.width / 4, size.height / 2);
    int top, bottom, left, right;
    top = (int)(20);
    bottom = 0;
    left = (int)(20);
    right = 0;
    int borderType = 0;
    float scalarV[3] = {1, 1, 1};
    Scalar value = {scalarV[0], scalarV[1], scalarV[2]};

    TEST_CYCLE_N(10)
    {
        cv::cann::cropResizeMakeBorder(cpuMat, checker, b, dsize, 0, 0, 1, borderType, value, top,
                                       left);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, CROP_RESIZE_BORDER_DVPP, DVPP_ASCEND_MAT_SIZES)
{
    Size size = GET_PARAM(0);
    Mat resized_cv, checker, cpuOpRet, cpuMat(size, CV_8UC3);
    declare.in(cpuMat, WARMUP_RNG);
    const Rect b(1, 0, size.width / 2, size.height);
    Size dsize = Size(size.width / 4, size.height / 2);
    int top, bottom, left, right;
    top = (int)(20);
    bottom = 0;
    left = (int)(20);
    right = 0;
    int borderType = 0;
    float scalarV[3] = {1, 1, 1};
    Scalar value = {scalarV[0], scalarV[1], scalarV[2]};

    TEST_CYCLE_N(10)
    {
        Mat cropped_cv(cpuMat, b);
        cv::resize(cropped_cv, resized_cv, dsize, 0, 0, 1);
        cv::copyMakeBorder(resized_cv, cpuOpRet, top, bottom, left, right, borderType, value);
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(NPU, CROP_RESIZE_DVPP, DVPP_ASCEND_MAT_SIZES)
{
    Size size = GET_PARAM(0);
    Mat resized_cv, checker, cpuOpRet, cpuMat(size, CV_8UC3);
    declare.in(cpuMat, WARMUP_RNG);
    const Rect b(1, 0, size.width / 2, size.height);
    Size dsize = Size(size.width / 4, size.height / 2);

    TEST_CYCLE_N(10) { cv::cann::cropResize(cpuMat, checker, b, dsize, 0, 0, 1); }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(CPU, CROP_RESIZE_DVPP, DVPP_ASCEND_MAT_SIZES)
{
    Size size = GET_PARAM(0);
    Mat resized_cv, checker, cpuOpRet, cpuMat(size, CV_8UC3);
    declare.in(cpuMat, WARMUP_RNG);
    const Rect b(1, 0, size.width / 2, size.height);
    Size dsize = Size(size.width / 4, size.height / 2);

    TEST_CYCLE_N(10)
    {
        Mat cropped_cv(cpuMat, b);
        cv::resize(cropped_cv, resized_cv, dsize, 0, 0, 1);
    }
    SANITY_CHECK_NOTHING();
}

// PERF_TEST_P(NPU, BTACH_CROP_RESIZE_DVPP_WARMUP, DVPP_ASCEND_MAT_SIZES)
// {
//     Mat resized_cv, cpuOpRet, cpuMat(GET_PARAM(0), CV_8UC3);
//     Size dsize = Size(64, 64);
//     const Rect b(1, 2, 128, 128);
//     RNG rng(12345);
//     double scalarV[3] = {1, 1, 1};
//     Scalar value = {scalarV[0], scalarV[1], scalarV[2], 0};
//     int top, bottom, left, right;
//     top = (int)(20);
//     bottom = 0;
//     left = (int)(20);
//     right = 0;
//     int batchNum = 128;
//     std::vector<cv::Mat> batchInput(batchNum, Mat()), checker(batchNum, Mat());
//     for (int i = 0; i < batchNum; i++)
//     {
//         batchInput[i] = Mat(cpuMat);
//         checker[i].create(dsize.width + left, dsize.height + top, cpuMat.type());
//     }
//     int borderType = 0;

//     TEST_CYCLE_N(10)
//     {
//         cv::cann::batchCropResizeMakeBorder(batchInput, checker, b, dsize, 0, 0, 0, borderType,
//                                             value, top, left, batchNum);
//     }

//     SANITY_CHECK_NOTHING();
// }

// PERF_TEST_P(NPU, BTACH_CROP_RESIZE_DVPP, DVPP_ASCEND_MAT_SIZES)
// {
//     Mat resized_cv, cpuOpRet, cpuMat(GET_PARAM(0), CV_8UC3);
//     Size dsize = Size(64, 64);
//     const Rect b(1, 2, 128, 128);
//     RNG rng(12345);
//     double scalarV[3] = {1, 1, 1};
//     Scalar value = {scalarV[0], scalarV[1], scalarV[2], 0};
//     int top, bottom, left, right;
//     top = (int)(20);
//     bottom = 0;
//     left = (int)(20);
//     right = 0;
//     int batchNum = 128;
//     std::vector<cv::Mat> batchInput(batchNum, Mat()), checker(batchNum, Mat());
//     for (int i = 0; i < batchNum; i++)
//     {
//         batchInput[i] = Mat(cpuMat);
//         checker[i].create(dsize.width + left, dsize.height + top, cpuMat.type());
//     }
//     int borderType = 0;

//     TEST_CYCLE_N(10)
//     {
//         cv::cann::batchCropResizeMakeBorder(batchInput, checker, b, dsize, 0, 0, 0, borderType,
//                                             value, top, left, batchNum);
//     }

//     SANITY_CHECK_NOTHING();
// }

// PERF_TEST_P(CPU, BTACH_CROP_RESIZE_DVPP, DVPP_ASCEND_MAT_SIZES)
// {
//     Mat resized_cv, checker, cpuOpRet, cpuMat(GET_PARAM(0), CV_8UC3);
//     declare.in(cpuMat, WARMUP_RNG);
//     const Rect b(1, 2, 128, 128);
//     Size dsize = Size(256, 256);
//     int top, bottom, left, right;
//     top = (int)(20);
//     bottom = 0;
//     left = (int)(20);
//     right = 0;
//     int borderType = 0;
//     double scalarV[3] = {1, 1, 1};
//     Scalar value = {scalarV[0], scalarV[1], scalarV[2]};
//     int batchNum = 128;

//     TEST_CYCLE_N(10)
//     {
//         for (int i = 0; i < batchNum; i++)
//         {
//             Mat cropped_cv(cpuMat, b);
//             cv::resize(cropped_cv, resized_cv, dsize, 0, 0, 1);
//             cv::copyMakeBorder(resized_cv, cpuOpRet, top, bottom, left, right, borderType,
//             value);
//         }
//     }
//     SANITY_CHECK_NOTHING();
// }

} // namespace
} // namespace opencv_test