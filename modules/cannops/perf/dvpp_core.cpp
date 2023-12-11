// // This file is part of OpenCV project.
// // It is subject to the license terms in the LICENSE file found in the top-level directory
// // of this distribution and at http://opencv.org/license.html.

// #include "perf_precomp.hpp"
// #include "opencv2/ts/cuda_test.hpp"
// #include "opencv2/cann_interface.hpp"
// #undef EXPECT_MAT_NEAR
// #define EXPECT_MAT_NEAR(m1, m2, eps) EXPECT_PRED_FORMAT3(cvtest::assertMatNear, m1, m2, eps)
// Mat randomMat(int w, int h, int dtype, float min, float max)
// {
//     Mat rnMat(w, h, dtype);
//     RNG rng(getTickCount());
//     rng.fill(rnMat, RNG::UNIFORM, min, max);
//     return rnMat;
// }
// #include <vector>
// namespace opencv_test
// {
// namespace
// {
// #define DEF_PARAM_TEST(name, ...) \
//     typedef ::perf::TestBaseWithParam<testing::tuple<__VA_ARGS__>> name
// #define DVPP_ASCEND_MAT_SIZES                                                                    \
//     Values(::perf::szSmall128, ::perf::sz720p, ::perf::szQVGA, ::perf::szVGA, \
//            ::perf::szSVGA, ::perf::szXGA, ::perf::szSXGA, ::perf::szWQHD, ::perf::sznHD,         \
//            ::perf::szqHD, ::perf::sz1080p, ::perf::sz2160p, ::perf::sz1440p, ::perf::sz3MP,      \
//            ::perf::sz5MP, ::perf::sz2K)
// #define DVPP_DTYPE testing::Values(MatType(CV_8UC3), MatType(CV_8UC1))
// DEF_PARAM_TEST(CORE, Size, MatType);

// PERF_TEST_P(CORE, RESIZE_NEW, testing::Combine(DVPP_ASCEND_MAT_SIZES, DVPP_DTYPE))
// {
//     Size size = GET_PARAM(0);
//     int type = GET_PARAM(1);
//     Mat resized_cv, checker;

//     Mat cpuMat = randomMat(size.height, size.width, type, 100.0, 255.0);
//     Size dsize = Size(cpuMat.rows / 2, cpuMat.cols / 2);
//     int interpolation = 1;

//     TEST_CYCLE()
//     {
//         cv::resize(cpuMat, resized_cv, dsize, 0, 0, interpolation);
//         cv::cann::resize(cpuMat, checker, dsize, 0, 0, interpolation);
//         EXPECT_MAT_NEAR(resized_cv, checker, 1);

//         cv::resize(cpuMat, resized_cv, Size(), 0.5, 0.5, interpolation);
//         cv::cann::resize(cpuMat, checker, Size(), 0.5, 0.5, interpolation);
//         EXPECT_MAT_NEAR(resized_cv, checker, 1);

//         AscendMat npuMat, npuChecker;
//         npuMat.upload(cpuMat);
//         cv::resize(cpuMat, resized_cv, dsize, 0, 0, interpolation);
//         cv::cann::resize(npuMat, npuChecker, dsize, 0, 0, interpolation);
//         npuChecker.download(checker);
//         EXPECT_MAT_NEAR(resized_cv, checker, 1);

//         cv::resize(cpuMat, resized_cv, Size(), 0.5, 0.5, interpolation);
//         cv::cann::resize(npuMat, npuChecker, Size(), 0.5, 0.5, interpolation);
//         npuChecker.download(checker);
//         EXPECT_MAT_NEAR(resized_cv, checker, 1);
//     }
//     SANITY_CHECK_NOTHING();
// }
// PERF_TEST_P(CORE, CROP_RESIZE, testing::Combine(DVPP_ASCEND_MAT_SIZES, DVPP_DTYPE))
// {
//     Size size = GET_PARAM(0);
//     int type = GET_PARAM(1);
//     Mat cpuMat = randomMat(size.height, size.width, type, 100.0, 255.0);
//     Mat resized_cv, checker, cpuOpRet;
//     Size dsize = Size(size.width / 4, size.height / 2);
//     const Rect b(0, 0, size.width / 2, size.height);

//     TEST_CYCLE()
//     {
//         cv::cann::cropResize(cpuMat, checker, b, dsize, 0, 0, 1);
//         Mat cropped_cv(cpuMat, b);
//         cv::resize(cropped_cv, cpuOpRet, dsize, 0, 0, 1);
//         EXPECT_MAT_NEAR(checker, cpuOpRet, 1);

//         AscendMat npuMat, npuChecker;
//         npuMat.upload(cpuMat);
//         cv::cann::cropResize(npuMat, npuChecker, b, dsize, 0, 0, 1);
//         npuChecker.download(checker);
//         EXPECT_MAT_NEAR(cpuOpRet, checker, 1);
//     }
//     SANITY_CHECK_NOTHING();
// }
// PERF_TEST_P(CORE, CROP_RESIZE_MAKE_BORDER, testing::Combine(DVPP_ASCEND_MAT_SIZES, DVPP_DTYPE))
// {
//     Size size = GET_PARAM(0);
//     int type = GET_PARAM(1);
//     Mat cpuMat = randomMat(size.height, size.width, type, 100.0, 255.0);

//     Mat resized_cv, checker, cpuOpRet;
//     Size dsize = Size(size.width / 4, size.height / 2);
//     const Rect b(0, 0, size.width / 2, size.height);
//     RNG rng(12345);
//     float scalarV[3] = {0, 0, 255};
//     int top, bottom, left, right;
//     top = 54;
//     bottom = 0;
//     left = 32;
//     right = 0;
//     int interpolation = 1;

//     Scalar value = {scalarV[0], scalarV[1], scalarV[2], 0};
//     TEST_CYCLE()
//     {
//         for (int borderType = 0; borderType < 2; borderType++)
//         {
//             cv::cann::cropResizeMakeBorder(cpuMat, checker, b, dsize, 0, 0, interpolation,
//                                            borderType, value, top, left);

//             Mat cropped_cv(cpuMat, b);
//             cv::resize(cropped_cv, resized_cv, dsize, 0, 0, interpolation);
//             cv::copyMakeBorder(resized_cv, cpuOpRet, top, bottom, left, right, borderType, value);
//             EXPECT_MAT_NEAR(checker, cpuOpRet, 1e-10);
//         }
//     }
//     SANITY_CHECK_NOTHING();
// }
// PERF_TEST_P(CORE, BATCH_CROP_RESIZE, testing::Combine(DVPP_ASCEND_MAT_SIZES, DVPP_DTYPE))
// {
//     Size size = GET_PARAM(0);
//     int type = GET_PARAM(1);
//     Mat cpuMat = randomMat(size.height, size.width, type, 100.0, 255.0);

//     Mat resized_cv, cpuOpRet;
//     Size dsize = Size(size.width / 4, size.height / 2);
//     const Rect b(0, 0, size.width / 2, size.height);
//     RNG rng(12345);
//     int scalarV[3] = {rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)};
//     int top, bottom, left, right;
//     top = (int)(0);
//     bottom = 0;
//     left = (int)(0);
//     right = 0;
//     int batchNum = 128;
//     std::vector<cv::Mat> batchInput(batchNum, Mat()), checker(batchNum, Mat());
//     for (int i = 0; i < batchNum; i++)
//     {
//         batchInput[i] = Mat(cpuMat);
//         checker[i].create(dsize.height + top, dsize.width + left, cpuMat.type());
//     }
//     int borderType = 0;
//     Scalar value = {scalarV[0], scalarV[1], scalarV[2]};

//     TEST_CYCLE()
//     {
//         cv::cann::batchCropResizeMakeBorder(batchInput, checker, b, dsize, 0, 0, 1, borderType,
//                                             value, top, left, batchNum);
//         Mat cropped_cv(cpuMat, b);
//         cv::resize(cropped_cv, resized_cv, dsize, 0, 0, 1);
//         cv::copyMakeBorder(resized_cv, cpuOpRet, top, bottom, left, right, borderType, value);
//         for (int i = 0; i < batchNum; i++)
//         {
//             EXPECT_MAT_NEAR(checker[i], cpuOpRet, 1e-10);
//         }
//     }

//     SANITY_CHECK_NOTHING();
// }
// PERF_TEST_P(CORE, COPY_MAKE_BORDER, testing::Combine(DVPP_ASCEND_MAT_SIZES, DVPP_DTYPE))
// {
//     Size size = GET_PARAM(0);
//     int type = GET_PARAM(1);
//     Mat cpuMat = randomMat(size.height, size.width, type, 100.0, 255.0);

//     Mat resized_cv, cpuOpRet, checker;
//     const Rect b(0, 0, size.width / 2, size.height);
//     RNG rng(12345);
//     int scalarV[3] = {rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)};
//     int top, bottom, left, right;
//     top = 50;
//     bottom = 60;
//     left = 32;
//     right = 32;

//     int borderType = 0;
//     Scalar value = {scalarV[0], scalarV[1], scalarV[2]};

//     TEST_CYCLE()
//     {
//         cv::cann::copyMakeBorder(cpuMat, checker, top, bottom, left, right, borderType, value);

//         cv::copyMakeBorder(cpuMat, cpuOpRet, top, bottom, left, right, borderType, value);
//         EXPECT_MAT_NEAR(checker, cpuOpRet, 1e-10);
//     }
//     SANITY_CHECK_NOTHING();
// }

// } // namespace
// } // namespace opencv_test
