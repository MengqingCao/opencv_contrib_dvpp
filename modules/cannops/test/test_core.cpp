// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <vector>

namespace opencv_test
{
namespace
{
TEST(CORE, MERGE)
{
    Mat m1 = (Mat_<uchar>(2, 2) << 1, 4, 7, 10);
    Mat m2 = (Mat_<uchar>(2, 2) << 2, 5, 8, 11);
    Mat m3 = (Mat_<uchar>(2, 2) << 3, 6, 9, 12);
    Mat channels[3] = {m1, m2, m3};
    Mat m;
    cv::merge(channels, 3, m);

    cv::cann::setDevice(0);

    AscendMat a1, a2, a3;
    a1.upload(m1);
    a2.upload(m2);
    a3.upload(m3);
    AscendMat aclChannels[3] = {a1, a2, a3};
    std::vector<AscendMat> aclChannelsVector;
    aclChannelsVector.push_back(a1);
    aclChannelsVector.push_back(a2);
    aclChannelsVector.push_back(a3);

    Mat checker1, checker2;
    cv::cann::merge(aclChannels, 3, checker1);
    cv::cann::merge(aclChannelsVector, checker2);

    EXPECT_MAT_NEAR(m, checker1, 0.0);
    EXPECT_MAT_NEAR(m, checker2, 0.0);

    cv::cann::resetDevice();
}

TEST(CORE, SPLIT)
{
    char d[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Mat m(2, 2, CV_8UC3, d);
    Mat channels[3];
    cv::split(m, channels);

    cv::cann::setDevice(0);

    AscendMat aclChannels[3];
    std::vector<AscendMat> aclChannelsVector;

    cv::cann::split(m, aclChannels);
    cv::cann::split(m, aclChannelsVector);

    Mat checker1[3], checker2[3];
    aclChannels[0].download(checker1[0]);
    aclChannels[1].download(checker1[1]);
    aclChannels[2].download(checker1[2]);

    aclChannelsVector[0].download(checker2[0]);
    aclChannelsVector[1].download(checker2[1]);
    aclChannelsVector[2].download(checker2[2]);

    EXPECT_MAT_NEAR(channels[0], checker1[0], 0.0);
    EXPECT_MAT_NEAR(channels[1], checker1[1], 0.0);
    EXPECT_MAT_NEAR(channels[2], checker1[2], 0.0);

    EXPECT_MAT_NEAR(channels[0], checker2[0], 0.0);
    EXPECT_MAT_NEAR(channels[1], checker2[1], 0.0);
    EXPECT_MAT_NEAR(channels[2], checker2[2], 0.0);

    AscendMat npuM;
    npuM.upload(m);
    cv::cann::split(npuM, aclChannels);
    cv::cann::split(npuM, aclChannelsVector);

    aclChannels[0].download(checker1[0]);
    aclChannels[1].download(checker1[1]);
    aclChannels[2].download(checker1[2]);

    aclChannelsVector[0].download(checker2[0]);
    aclChannelsVector[1].download(checker2[1]);
    aclChannelsVector[2].download(checker2[2]);

    EXPECT_MAT_NEAR(channels[0], checker1[0], 0.0);
    EXPECT_MAT_NEAR(channels[1], checker1[1], 0.0);
    EXPECT_MAT_NEAR(channels[2], checker1[2], 0.0);

    EXPECT_MAT_NEAR(channels[0], checker2[0], 0.0);
    EXPECT_MAT_NEAR(channels[1], checker2[1], 0.0);
    EXPECT_MAT_NEAR(channels[2], checker2[2], 0.0);
    cv::cann::resetDevice();
}

TEST(CORE, TRANSPOSE)
{
    Mat cpuMat = randomMat(10, 10, CV_32SC3), cpuRetMat, checker;
    cv::transpose(cpuMat, cpuRetMat);
    cv::cann::transpose(cpuMat, checker);
    EXPECT_MAT_NEAR(cpuRetMat, checker, 0.0);

    AscendMat npuMat, npuChecker;
    npuMat.upload(cpuMat);
    cv::cann::transpose(npuMat, npuChecker);
    npuChecker.download(checker);
    EXPECT_MAT_NEAR(cpuRetMat, checker, 0.0);
}

TEST(CORE, FLIP)
{
    Mat cpuMat = randomMat(10, 10, CV_32SC3), cpuRetMat, checker;

    int flipMode;

    for (flipMode = -1; flipMode < 2; flipMode++)
    {
        cv::flip(cpuMat, cpuRetMat, flipMode);
        cv::cann::flip(cpuMat, checker, flipMode);
        EXPECT_MAT_NEAR(cpuRetMat, checker, 0.0);
    }

    AscendMat npuMat, npuChecker;
    npuMat.upload(cpuMat);
    for (flipMode = -1; flipMode < 2; flipMode++)
    {
        cv::flip(cpuMat, cpuRetMat, flipMode);
        cv::cann::flip(npuMat, npuChecker, flipMode);
        npuChecker.download(checker);
        EXPECT_MAT_NEAR(cpuRetMat, checker, 0.0);
    }
}

TEST(CORE, ROTATE)
{
    Mat cpuRetMat, checker, cpuMat = randomMat(3, 5, CV_16S, 0.0, 255.0);

    int rotateMode;
    for (rotateMode = 0; rotateMode < 3; rotateMode++)
    {
        cv::rotate(cpuMat, cpuRetMat, rotateMode);
        cv::cann::rotate(cpuMat, checker, rotateMode);
        EXPECT_MAT_NEAR(cpuRetMat, checker, 0.0);
    }

    AscendMat npuMat, npuChecker;
    npuMat.upload(cpuMat);
    for (rotateMode = 0; rotateMode < 3; rotateMode++)
    {
        cv::rotate(cpuMat, cpuRetMat, rotateMode);
        cv::cann::rotate(npuMat, npuChecker, rotateMode);
        npuChecker.download(checker);
        EXPECT_MAT_NEAR(cpuRetMat, checker, 0.0);
    }
}

TEST(CORE, CROP)
{
    Mat cpuOpRet, checker, cpuMat = randomMat(6, 6, CV_32SC3, 0.0, 255.0);
    Rect b(1, 2, 4, 4);
    Mat cropped_cv(cpuMat, b);
    AscendMat cropped_cann(cpuMat, b);
    cropped_cann.download(checker);
    EXPECT_MAT_NEAR(cropped_cv, checker, 1e-10);
}

TEST(CORE, CROP_OVERLOAD)
{
    Mat cpuOpRet, checker, cpuMat = randomMat(6, 6, CV_16SC3, 0.0, 255.0);
    const Rect b(1, 2, 4, 4);
    Mat cropped_cv = cpuMat(b);
    AscendMat cropped_cann = cv::cann::crop(cpuMat, b);
    cropped_cann.download(checker);
    EXPECT_MAT_NEAR(cropped_cv, checker, 1e-10);

    AscendMat npuMat;
    npuMat.upload(cpuMat);
    cropped_cann = cv::cann::crop(npuMat, b);
    cropped_cann.download(checker);
    EXPECT_MAT_NEAR(cropped_cv, checker, 1e-10);
}

TEST(CORE, RESIZE)
{
    Mat resized_cv, checker, cpuMat = randomMat(10, 10, CV_32F, 100.0, 255.0);
    Size dsize = Size(6, 6);
    // only support {2 INTER_CUBIC} and {3 INTER_AREA}
    // only the resize result of INTER_AREA is close to CV's.
    int flags = 3;
    cv::cann::setDevice(0);
    cv::resize(cpuMat, resized_cv, dsize, 0, 0, flags);
    cv::cann::resize(cpuMat, checker, dsize, 0, 0, flags);
    EXPECT_MAT_NEAR(resized_cv, checker, 1e-4);

    cv::resize(cpuMat, resized_cv, Size(), 0.5, 0.5, flags);
    cv::cann::resize(cpuMat, checker, Size(), 0.5, 0.5, flags);
    EXPECT_MAT_NEAR(resized_cv, checker, 1e-4);

    AscendMat npuMat, npuChecker;
    npuMat.upload(cpuMat);
    cv::resize(cpuMat, resized_cv, dsize, 0, 0, flags);
    cv::cann::resize(npuMat, npuChecker, dsize, 0, 0, flags);
    npuChecker.download(checker);
    EXPECT_MAT_NEAR(resized_cv, checker, 1e-4);

    cv::resize(cpuMat, resized_cv, Size(), 0.5, 0.5, flags);
    cv::cann::resize(npuMat, npuChecker, Size(), 0.5, 0.5, flags);
    npuChecker.download(checker);
    EXPECT_MAT_NEAR(resized_cv, checker, 1e-4);
    cv::cann::resetDevice();
}

TEST(CORE, RESIZE_DVPP)
{
    Mat resized_cv, checker, cpuMat = randomMat(256, 256, CV_8UC3, 100.0, 255.0);
    Size dsize = Size(64, 64);
    cv::resize(cpuMat, resized_cv, dsize, 0, 0, 1);
    cv::cann::resizedvpp(cpuMat, checker, dsize, 0, 0, 0);
    EXPECT_MAT_NEAR(resized_cv, checker, 1e-10);

    cv::resize(cpuMat, resized_cv, Size(), 0.5, 0.5, 1);
    cv::cann::resizedvpp(cpuMat, checker, Size(), 0.5, 0.5, 0);
    EXPECT_MAT_NEAR(resized_cv, checker, 1e-10);
}

TEST(CORE, CROP_OVERLOAD_DVPP)
{
    Mat cpuOpRet, checker, cpuMat = randomMat(256, 256, CV_8UC3, 0.0, 255.0);
    const Rect b(10, 20, 64, 64);
    cv::cann::setDevice(0);
    Mat cropped_cv = cpuMat(b);
    Mat cropped_cann = cv::cann::cropdvpp(cpuMat, b);
    EXPECT_MAT_NEAR(cropped_cv, cropped_cann, 1e-10);
    cv::cann::resetDevice();
}

TEST(CORE, INVERT)
{
    Mat a = (cv::Mat_<float>(3, 3) << 2.42104644730331, 1.81444796521479, -3.98072565304758, 0,
             7.08389214348967e-3, 5.55326770986007e-3, 0, 0, 7.44556154284261e-3);
    Mat b = a.t() * a;
    Mat cpuOpRet, checker, checkerInv, i = Mat_<float>::eye(3, 3);
    cv::cann::setDevice(DEVICE_ID);
    cv::invert(b, cpuOpRet);
    cv::cann::invert(b, checker);

    std::cout << checker << '\n' << '\n' << cpuOpRet << std::endl;
    ASSERT_LT(cvtest::norm(b * checker, i, 1), 0.1);

    // cv::cann::invert(checker, checkerInv);
    // std::cout  << '\n' << checkerInv << '\n' << '\n' << b << std::endl;
    // EXPECT_MAT_NEAR(checkerInv, b, 1e-5);

    // EXPECT_MAT_NEAR(checker, cpuOpRet, 1e-10);
    cv::cann::resetDevice();
}

TEST(CORE, CROP_RESIZE)
{
    cv::cann::setDevice(DEVICE_ID);

    Mat resized_cv, checker, cpuOpRet, cpuMat = randomMat(256, 256, CV_8UC3, 100.0, 255.0);
    Size dsize = Size(64, 64);
    const Rect b(1, 2, 128, 128);
    RNG rng(12345);
    double scalarV[3] = {1, 1, 1};
    int top, bottom, left, right;
    top = (int)(0);
    bottom = 0;
    left = (int)(0);
    right = 0;
    int borderType = 0;
    // HI_BORDER_CONSTANT = 0 BORDER_CONSTANT = 0

    cv::cann::CropResizeMakeBorder(cpuMat, checker, b, dsize, 0, 0, 0, borderType, scalarV,
                                        top, left);

    Mat cropped_cv(cpuMat, b);
    cv::resize(cropped_cv, resized_cv, dsize, 0, 0, 1);
    Scalar value = {scalarV[0], scalarV[1], scalarV[2]};

    cv::copyMakeBorder(resized_cv, cpuOpRet, top, bottom, left, right, borderType, value);
    EXPECT_MAT_NEAR(checker, cpuOpRet, 1e-10);
    cv::cann::resetDevice();
}

TEST(CORE, BATCH_CROP_RESIZE)
{
    cv::cann::setDevice(DEVICE_ID);

    Mat resized_cv, cpuOpRet, cpuMat = randomMat(256, 256, CV_8UC3, 100.0, 255.0);
    Size dsize = Size(64, 64);
    const Rect b(1, 2, 128, 128);
    RNG rng(12345);
    double scalarV[3] = {1, 1, 1};
    int top, bottom, left, right;
    top = (int)(0);
    bottom = 0;
    left = (int)(0);
    right = 0;
    int batchNum = 128;
    std::vector<cv::Mat> batchInput(batchNum, Mat()), checker(batchNum, Mat());
    for (int i = 0; i < batchNum; i++)
    {
        batchInput[i] = Mat(cpuMat);
        checker[i].create(dsize.width + left, dsize.height + top, cpuMat.type());
    }
    int borderType = 0;
    // HI_BORDER_CONSTANT = 0 BORDER_CONSTANT = 0

    cv::cann::batchCropResizeMakeBorder(batchInput, checker, b, dsize, 0, 0, 0, borderType, scalarV,
                                        top, left, batchNum);

    Mat cropped_cv(cpuMat, b);
    cv::resize(cropped_cv, resized_cv, dsize, 0, 0, 1);
    Scalar value = {scalarV[0], scalarV[1], scalarV[2]};

    cv::copyMakeBorder(resized_cv, cpuOpRet, top, bottom, left, right, borderType, value);
    for (int i = 0; i < batchNum; i++)
    {
        EXPECT_MAT_NEAR(checker[i], cpuOpRet, 1e-10);
    }
    cv::cann::resetDevice();
}

} // namespace
} // namespace opencv_test
