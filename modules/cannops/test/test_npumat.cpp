// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test
{
namespace
{

class DummyAllocator : public NpuMat::Allocator
{
public:
    std::shared_ptr<uchar> allocate(size_t size) CV_OVERRIDE
    {
        CV_UNUSED(size);
        return std::shared_ptr<uchar>();
    }
    bool allocate(cv::cann::NpuMat* mat, int rows, int cols, size_t elemSize) CV_OVERRIDE
    {
        CV_UNUSED(rows);
        CV_UNUSED(cols);
        CV_UNUSED(elemSize);
        mat->data = std::shared_ptr<uchar>((uchar*)0x12345, [](void* ptr) { CV_UNUSED(ptr); });
        return true;
    }
};

TEST(NpuMat, Construct)
{
    cv::cann::setDevice(0);
    // 1 Default constructor.
    NpuMat defaultNpuMat;
    NpuMat::Allocator* defaultAllocator = NpuMat::defaultAllocator();
    ASSERT_EQ(defaultNpuMat.allocator, defaultAllocator);

    // 2 get & set allocator.
    DummyAllocator dummyAllocator;
    NpuMat::setDefaultAllocator(&dummyAllocator);
    ASSERT_EQ(defaultNpuMat.defaultAllocator(), &dummyAllocator);
    NpuMat::setDefaultAllocator(defaultAllocator);

    // 3 constructs NpuMat of the specified size and type
    NpuMat specifiedSizeNpuMat1(5, 6, CV_8UC3);
    NpuMat specifiedSizeNpuMat2(Size(300, 200), CV_64F);

    ASSERT_EQ(specifiedSizeNpuMat1.rows, 5);
    ASSERT_EQ(specifiedSizeNpuMat1.cols, 6);
    ASSERT_EQ(specifiedSizeNpuMat1.depth(), CV_8U);
    ASSERT_EQ(specifiedSizeNpuMat1.channels(), 3);

    ASSERT_EQ(specifiedSizeNpuMat2.cols, 300);
    ASSERT_EQ(specifiedSizeNpuMat2.rows, 200);
    ASSERT_EQ(specifiedSizeNpuMat2.depth(), CV_64F);
    ASSERT_EQ(specifiedSizeNpuMat2.channels(), 1);

    // 4 constructs NpuMat and fills it with the specified value s
    srand((unsigned int)(time(NULL)));
    Scalar sc(rand() % 256, rand() % 256, rand() % 256, rand() % 256);

    Mat scalarToMat(7, 8, CV_8UC3, sc);
    NpuMat scalarToNpuMat1(7, 8, CV_8UC3, sc);
    Mat scalarToMatChecker;
    scalarToNpuMat1.download(scalarToMatChecker);

    EXPECT_MAT_NEAR(scalarToMat, scalarToMatChecker, 0.0);

    NpuMat scalarToNpuMat2(Size(123, 345), CV_32S);

    ASSERT_EQ(scalarToNpuMat1.rows, 7);
    ASSERT_EQ(scalarToNpuMat1.cols, 8);
    ASSERT_EQ(scalarToNpuMat1.depth(), CV_8U);
    ASSERT_EQ(scalarToNpuMat1.channels(), 3);

    ASSERT_EQ(scalarToNpuMat2.cols, 123);
    ASSERT_EQ(scalarToNpuMat2.rows, 345);
    ASSERT_EQ(scalarToNpuMat2.depth(), CV_32S);
    ASSERT_EQ(scalarToNpuMat2.channels(), 1);

    // 6 builds NpuMat from host memory
    Scalar sc2(rand() % 256, rand() % 256, rand() % 256, rand() % 256);
    Mat randomMat(7, 8, CV_8UC3, sc2);
    InputArray arr = randomMat;

    NpuMat fromInputArray(arr, AscendStream::Null());
    Mat randomMatChecker;
    fromInputArray.download(randomMatChecker);
    EXPECT_MAT_NEAR(randomMat, randomMatChecker, 0.0);

    cv::cann::resetDevice();
}

TEST(NpuMat, Assignment)
{
    DummyAllocator dummyAllocator;
    NpuMat mat1;
    NpuMat mat2(3, 4, CV_8SC1, &dummyAllocator);
    mat1 = mat2;

    ASSERT_EQ(mat1.rows, 3);
    ASSERT_EQ(mat1.cols, 4);
    ASSERT_EQ(mat1.depth(), CV_8S);
    ASSERT_EQ(mat1.channels(), 1);
    ASSERT_EQ(mat1.data.get(), (uchar*)0x12345);
}

TEST(NpuMat, SetTo)
{
    cv::cann::setDevice(0);

    srand((unsigned int)(time(NULL)));
    Scalar sc(rand() % 256, rand() % 256, rand() % 256, rand() % 256);

    NpuMat npuMat(2, 2, CV_8UC4);
    npuMat.setTo(sc);
    Mat mat(2, 2, CV_8UC4, sc);
    Mat checker;
    npuMat.download(checker);

    EXPECT_MAT_NEAR(mat, checker, 0.0);

    cv::cann::resetDevice();
}

TEST(NpuMat, ConvertTo)
{
    cv::cann::setDevice(0);

    srand((unsigned int)(time(NULL)));
    Scalar sc(rand() % 256, rand() % 256, rand() % 256, rand() % 256);

    NpuMat npuMat(2, 2, CV_8UC4, sc);
    NpuMat convertedNpuMat;
    npuMat.convertTo(convertedNpuMat, CV_16S);
    Mat mat(2, 2, CV_16SC4, sc);
    Mat checker;
    convertedNpuMat.download(checker);

    EXPECT_MAT_NEAR(mat, checker, 0.0);

    cv::cann::resetDevice();
}

} // namespace
} // namespace opencv_test
