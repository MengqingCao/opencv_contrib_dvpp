// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/ts/cuda_test.hpp>

namespace opencv_test
{
namespace
{

class DummyAllocator : public AclMat::Allocator
{
public:
    bool allocate(cv::cann::AclMat* mat, int rows, int cols, size_t elemSize) CV_OVERRIDE
    {
        CV_UNUSED(rows);
        CV_UNUSED(cols);
        CV_UNUSED(elemSize);
        mat->data = (uchar*)0x12345;
        mat->refcount = (int*)cv::fastMalloc(sizeof(int));
        return true;
    }
    void free(cv::cann::AclMat* mat) CV_OVERRIDE
    {
        mat->data = (uchar*)0x54321;
        cv::fastFree(mat->refcount);
    }
};

TEST(AclMat, Construct)
{
    cv::cann::setDevice(0);
    // 1 Default constructor.
    AclMat defaultAclMat;
    AclMat::Allocator* defaultAllocator = AclMat::defaultAllocator();
    ASSERT_EQ(defaultAclMat.allocator, defaultAllocator);

    // 2 get & set allocator.
    DummyAllocator dummyAllocator;
    AclMat::setDefaultAllocator(&dummyAllocator);
    ASSERT_EQ(defaultAclMat.defaultAllocator(), &dummyAllocator);
    AclMat::setDefaultAllocator(defaultAllocator);

    // 3 constructs AclMat of the specified size and type
    AclMat specifiedSizeAclMat1(5, 6, CV_8UC3);
    AclMat specifiedSizeAclMat2(Size(300, 200), CV_64F);

    ASSERT_EQ(specifiedSizeAclMat1.rows, 5);
    ASSERT_EQ(specifiedSizeAclMat1.cols, 6);
    ASSERT_EQ(specifiedSizeAclMat1.depth(), CV_8U);
    ASSERT_EQ(specifiedSizeAclMat1.channels(), 3);

    ASSERT_EQ(specifiedSizeAclMat2.cols, 300);
    ASSERT_EQ(specifiedSizeAclMat2.rows, 200);
    ASSERT_EQ(specifiedSizeAclMat2.depth(), CV_64F);
    ASSERT_EQ(specifiedSizeAclMat2.channels(), 1);

    // 4 constructs AclMat and fills it with the specified value s
    srand((unsigned int)(time(NULL)));
    Scalar sc(rand() % 256, rand() % 256, rand() % 256, rand() % 256);

    Mat scalarToMat(7, 8, CV_8UC3, sc);
    AclMat scalarToAclMat1(7, 8, CV_8UC3, sc);
    Mat scalarToMatChecker;
    scalarToAclMat1.download(scalarToMatChecker);

    EXPECT_MAT_NEAR(scalarToMat, scalarToMatChecker, 0.0);

    AclMat scalarToAclMat2(Size(123, 345), CV_32S);

    ASSERT_EQ(scalarToAclMat1.rows, 7);
    ASSERT_EQ(scalarToAclMat1.cols, 8);
    ASSERT_EQ(scalarToAclMat1.depth(), CV_8U);
    ASSERT_EQ(scalarToAclMat1.channels(), 3);

    ASSERT_EQ(scalarToAclMat2.cols, 123);
    ASSERT_EQ(scalarToAclMat2.rows, 345);
    ASSERT_EQ(scalarToAclMat2.depth(), CV_32S);
    ASSERT_EQ(scalarToAclMat2.channels(), 1);

    // 5 constructor for AclMat headers pointing to user-allocated data
    void* userAllocatedData = malloc(1);
    AclMat userAllocatedAclMat1(9, 10, CV_16SC2, userAllocatedData);
    AclMat userAllocatedAclMat2(Size(1024, 2048), CV_16F, userAllocatedData);

    ASSERT_EQ(userAllocatedAclMat1.rows, 9);
    ASSERT_EQ(userAllocatedAclMat1.cols, 10);
    ASSERT_EQ(userAllocatedAclMat1.depth(), CV_16S);
    ASSERT_EQ(userAllocatedAclMat1.channels(), 2);
    ASSERT_EQ(userAllocatedAclMat1.data, userAllocatedData);

    ASSERT_EQ(userAllocatedAclMat2.cols, 1024);
    ASSERT_EQ(userAllocatedAclMat2.rows, 2048);
    ASSERT_EQ(userAllocatedAclMat2.depth(), CV_16F);
    ASSERT_EQ(userAllocatedAclMat2.channels(), 1);
    ASSERT_EQ(userAllocatedAclMat1.data, userAllocatedData);

    // 6 builds AclMat from host memory
    Scalar sc2(rand() % 256, rand() % 256, rand() % 256, rand() % 256);
    Mat randomMat(7, 8, CV_8UC3, sc2);
    InputArray arr = randomMat;

    AclMat fromInputArray(arr);
    Mat randomMatChecker;
    fromInputArray.download(randomMatChecker);
    EXPECT_MAT_NEAR(randomMat, randomMatChecker, 0.0);

    cv::cann::resetDevice();
}

TEST(AclMat, RefCount)
{
    DummyAllocator dummyAllocator;
    AclMat* mat = new AclMat(1, 1, CV_8U, &dummyAllocator);
    ASSERT_EQ(*(mat->refcount), 1);
    ASSERT_EQ(mat->data, (uchar*)0x12345);

    AclMat* copy1 = new AclMat(*mat);
    ASSERT_EQ(mat->refcount, copy1->refcount);
    ASSERT_EQ(*(copy1->refcount), 2);

    AclMat* copy2 = new AclMat(*copy1);
    ASSERT_EQ(mat->refcount, copy2->refcount);
    ASSERT_EQ(*(copy2->refcount), 3);

    delete copy1;
    ASSERT_EQ(mat->data, (uchar*)0x12345);
    ASSERT_EQ(*(mat->refcount), 2);

    delete copy2;
    ASSERT_EQ(mat->data, (uchar*)0x12345);
    ASSERT_EQ(*(mat->refcount), 1);

    delete mat;
}

TEST(AclMat, Assignment)
{
    DummyAllocator dummyAllocator;
    AclMat mat1;
    AclMat mat2(3, 4, CV_8SC1, &dummyAllocator);
    mat1 = mat2;

    ASSERT_EQ(mat1.rows, 3);
    ASSERT_EQ(mat1.cols, 4);
    ASSERT_EQ(mat1.depth(), CV_8S);
    ASSERT_EQ(mat1.channels(), 1);
    ASSERT_EQ(mat1.data, (uchar*)0x12345);
}

TEST(AclMat, SetTo)
{
    cv::cann::setDevice(0);

    srand((unsigned int)(time(NULL)));
    Scalar sc(rand() % 256, rand() % 256, rand() % 256, rand() % 256);

    AclMat aclMat(2, 2, CV_8UC4);
    aclMat.setTo(sc);
    Mat mat(2, 2, CV_8UC4, sc);
    Mat checker;
    aclMat.download(checker);

    EXPECT_MAT_NEAR(mat, checker, 0.0);

    cv::cann::resetDevice();
}

TEST(AclMat, ConvertTo)
{
    cv::cann::setDevice(0);

    srand((unsigned int)(time(NULL)));
    Scalar sc(rand() % 256, rand() % 256, rand() % 256, rand() % 256);

    AclMat aclMat(2, 2, CV_8UC4, sc);
    AclMat convertedAclMat;
    aclMat.convertTo(convertedAclMat, CV_16S);
    Mat mat(2, 2, CV_16SC4, sc);
    Mat checker;
    convertedAclMat.download(checker);

    EXPECT_MAT_NEAR(mat, checker, 0.0);

    cv::cann::resetDevice();
}

TEST(AclMat, ExpandTo)
{
    cv::cann::setDevice(0);

    Scalar sc1(1);
    Scalar sc2(1, 1, 1);
    AclMat aclMat(10, 10, CV_8UC1, sc1);
    Mat mat(10, 10, CV_8UC3, sc2);
    AclMat expandedAclMat;
    aclMat.expandTo(expandedAclMat, 3);
    Mat checker;
    expandedAclMat.download(checker);

    EXPECT_MAT_NEAR(mat, checker, 0.0);

    cv::cann::resetDevice();
}

TEST(AclStream, AsyncProcess)
{
    cv::cann::setDevice(0);

    DummyAllocator dummyAllocator;
    AclMat* mat = new AclMat(&dummyAllocator);
    AclStream stream;

    stream.addToAsyncRelease(*mat);
    stream.waitForCompletion();

    // TODO: need sync point to check:
    // 1. mat->data is not freed after it add to async release list even mat is deleted.
    // 2. mat->data is freed after callback is called.

    cv::cann::resetDevice();
}

} // namespace
} // namespace opencv_test
