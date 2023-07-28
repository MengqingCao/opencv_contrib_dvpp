// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/cann_arithm.hpp"
using namespace perf;

class CannEnvironment : public ::testing::Environment
{
public:
    virtual ~CannEnvironment() = default;
    virtual void SetUp() CV_OVERRIDE {
        cv::cann::initAcl();

        // for device warmup
        Scalar s1(1,2,3), s2(4,5,6);
        Mat src1(10, 10, CV_32SC3, s1), src2(10, 10, CV_32SC3, s2);
        cv::cann::setDevice(0);

        cv::cann::AclMat npu_src1, npu_src2, dst;
        npu_src1.upload(src1);
        npu_src2.upload(src2);
        cv::cann::add(npu_src1, npu_src2, dst);
        cv::cann::resetDevice();
        }
    virtual void TearDown() CV_OVERRIDE { cv::cann::finalizeAcl(); }
};

static void initTests()
{
    CannEnvironment* cannEnv = new CannEnvironment();
    ::testing::AddGlobalTestEnvironment(cannEnv);
}

CV_PERF_TEST_MAIN("cannarithm", initTests())
