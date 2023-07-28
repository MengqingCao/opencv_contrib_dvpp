// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv
{
namespace cann
{
void aclOneInput(const AclMat& src, AclMat& dst, const char* op, AclStream& stream,
                 std::vector<AclAttribute*>& attrs)
{
    CannPreparation prepare;
    for (auto& attrIterator : attrs)
    {
        attrIterator->addAttr(prepare.opAttr_);
    }

    int64_t dimSrc[] = {1, src.rows, src.cols, src.channels()};
    int64_t dimDst[] = {1, dst.rows, dst.cols, dst.channels()};
    CANN_PREPARE_INPUTDESC(prepare, getACLType(src.depth()), sizeof(dimSrc) / sizeof(dimSrc[0]),
                           dimSrc, ACL_FORMAT_NHWC);
    CANN_PREPARE_OUTPUTDESC(prepare, getACLType(dst.depth()), sizeof(dimDst) / sizeof(dimDst[0]),
                            dimDst, ACL_FORMAT_NHWC);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<uchar*>(src.data), src.rows * src.step);
    CANN_PREPARE_OUTPUTBUFFER(prepare, const_cast<uchar*>(dst.data), dst.rows * dst.step);

    aclrtStream rawStream = AclStreamAccessor::getStream(stream);

    CV_ACL_SAFE_CALL(aclopCompileAndExecute(
        op, prepare.inputDesc_.size(), prepare.inputDesc_.data(), prepare.inputBuffers_.data(),
        prepare.outputDesc_.size(), prepare.outputDesc_.data(), prepare.outputBuffers_.data(),
        prepare.opAttr_, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, rawStream));
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtSynchronizeStream(rawStream));
    else
    {
        stream.addToAsyncRelease(src);
        stream.addToAsyncRelease(dst);
    }
}

void aclTwoInputs(const AclMat& src1, const AclMat& src2, AclMat& dst, const char* op,
                  AclStream& stream)
{
    CannPreparation prepare;
    aclrtStream rawStream = AclStreamAccessor::getStream(stream);

    int64_t dimSrc1[] = {1, src1.rows, src1.cols, src1.channels()};
    int64_t dimSrc2[] = {1, src2.rows, src2.cols, src2.channels()};

    int64_t dimDst[] = {1, dst.rows, dst.cols, dst.channels()};

    CANN_PREPARE_INPUTDESC(prepare, getACLType(src1.depth()), sizeof(dimSrc1) / sizeof(dimSrc1[0]),
                           dimSrc1, ACL_FORMAT_NHWC);

    CANN_PREPARE_INPUTDESC(prepare, getACLType(src2.depth()), sizeof(dimSrc2) / sizeof(dimSrc2[0]),
                           dimSrc2, ACL_FORMAT_NHWC);

    CANN_PREPARE_OUTPUTDESC(prepare, getACLType(dst.depth()), sizeof(dimDst) / sizeof(dimDst[0]),
                            dimDst, ACL_FORMAT_NHWC);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<uchar*>(src1.data), src1.rows * src1.step);
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<uchar*>(src2.data), src2.rows * src2.step);
    CANN_PREPARE_OUTPUTBUFFER(prepare, const_cast<uchar*>(dst.data), dst.rows * dst.step);

    CV_ACL_SAFE_CALL(aclopCompileAndExecute(
        op, prepare.inputDesc_.size(), prepare.inputDesc_.data(), prepare.inputBuffers_.data(),
        prepare.outputDesc_.size(), prepare.outputDesc_.data(), prepare.outputBuffers_.data(),
        prepare.opAttr_, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, rawStream));
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtSynchronizeStream(rawStream));
    else
    {
        stream.addToAsyncRelease(src1);
        stream.addToAsyncRelease(src2);
        stream.addToAsyncRelease(dst);
    }
}

void transNCHWToNHWC(const AclMat& src, AclMat& dst, AclStream& stream)
{
    CannPreparation prepare;
    CANN_PREPARE_ADD_ATTR(prepare, String, "src_format", "NCHW");
    CANN_PREPARE_ADD_ATTR(prepare, String, "dst_format", "NHWC");

    int64_t dimSrc[] = {1, src.channels(), src.rows, src.cols};
    int64_t dimDst[] = {1, dst.rows, dst.cols, dst.channels()};

    CANN_PREPARE_INPUTDESC(prepare, getACLType(src.depth()), sizeof(dimSrc) / sizeof(dimSrc[0]),
                           dimSrc, ACL_FORMAT_NCHW);
    CANN_PREPARE_OUTPUTDESC(prepare, getACLType(dst.depth()), sizeof(dimDst) / sizeof(dimDst[0]),
                            dimDst, ACL_FORMAT_NHWC);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<uchar*>(src.data), src.rows * src.step);
    CANN_PREPARE_OUTPUTBUFFER(prepare, const_cast<uchar*>(dst.data), dst.rows * dst.step);

    aclrtStream rawStream = AclStreamAccessor::getStream(stream);

    CV_ACL_SAFE_CALL(aclopCompileAndExecute("TransData", prepare.inputDesc_.size(),
                                            prepare.inputDesc_.data(), prepare.inputBuffers_.data(),
                                            prepare.outputDesc_.size(), prepare.outputDesc_.data(),
                                            prepare.outputBuffers_.data(), prepare.opAttr_,
                                            ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, rawStream));
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtSynchronizeStream(rawStream));
    else
    {
        stream.addToAsyncRelease(src);
        stream.addToAsyncRelease(dst);
    }
}

aclDataType getACLType(int opencvdepth)
{
    switch (opencvdepth)
    {
        case CV_8S:
            return ACL_INT8;
        case CV_16S:
            return ACL_INT16;
        case CV_8U:
            return ACL_UINT8;
        case CV_16U:
            return ACL_UINT16;
        case CV_32S:
            return ACL_INT32;
        case CV_64F:
            return ACL_DOUBLE;
        case CV_16F:
            return ACL_FLOAT16;
        default:
            return ACL_DT_UNDEFINED;
    }
}

} // namespace cann
} // namespace cv
