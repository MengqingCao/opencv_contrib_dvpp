// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <map>

namespace cv
{
namespace cann
{
void opMatMat(AclMat&, AclMat&, AclMat&, const char*, AclStream& stream = AclStream::Null());
void opMatMat(AclMat& src1, AclMat& src2, AclMat& dst, const char* op, AclStream& stream)
{
    aclTwoInputs(src1, src2, dst, op, stream);
}

void opMatScalar(AclMat&, AclMat&, bool, Scalar, const char*,
                 AclStream& stream = AclStream::Null());
void opMatScalar(AclMat& src, AclMat& dst, bool inv, Scalar s, const char* op, AclStream& stream)
{
    Mat scMat(1, 1, src.type(), s);
    AclMat scAclMat;
    scAclMat.upload(scMat);
    if (inv)
        aclTwoInputs(scAclMat, src, dst, op, stream);
    else
        aclTwoInputs(src, scAclMat, dst, op, stream);
}

void arithm_op(InputArray _src1, InputArray _src2, OutputArray _dst, InputArray _mask, float scale, int dtype,
               const char* op, AclStream& stream = AclStream::Null());
void arithm_op(InputArray _src1, InputArray _src2, OutputArray _dst, InputArray _mask, float scale,  int dtype,
               const char* op, AclStream& stream)
{
    const int kind1 = _src1.kind();
    const int kind2 = _src2.kind();

    const bool isScalar1 = (kind1 == _InputArray::MATX);
    const bool isScalar2 = (kind2 == _InputArray::MATX);

    AclMat src1, src2;

    if (!isScalar1)
        src1 = getInputMat(_src1, stream);

    if (!isScalar2)
        src2 = getInputMat(_src2, stream);

    Mat scalar;
    if (isScalar1)
        scalar = _src1.getMat();
    else if (isScalar2)
        scalar = _src2.getMat();

    Scalar val;
    if (!scalar.empty())
    {
        CV_Assert(scalar.total() <= 4);
        scalar.convertTo(Mat_<double>(scalar.rows, scalar.cols, &val[0]), CV_64F);
    }

    const int sdepth = src1.empty() ? src2.depth() : src1.depth();
    const int cn = src1.empty() ? src2.channels() : src1.channels();
    const Size size = src1.empty() ? src2.size() : src1.size();

    if (dtype < 0)
        dtype = sdepth;

    const int ddepth = CV_MAT_DEPTH(dtype);

    CV_Assert(sdepth <= CV_64F && ddepth <= CV_64F);
    CV_Assert(!scalar.empty() || (src2.depth() == src1.depth() && src2.size() == src1.size()));

    AclMat dst = getOutputMat(_dst, size.height, size.width, CV_MAKE_TYPE(ddepth, cn));

    if (isScalar1)
        opMatScalar(src2, dst, true, val, op, stream);
    else if (isScalar2)
        opMatScalar(src1, dst, false, val, op, stream);
    else
        opMatMat(src1, src2, dst, op, stream);

    // TODO implement emtpy for AclMat in InputArray
    AclMat mask = getInputMat(_mask, stream);
    if (!mask.empty())
    {
        int mtype = mask.type();

        CV_Assert((mtype == CV_8UC1 || mtype == CV_8SC1) && mask.size() == size);
        // TODO use MaskSelect?
        AclMat formatedMask;
        if (mask.depth() != dst.depth())
            mask.convertTo(formatedMask, dst.depth());
        else
            formatedMask = mask;

        AclMat expandedMask;
        if (dst.channels() != 1)
            formatedMask.expandTo(expandedMask, dst.channels());
        else
            expandedMask = formatedMask;

        // TODO call DIV before expand?
        AclMat divRet;
        arithm_op(expandedMask, expandedMask, divRet, noArray(), 1, -1, "Div", stream);
        AclMat dstCopy = dst;
        // TODO dst memory and dskCopy mempry point to a same memory area, seems no harm yet.
        arithm_op(dstCopy, divRet, dst, noArray(), 1,  -1, "Mul", stream);
    }

    if(scale != 1)
    {
        AclMat dstCpy = dst;
        AclFloatAttribute scaleOP("value", scale);
        std::vector<AclAttribute*> attrs{&scaleOP};
        aclOneInput(dstCpy, dst, "Muls", stream, attrs);
    }

    syncOutput(dst, _dst);
}

void add(InputArray src1, InputArray src2, OutputArray dst, InputArray mask, int dtype,
         AclStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, dtype, "Add", stream);
}

void subtract(InputArray src1, InputArray src2, OutputArray dst, InputArray mask, int dtype,
              AclStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, dtype, "Sub", stream);
}

void multiply(InputArray src1, InputArray src2, OutputArray dst, float scale, int dtype, AclStream& stream)
{
    arithm_op(src1, src2, dst, noArray(), scale, dtype, "Mul", stream);
}

void divide(InputArray src1, InputArray src2, OutputArray dst, float scale, int dtype, AclStream& stream)
{
    arithm_op(src1, src2, dst, noArray(), scale, dtype, "Div", stream);
}

void bitwise_and(InputArray src1, InputArray src2, OutputArray dst, InputArray mask,
                 AclStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseAnd", stream);
}

void bitwise_or(InputArray src1, InputArray src2, OutputArray dst, InputArray mask,
                AclStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseOr", stream);
}

void bitwise_xor(InputArray src1, InputArray src2, OutputArray dst, InputArray mask,
                 AclStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseXor", stream);
}


} // namespace cann
} // namespace cv
