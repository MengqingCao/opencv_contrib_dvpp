// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv
{
namespace cann
{

static inline void applyMask(const AscendMat& src, AscendMat& dst, AscendMat& mask,
                             AscendStream& stream)
{
    int mtype = mask.type();
    CV_Assert((mtype == CV_8UC1 || mtype == CV_8SC1) && mask.size() == src.size());
    AscendMat onesMask, castedMask;
    onesMask.create(mask.rows, mask.cols, mask.type());

    OperatorRunner runner;
    runner.setOp("Div")
        .addInput(mask, "x1")
        .addInput(mask, "x2")
        .addOutput(onesMask, "y")
        .run(stream);

    onesMask.convertTo(castedMask, dst.depth(), stream);
    arithm_op(src, castedMask, dst, "Mul", stream);
}

static inline void applyScale(const AscendMat& src, AscendMat& dst, float scale,
                              AscendStream& stream)
{
    OperatorRunner runner;
    arithm_op(src, scale, dst, "Muls", stream);
}

void arithm_op(const AscendMat& src1, const AscendMat& src2, AscendMat& dst, const char* op,
               AscendStream& stream)
{
    OperatorRunner runner;
    runner.setOp(op).addInput(src1, "x1").addInput(src2, "x2").addOutput(dst, "y").run(stream);
}

void arithm_op(const AscendMat& src, const Scalar& sc, AscendMat& dst, const char* op,
               AscendStream& stream)
{
    OperatorRunner runner;
    runner.setOp(op)
        .addInput(src, "x1")
        .addInput(sc, src.type(), "x2")
        .addOutput(dst, "y")
        .run(stream);
}

void arithm_op(const Scalar& sc, const AscendMat& src, AscendMat& dst, const char* op,
               AscendStream& stream)
{
    OperatorRunner runner;
    runner.setOp(op)
        .addInput(sc, src.type(), "x1")
        .addInput(src, "x2")
        .addOutput(dst, "y")
        .run(stream);
}

void arithm_op(const AscendMat& src, AscendMat& dst, const char* op, AscendStream& stream)
{
    OperatorRunner runner;
    runner.setOp(op).addInput(src, "x").addOutput(dst, "y").run(stream);
}

void arithm_op(const AscendMat& src, float scalar, AscendMat& dst, const char* op,
               AscendStream& stream)
{
    OperatorRunner runner;
    runner.setOp(op).addInput(src, "x").addAttr(scalar, "value").addOutput(dst, "y").run(stream);
}

static void arithm_op(InputArray _src1, InputArray _src2, OutputArray _dst, InputArray _mask,
                      float scale, int dtype, const char* op, AscendStream& stream)
{
    const bool isScalar1 = (_src1.kind() == _InputArray::MATX);
    const bool isScalar2 = (_src2.kind() == _InputArray::MATX);

    if (isScalar1 && isScalar2)
        CV_Error(Error::StsBadArg, "At list one matrix parameter shoule be passwd.");

    AscendMat src1, src2;
    Mat scalar;

    if (!isScalar1)
        src1 = getInputMat(_src1, stream);
    if (!isScalar2)
        src2 = getInputMat(_src2, stream);

    if (isScalar1)
        scalar = _src1.getMat();
    else if (isScalar2)
        scalar = _src2.getMat();

    const int sdepth = src1.empty() ? src2.depth() : src1.depth();
    const int cn = src1.empty() ? src2.channels() : src1.channels();
    const Size size = src1.empty() ? src2.size() : src1.size();

    if (dtype < 0)
        dtype = sdepth;

    const int ddepth = CV_MAT_DEPTH(dtype);
    CV_Assert(sdepth <= CV_16F && ddepth <= CV_16F);
    CV_Assert(!scalar.empty() || src2.empty() ||
              (src2.depth() == src1.depth() && src2.size() == src1.size()));

    Scalar val;

    if (!scalar.empty())
    {
        CV_Assert(scalar.total() <= 4);
        scalar.convertTo(Mat_<double>(scalar.rows, scalar.cols, &val[0]), CV_64F);
    }

    AscendMat dst = getOutputMat(_dst, size.height, size.width, CV_MAKE_TYPE(ddepth, cn), stream);

    AscendMat castedSrc1, castedSrc2, castedRet;
    if (scale != 1 && dtype < CV_32F)
    {
        castedRet.create(size.height, size.width, CV_MAKE_TYPE(CV_32F, cn));
        if (!isScalar1)
            src1.convertTo(castedSrc1, CV_32F, stream);

        if (!isScalar2)
            src2.convertTo(castedSrc2, CV_32F, stream);
    }
    else
    {
        castedSrc1 = src1;
        castedSrc2 = src2;
        castedRet = dst;
    }

    OperatorRunner runner;
    if (isScalar1)
        arithm_op(val, castedSrc2, castedRet, op, stream);
    else if (isScalar2)
        arithm_op(castedSrc1, val, castedRet, op, stream);
    else
    {
        if (src2.empty())
            arithm_op(castedSrc1, castedRet, op, stream);
        else
            arithm_op(castedSrc1, castedSrc2, castedRet, op, stream);
    }

    AscendMat mask = getInputMat(_mask, stream);
    if (!mask.empty())
        applyMask(castedRet, castedRet, mask, stream);

    if (scale != 1)
        applyScale(castedRet, castedRet, scale, stream);

    if (castedRet.depth() != dst.depth())
    {
        runner.setOp("Round").addInput(castedRet, "x").addOutput(castedRet, "y").run(stream);
        castedRet.convertTo(dst, stream);
    }

    syncOutput(dst, _dst, stream);
}

void add(InputArray src1, InputArray src2, OutputArray dst, InputArray mask, int dtype,
         AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, dtype, "Add", stream);
}

void subtract(InputArray src1, InputArray src2, OutputArray dst, InputArray mask, int dtype,
              AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, dtype, "Sub", stream);
}

void multiply(InputArray src1, InputArray src2, OutputArray dst, float scale, int dtype,
              AscendStream& stream)
{
    arithm_op(src1, src2, dst, noArray(), scale, dtype, "Mul", stream);
}

void divide(InputArray src1, InputArray src2, OutputArray dst, float scale, int dtype,
            AscendStream& stream)
{
    arithm_op(src1, src2, dst, noArray(), scale, dtype, "RealDiv", stream);
}

void bitwise_and(InputArray src1, InputArray src2, OutputArray dst, InputArray mask,
                 AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseAnd", stream);
}

void bitwise_or(InputArray src1, InputArray src2, OutputArray dst, InputArray mask,
                AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseOr", stream);
}

void bitwise_xor(InputArray src1, InputArray src2, OutputArray dst, InputArray mask,
                 AscendStream& stream)
{
    arithm_op(src1, src2, dst, mask, 1, -1, "BitwiseXor", stream);
}

void bitwise_not(InputArray src, OutputArray dst, InputArray mask, AscendStream& stream)
{
    arithm_op(src, noArray(), dst, mask, 1, -1, "Invert", stream);
}

void addWeighted(InputArray _src1, double alpha, InputArray _src2, double beta, double gamma,
                 OutputArray _dst, int dtype, AscendStream& stream)
{
    AscendMat src1, src2;
    src1 = getInputMat(_src1, stream);
    src2 = getInputMat(_src2, stream);

    if (dtype < 0)
        dtype = src1.depth();

    CV_Assert(src2.depth() == src1.depth() && src2.size() == src1.size() &&
              src1.channels() == src2.channels());

    int type = CV_MAKE_TYPE(dtype, src1.channels());
    AscendMat dst = getOutputMat(_dst, src1.rows, src1.cols, type, stream);

    // TODO Consider overflow, should extend type or not?
    AscendMat src1Weighted(src1.size(), type), src2Weighted(src1.size(), type),
        srcWeightedSumRet(src1.size(), type);

    arithm_op(src1, (float)alpha, src1Weighted, "Muls", stream);
    arithm_op(src2, (float)beta, src2Weighted, "Muls", stream);
    arithm_op(src1Weighted, src2Weighted, srcWeightedSumRet, "Add", stream);
    arithm_op(srcWeightedSumRet, (float)gamma, dst, "Adds", stream);

    syncOutput(dst, _dst, stream);
}

double threshold(AscendMat& src, AscendMat& dst, double thresh, double maxval, int type,
                 AscendStream& stream)
{
    // ThresholdTypes is defined in opencv2/imgproc, This type is the only Symbol we need.
    // Add imgproc to dependence is too heavy, use magic number instead.
    CV_Assert(type <= 4 /*THRESH_TOZERO_INV*/);

    AscendMat threshMat(src.size(), src.type());

    OperatorRunner runner;
    runner.setOp("Threshold")
        .addInput(src, "x")
        .addOutput(threshMat, "y")
        .addAttr((float)thresh, "threshold")
        .run(stream);

    // THRESH_*_INV, THRESH_TRUNC need a inverse threshMat.
    // THRESH_BINARY_INV = 1, THRESH_TRUNC = 2, THRESH_TOZERO_INV = 4,
    if (type == 1 || type == 2 || type == 4)
    {
        AscendMat threshInvMat(src.size(), src.type());
        AscendMat ones(src.size(), src.type());
        Scalar s(1, 1, 1, 1);
        ones.setTo(s, stream);
        arithm_op(ones, threshMat, threshInvMat, "Sub", stream);

        if (type == 1)
            arithm_op(threshInvMat, (float)maxval, dst, "Muls", stream);
        else if (type == 2)
        {
            AscendMat ToZeroInvMat(src.size(), src.type());
            AscendMat TruncMat(src.size(), src.type());
            arithm_op(threshInvMat, src, ToZeroInvMat, "Mul", stream);
            arithm_op(threshMat, (float)thresh, TruncMat, "Muls", stream);
            arithm_op(ToZeroInvMat, TruncMat, dst, "Add", stream);
        }
        else
            arithm_op(threshInvMat, src, dst, "Mul", stream);
    }
    else
    {
        if (type == 0) /* THRESH_BINARY = 0 */
            arithm_op(threshMat, (float)maxval, dst, "Muls", stream);
        else if (type == 3) /* THRESH_TOZERO = 3 */
            arithm_op(threshMat, src, dst, "Mul", stream);
        else
            CV_Error(Error::AscendApiCallError, "Unknown/unsupported threshold type");
    }
    return thresh;
}

double threshold(InputArray _src, OutputArray _dst, double thresh, double maxval, int type,
                 AscendStream& stream)
{
    AscendMat src = getInputMat(_src, stream);
    AscendMat dst = getOutputMat(_dst, src.rows, src.cols, src.type(), stream);
    double ret = threshold(src, dst, thresh, maxval, type, stream);
    syncOutput(dst, _dst, stream);
    return ret;
}

} // namespace cann
} // namespace cv
