// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv
{
namespace cann
{
static void arithm_op(InputArray _src1, InputArray _src2, OutputArray _dst, InputArray _mask,
                      float scale, int dtype, const char* op, AscendStream& stream)
{
    const bool isScalar1 = (_src1.kind() == _InputArray::MATX);
    const bool isScalar2 = (_src2.kind() == _InputArray::MATX);

    if (isScalar1 && isScalar2)
        CV_Error(Error::StsBadArg, "At list one matrix parameter shoule be passwd.");

    NpuMat src1, src2;
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

    NpuMat dst = getOutputMat(_dst, size.height, size.width, CV_MAKE_TYPE(ddepth, cn), stream);

    if (isScalar1)
        callAscendOperator(src2, val, true, dst, op, stream);
    else if (isScalar2)
        callAscendOperator(src1, val, false, dst, op, stream);
    else
    {
        if (src2.empty())
            callAscendOperator(src1, dst, op, stream);
        else
            callAscendOperator(src1, src2, dst, op, stream);
    }

    NpuMat mask = getInputMat(_mask, stream);
    if (!mask.empty())
    {
        int mtype = mask.type();
        CV_Assert((mtype == CV_8UC1 || mtype == CV_8SC1) && mask.size() == size);
        NpuMat onesMask, castedMask;
        onesMask.create(mask.rows, mask.cols, mask.type());
        callAscendOperator(mask, mask, onesMask, "Div", stream);
        onesMask.convertTo(castedMask, dst.depth(), stream);
        callAscendOperator(dst, castedMask, dst, "Mul", stream);
    }

    if (scale != 1)
    {
        muls(dst, scale, dst, stream);
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
    arithm_op(src1, src2, dst, noArray(), scale, dtype, "Div", stream);
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
    NpuMat src1, src2;
    src1 = getInputMat(_src1, stream);
    src2 = getInputMat(_src2, stream);

    if (dtype < 0)
        dtype = src1.depth();

    CV_Assert(src2.depth() == src1.depth() && src2.size() == src1.size() &&
              src1.channels() == src2.channels());

    int type = CV_MAKE_TYPE(dtype, src1.channels());
    NpuMat dst = getOutputMat(_dst, src1.rows, src1.cols, type, stream);

    // TODO Consider overflow, should extend type or not?
    NpuMat src1Weighted(src1.size(), type), src2Weighted(src1.size(), type),
        srcWeightedSumRet(src1.size(), type);
    muls(src1, alpha, src1Weighted, stream);
    muls(src2, beta, src2Weighted, stream);
    callAscendOperator(src1Weighted, src2Weighted, srcWeightedSumRet, "Add", stream);
    adds(srcWeightedSumRet, gamma, dst, stream);

    syncOutput(dst, _dst, stream);
}

double threshold(NpuMat& src, NpuMat& dst, double thresh, double maxval, int type,
                 AscendStream& stream)
{
    // ThresholdTypes is defined in opencv2/imgproc, This type is the only Symbol we need.
    // Add imgproc to dependence is too heavy, use magic number instead.
    CV_Assert(type <= 4 /*THRESH_TOZERO_INV*/);

    NpuMat threshMat(src.size(), src.type());

    AclFloatAttribute attr("threshold", (float)thresh);
    std::vector<AclAttribute*> attrs{&attr};
    callAscendOperator(src, threshMat, "Threshold", stream, attrs);

    // THRESH_*_INV, THRESH_TRUNC need a inverse threshMat.
    // THRESH_BINARY_INV = 1, THRESH_TRUNC = 2, THRESH_TOZERO_INV = 4,
    if (type == 1 || type == 2 || type == 4)
    {
        NpuMat threshInvMat(src.size(), src.type());
        NpuMat ones(src.size(), src.type());
        Scalar s(1, 1, 1, 1);
        ones.setTo(s, stream);
        callAscendOperator(ones, threshMat, threshInvMat, "Sub", stream);

        if (type == 1)
        {
            muls(threshInvMat, maxval, dst, stream);
        }
        else if (type == 2)
        {
            NpuMat ToZeroInvMat(src.size(), src.type());
            NpuMat TruncMat(src.size(), src.type());
            callAscendOperator(threshInvMat, src, ToZeroInvMat, "Mul", stream);
            muls(threshMat, thresh, TruncMat, stream);
            callAscendOperator(ToZeroInvMat, TruncMat, dst, "Add", stream);
        }
        else
        {
            callAscendOperator(threshInvMat, src, dst, "Mul", stream);
        }
    }
    else
    {
        if (type == 0) /* THRESH_BINARY = 0 */
        {
            muls(threshMat, maxval, dst, stream);
        }
        else if (type == 3) /* THRESH_TOZERO = 3 */
        {
            callAscendOperator(threshMat, src, dst, "Mul", stream);
        }
        else
        {
            CV_Error(Error::AscendApiCallError, "Unknown/unsupported threshold type");
        }
    }
    return thresh;
}

double threshold(InputArray _src, OutputArray _dst, double thresh, double maxval, int type,
                 AscendStream& stream)
{
    NpuMat src = getInputMat(_src, stream);
    NpuMat dst = getOutputMat(_dst, src.rows, src.cols, src.type(), stream);
    double ret = threshold(src, dst, thresh, maxval, type, stream);
    syncOutput(dst, _dst, stream);
    return ret;
}

#define OpScalar(name, op)                                                        \
    void name(const NpuMat& arr, float scalar, NpuMat& dst, AscendStream& stream) \
    {                                                                             \
        AclFloatAttribute attr("value", scalar);                                  \
        std::vector<AclAttribute*> attrs{&attr};                                  \
        callAscendOperator(arr, dst, #op, stream, attrs);                         \
    }

OpScalar(muls, Muls);
OpScalar(adds, Adds);

} // namespace cann
} // namespace cv
