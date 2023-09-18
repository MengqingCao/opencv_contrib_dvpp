// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv
{
namespace cann
{
static void matAlphaSet(AscendMat& mat, int dtype, AscendStream& stream)
{
    if (dtype < 0)
        dtype = mat.depth();

    if (mat.depth() == CV_8U || mat.depth() == CV_16U)
    {
        size_t size = mat.rows * mat.step;
        aclrtMemsetWarpper(mat.data, 255, size, stream);
    }
    else
    {
        if (dtype == CV_32F)
            mat.setTo(1.0f, stream);
        else
        {
            mat.setTo((dtype == CV_8U ? (1 << 8) : (1 << 16)) - 1, stream);
        }
    }
}

inline void checkImg(const AscendMat& mat)
{
    int depth = mat.depth();
    CV_Assert(!mat.empty());
    CV_Assert(depth == CV_8U || depth == CV_16U || depth == CV_32F);
}

inline void cvtBGRtoBGR(InputArray& _src, OutputArray& _dst, int dcn, bool swapBlue,
                        AscendStream& stream)
{
    AscendMat src = getInputMat(_src, stream);
    checkImg(src);
    CV_Assert(src.channels() == 3 || src.channels() == 4);

    AscendMat matChannels[4];
    split(src, matChannels, stream);

    if (swapBlue)
    {
        std::swap(matChannels[0], matChannels[2]);
    }

    if (dcn == 4 && src.channels() != 4)
    {
        AscendMat& alpha = matChannels[3];
        alpha.create(src.rows, src.cols, CV_MAKE_TYPE(src.depth(), 1));
        matAlphaSet(alpha, -1, stream);
    }

    merge(matChannels, dcn, _dst, stream);
}

// TODO duplicated code
static const float B2YF = 0.114f;
static const float G2YF = 0.587f;
static const float R2YF = 0.299f;

inline void cvtBGRtoGray(InputArray& _src, OutputArray& _dst, int, bool swapBlue,
                         AscendStream& stream)
{
    AscendMat src = getInputMat(_src, stream);
    checkImg(src);
    CV_Assert(src.channels() == 3 || src.channels() == 4);

    float coeffs[] = {B2YF, G2YF, R2YF};

    AscendMat formatMat;
    if (src.depth() != CV_32F)
    {
        src.convertTo(formatMat, CV_32F);
    }
    else
    {
        formatMat = src;
    }

    // For RGB
    if (swapBlue)
    {
        std::swap(coeffs[0], coeffs[2]);
    }

    Scalar sc = {coeffs[0], coeffs[1], coeffs[2], 0};
    AscendMat grayRet;
    multiply(formatMat, sc, grayRet, 1, -1, stream);

    AscendMat matChannels[4];
    split(grayRet, matChannels, stream);

    AscendMat dst = getOutputMat(_dst, src.rows, src.cols, CV_MAKE_TYPE(src.depth(), 1), stream);

    AclIntAttribute matSize("N", 3);
    std::vector<AclAttribute*> attrs{&matSize};

    if (src.depth() != CV_32F)
    {
        formatMat.create(grayRet.rows, grayRet.cols, CV_MAKE_TYPE(grayRet.depth(), 1));
        callAscendOperator(matChannels, 3, formatMat, "AddN", stream, attrs);

        // do not use convertTo here, dst.data will overwrited.
        callAscendOperator(formatMat, dst, "Cast", stream);
    }
    else
        callAscendOperator(matChannels, 3, dst, "AddN", stream, attrs);
    syncOutput(dst, _dst, stream);
}

inline void cvtGraytoBGR(InputArray& _src, OutputArray& _dst, int dcn, bool, AscendStream& stream)
{
    AscendMat src = getInputMat(_src, stream);
    checkImg(src);
    CV_Assert(src.channels() == 1);

    AscendMat matChannels[4];
    for (int i = 0; i < 3; i++)
    {
        matChannels[i] = src;
    }

    if (dcn == 4)
    {
        AscendMat& alpha = matChannels[3];
        alpha.create(src.rows, src.cols, CV_MAKE_TYPE(src.depth(), 1));
        matAlphaSet(alpha, -1, stream);
    }

    merge(matChannels, dcn, _dst, stream);
}

static const float RGB2XYZ_D65[] = {0.412453, 0.357580, 0.180423, 0.212671, 0.715160,
                                    0.072169, 0.019334, 0.119193, 0.950227};

static const float XYZ2RGB_D65[] = {3.240479, -1.53715, -0.498535, -0.969256, 1.875991,
                                    0.041556, 0.055648, -0.204043, 1.057311};

inline void matMulRGB(InputArray& _src, OutputArray& _dst, float* matrix, AscendStream& stream)
{
    AscendMat src = getInputMat(_src, stream);
    checkImg(src);
    CV_Assert(src.channels() == 3);

    AscendMat formatMat;
    if (src.depth() != CV_32F)
    {
        src.convertTo(formatMat, CV_32F);
    }
    else
    {
        formatMat = src;
    }

    // TODO async!!!
    Mat transMat(1, 3, CV_32FC3, matrix);
    AscendMat transAscendMat;
    transAscendMat.upload(transMat, stream);

    AclBoolAttribute transposeX1("adj_x1", false);
    AclBoolAttribute transposeX2("adj_x2", true);
    std::vector<AclAttribute*> matMulAttr{&transposeX1, &transposeX2};

    AscendMat dst = getOutputMat(_dst, src.rows, src.cols, src.type(), stream);

    if (src.depth() != CV_32F)
    {
        AscendMat formatRet(formatMat.size(), formatMat.type()),
            thresholdRet(formatMat.size(), formatMat.type());
        callAscendOperator(formatMat, transAscendMat, formatRet, "BatchMatMulV2", stream, matMulAttr);
        uint16_t thresh = (src.depth() == CV_8U ? (1 << 8) : (1 << 16)) - 1;
        threshold(formatRet, thresholdRet, thresh, 0, 2 /*THRESH_TRUNC*/, stream);
        threshold(thresholdRet, formatRet, 0, 0, 3 /*THRESH_TOZERO*/, stream);
        callAscendOperator(formatRet, dst, "Cast", stream);
    }
    else
        callAscendOperator(formatMat, transAscendMat, dst, "BatchMatMulV2", stream, matMulAttr);

    syncOutput(dst, _dst, stream);
}

// TODO should deal with overflow. set 255 instead of cut off.
inline void cvtBGRtoXYZ(InputArray& src, OutputArray& dst, int, bool swapBlue, AscendStream& stream)
{
    float coeffs[9];
    memcpy(coeffs, RGB2XYZ_D65, 9 * sizeof(float));
    if (!swapBlue)
    {
        std::swap(coeffs[0], coeffs[2]);
        std::swap(coeffs[3], coeffs[5]);
        std::swap(coeffs[6], coeffs[8]);
    }
    matMulRGB(src, dst, coeffs, stream);
}

inline void cvtXYZtoBGR(InputArray& src, OutputArray& dst, int dcn, bool swapBlue,
                        AscendStream& stream)
{
    float coeffs[9];
    memcpy(coeffs, XYZ2RGB_D65, 9 * sizeof(float));
    if (!swapBlue)
    {
        std::swap(coeffs[0], coeffs[6]);
        std::swap(coeffs[1], coeffs[7]);
        std::swap(coeffs[2], coeffs[8]);
    }

    if (dcn == 4)
    {
        AscendMat RGB[4], tempMat1;
        matMulRGB(src, tempMat1, coeffs, stream);

        split(tempMat1, RGB, stream);
        RGB[3].create(RGB[0].rows, RGB[1].cols, RGB[0].type());
        matAlphaSet(RGB[3], -1, stream);
        merge(RGB, 4, dst, stream);
    }
    else
        matMulRGB(src, dst, coeffs, stream);
}

// TODO duplicated code
static const float YCRF = 0.713f;
static const float YCBF = 0.564f;
static const float R2VF = 0.877f;
static const float B2UF = 0.492f;
inline void cvtBGRtoYCrCb(InputArray& _src, OutputArray& _dst, float* coeffs, bool swapBlue,
                          bool yuvOrder, AscendStream& stream)
{
    AscendMat src = getInputMat(_src, stream);
    checkImg(src);
    CV_Assert(src.channels() == 3);

    int buleIdx = swapBlue ? 2 : 0;
    int depth = src.depth();
    float delta = (depth == CV_8U) ? 128 : ((depth == CV_16U) ? 32768 : 0.5);

    AscendMat formatMat;
    if (src.depth() != CV_32F)
    {
        src.convertTo(formatMat, CV_32F);
    }
    else
    {
        formatMat = src;
    }

    AscendMat YCrCb[3], RGB[3];
    split(formatMat, RGB, stream);
    cvtBGRtoGray(formatMat, YCrCb[0], 1, swapBlue, stream);
    YCrCb[1].create(YCrCb[0].rows, YCrCb[0].cols, YCrCb[0].type());
    YCrCb[2].create(YCrCb[0].rows, YCrCb[0].cols, YCrCb[0].type());

    AscendMat tempMat1(formatMat.size(), CV_MAKE_TYPE(formatMat.depth(), 1)),
        tempMat2(formatMat.size(), CV_MAKE_TYPE(formatMat.depth(), 1));

    callAscendOperator(RGB[buleIdx ^ 2], YCrCb[0], tempMat1, "Sub", stream);
    muls(tempMat1, coeffs[0], tempMat2, stream);
    adds(tempMat2, delta, YCrCb[1], stream);

    callAscendOperator(RGB[buleIdx], YCrCb[0], tempMat1, "Sub", stream);
    muls(tempMat1, coeffs[1], tempMat2, stream);
    adds(tempMat2, delta, YCrCb[2], stream);

    if (yuvOrder)
    {
        std::swap(YCrCb[1], YCrCb[2]);
    }

    if (src.depth() != CV_32F)
    {
        AscendMat formatRet(formatMat.size(), formatMat.type()),
            thresholdRet(formatMat.size(), formatMat.type());
        merge(YCrCb, 3, formatRet, stream);
        uint16_t thresh = (src.depth() == CV_8U ? (1 << 8) : (1 << 16)) - 1;
        threshold(formatRet, thresholdRet, thresh, 0, 2 /*THRESH_TRUNC*/, stream);
        threshold(thresholdRet, formatRet, 0, 0, 3 /*THRESH_TOZERO*/, stream);
        AscendMat dst = getOutputMat(_dst, src.rows, src.cols, src.type(), stream);
        callAscendOperator(formatRet, dst, "Cast", stream);
        syncOutput(dst, _dst, stream);
    }
    else
        merge(YCrCb, 3, _dst, stream);
}

static const float CR2RF = 1.403f;
static const float CR2GF = -0.714f;
static const float CB2GF = -0.344f;
static const float CB2BF = 1.773f;

static const float V2RF = 1.140f;
static const float V2GF = -0.581f;
static const float U2GF = -0.395f;
static const float U2BF = 2.032f;

inline void cvtYCrCbtoBGR(InputArray& _src, OutputArray& _dst, int dcn, float* coeffs,
                          bool swapBlue, bool yuvOrder, AscendStream& stream)
{
    AscendMat src = getInputMat(_src, stream);
    checkImg(src);
    CV_Assert(src.channels() == 3);

    int buleIdx = swapBlue ? 2 : 0;
    int depth = src.depth();
    float delta = (depth == CV_8U) ? 128 : ((depth == CV_16U) ? 32768 : 0.5);

    AscendMat formatMat;
    if (src.depth() != CV_32F)
    {
        src.convertTo(formatMat, CV_32F);
    }
    else
    {
        formatMat = src;
    }

    AscendMat YCrCb[3], RGB[4];
    split(formatMat, YCrCb, stream);
    if (yuvOrder)
    {
        std::swap(YCrCb[1], YCrCb[2]);
    }
    RGB[0].create(formatMat.rows, formatMat.cols, CV_MAKE_TYPE(formatMat.depth(), 1));
    RGB[1].create(formatMat.rows, formatMat.cols, CV_MAKE_TYPE(formatMat.depth(), 1));
    RGB[2].create(formatMat.rows, formatMat.cols, CV_MAKE_TYPE(formatMat.depth(), 1));
    AscendMat tempMat1(formatMat.size(), CV_MAKE_TYPE(formatMat.depth(), 1)),
        tempMat2(formatMat.size(), CV_MAKE_TYPE(formatMat.depth(), 1)),
        CbSubDelta(formatMat.size(), CV_MAKE_TYPE(formatMat.depth(), 1)),
        CrSubDelta(formatMat.size(), CV_MAKE_TYPE(formatMat.depth(), 1));

    adds(YCrCb[1], (0.0f - delta), CrSubDelta, stream);
    adds(YCrCb[2], (0.0f - delta), CbSubDelta, stream);

    muls(CrSubDelta, coeffs[0], tempMat1, stream);
    callAscendOperator(YCrCb[0], tempMat1, RGB[buleIdx ^ 2], "Add", stream);

    muls(CrSubDelta, coeffs[1], tempMat1, stream);
    callAscendOperator(YCrCb[0], tempMat1, tempMat2, "Add", stream);
    muls(CbSubDelta, coeffs[2], tempMat1, stream);
    callAscendOperator(tempMat2, tempMat1, RGB[1], "Add", stream);

    muls(CbSubDelta, coeffs[3], tempMat1, stream);
    callAscendOperator(YCrCb[0], tempMat1, RGB[buleIdx], "Add", stream);

    if (dcn == 4)
    {
        RGB[3].create(RGB[0].rows, RGB[0].cols, RGB[0].type());
        matAlphaSet(RGB[3], src.depth(), stream);
    }

    if (src.depth() != CV_32F)
    {
        AscendMat formatRet(formatMat.size(), CV_MAKE_TYPE(formatMat.depth(), dcn)),
            thresholdRet(formatMat.size(), CV_MAKE_TYPE(formatMat.depth(), dcn));
        merge(RGB, dcn, formatRet, stream);
        uint16_t thresh = (src.depth() == CV_8U ? (1 << 8) : (1 << 16)) - 1;
        threshold(formatRet, thresholdRet, thresh, 0, 2 /*THRESH_TRUNC*/, stream);
        threshold(thresholdRet, formatRet, 0, 0, 3 /*THRESH_TOZERO*/, stream);
        AscendMat dst = getOutputMat(_dst, src.rows, src.cols, CV_MAKE_TYPE(src.depth(), dcn), stream);
        callAscendOperator(formatRet, dst, "Cast", stream);
        syncOutput(dst, _dst, stream);
    }
    else
        merge(RGB, dcn, _dst, stream);
}

inline void BGR2BGRA(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    cvtBGRtoBGR(src, dst, 4, false, stream);
}

inline void BGRA2BGR(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    cvtBGRtoBGR(src, dst, 3, false, stream);
}

inline void BGR2RGBA(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    cvtBGRtoBGR(src, dst, 4, true, stream);
}

inline void RGBA2BGR(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    cvtBGRtoBGR(src, dst, 3, true, stream);
}

inline void BGR2RGB(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    cvtBGRtoBGR(src, dst, 3, true, stream);
}

inline void BGRA2RGBA(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    cvtBGRtoBGR(src, dst, 4, true, stream);
}

inline void BGR2GRAY(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    cvtBGRtoGray(src, dst, 1, false, stream);
}

inline void RGB2GRAY(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    cvtBGRtoGray(src, dst, 1, true, stream);
}

inline void GRAY2BGR(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    cvtGraytoBGR(src, dst, 3, false, stream);
}

inline void GRAY2BGRA(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    cvtGraytoBGR(src, dst, 4, false, stream);
}

inline void BGRA2GRAY(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    cvtBGRtoGray(src, dst, 1, false, stream);
}

inline void RGBA2GRAY(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    cvtBGRtoGray(src, dst, 1, true, stream);
}

inline void BGR2XYZ(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    cvtBGRtoXYZ(src, dst, 3, false, stream);
}

inline void RGB2XYZ(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    cvtBGRtoXYZ(src, dst, 3, true, stream);
}

inline void XYZ2BGR(InputArray src, OutputArray& dst, int dcn, AscendStream& stream)
{
    if (dcn <= 0)
        dcn = 3;
    cvtXYZtoBGR(src, dst, dcn, false, stream);
}

inline void XYZ2RGB(InputArray src, OutputArray& dst, int dcn, AscendStream& stream)
{
    if (dcn <= 0)
        dcn = 3;
    cvtXYZtoBGR(src, dst, dcn, true, stream);
}

inline void BGR2YCrCb(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    float coeffs[2];
    coeffs[0] = YCRF;
    coeffs[1] = YCBF;
    cvtBGRtoYCrCb(src, dst, coeffs, false, false, stream);
}

inline void RGB2YCrCb(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    float coeffs[2];
    coeffs[0] = YCRF;
    coeffs[1] = YCBF;
    cvtBGRtoYCrCb(src, dst, coeffs, true, false, stream);
}

inline void YCrCb2BGR(InputArray src, OutputArray& dst, int dcn, AscendStream& stream)
{
    float coeffs[4];
    coeffs[0] = CR2RF;
    coeffs[1] = CR2GF;
    coeffs[2] = CB2GF;
    coeffs[3] = CB2BF;
    if (dcn <= 0)
        dcn = 3;
    cvtYCrCbtoBGR(src, dst, dcn, coeffs, false, false, stream);
}

inline void YCrCb2RGB(InputArray src, OutputArray& dst, int dcn, AscendStream& stream)
{
    float coeffs[4];
    coeffs[0] = CR2RF;
    coeffs[1] = CR2GF;
    coeffs[2] = CB2GF;
    coeffs[3] = CB2BF;
    if (dcn <= 0)
        dcn = 3;
    cvtYCrCbtoBGR(src, dst, dcn, coeffs, true, false, stream);
}

inline void BGR2YUV(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    float coeffs[2];
    coeffs[0] = R2VF;
    coeffs[1] = B2UF;
    cvtBGRtoYCrCb(src, dst, coeffs, false, true, stream);
}

inline void RGB2YUV(InputArray src, OutputArray& dst, int, AscendStream& stream)
{
    float coeffs[2];
    coeffs[0] = R2VF;
    coeffs[1] = B2UF;
    cvtBGRtoYCrCb(src, dst, coeffs, true, true, stream);
}

inline void YUV2BGR(InputArray src, OutputArray& dst, int dcn, AscendStream& stream)
{
    float coeffs[4];
    coeffs[0] = V2RF;
    coeffs[1] = V2GF;
    coeffs[2] = U2GF;
    coeffs[3] = U2BF;
    if (dcn <= 0)
        dcn = 3;
    cvtYCrCbtoBGR(src, dst, dcn, coeffs, false, true, stream);
}

inline void YUV2RGB(InputArray src, OutputArray& dst, int dcn, AscendStream& stream)
{
    float coeffs[4];
    coeffs[0] = V2RF;
    coeffs[1] = V2GF;
    coeffs[2] = U2GF;
    coeffs[3] = U2BF;
    if (dcn <= 0)
        dcn = 3;
    cvtYCrCbtoBGR(src, dst, dcn, coeffs, true, true, stream);
}

void cvtColor(InputArray src, OutputArray dst, int code, int dcn, AscendStream& stream)
{
    typedef void (*func_t)(InputArray& src, OutputArray& dst, int dcn, AscendStream& stream);
    static const func_t funcs[] = {
        BGR2BGRA,  // CV_BGR2BGRA    =0
        BGRA2BGR,  // CV_BGRA2BGR    =1
        BGR2RGBA,  // CV_BGR2RGBA    =2
        RGBA2BGR,  // CV_RGBA2BGR    =3
        BGR2RGB,   // CV_BGR2RGB     =4
        BGRA2RGBA, // CV_BGRA2RGBA   =5

        BGR2GRAY,  // CV_BGR2GRAY    =6
        RGB2GRAY,  // CV_RGB2GRAY    =7
        GRAY2BGR,  // CV_GRAY2BGR    =8
        GRAY2BGRA, // CV_GRAY2BGRA   =9
        BGRA2GRAY, // CV_BGRA2GRAY   =10
        RGBA2GRAY, // CV_RGBA2GRAY   =11

        0, // CV_BGR2BGR565  =12
        0, // CV_RGB2BGR565  =13
        0, // CV_BGR5652BGR  =14
        0, // CV_BGR5652RGB  =15
        0, // CV_BGRA2BGR565 =16
        0, // CV_RGBA2BGR565 =17
        0, // CV_BGR5652BGRA =18
        0, // CV_BGR5652RGBA =19

        0, // CV_GRAY2BGR565 =20
        0, // CV_BGR5652GRAY =21

        0, // CV_BGR2BGR555  =22
        0, // CV_RGB2BGR555  =23
        0, // CV_BGR5552BGR  =24
        0, // CV_BGR5552RGB  =25
        0, // CV_BGRA2BGR555 =26
        0, // CV_RGBA2BGR555 =27
        0, // CV_BGR5552BGRA =28
        0, // CV_BGR5552RGBA =29

        0, // CV_GRAY2BGR555 =30
        0, // CV_BGR5552GRAY =31

        BGR2XYZ, // CV_BGR2XYZ     =32
        RGB2XYZ, // CV_RGB2XYZ     =33
        XYZ2BGR, // CV_XYZ2BGR     =34
        XYZ2RGB, // CV_XYZ2RGB     =35

        BGR2YCrCb, // CV_BGR2YCrCb   =36
        RGB2YCrCb, // CV_RGB2YCrCb   =37
        YCrCb2BGR, // CV_YCrCb2BGR   =38
        YCrCb2RGB, // CV_YCrCb2RGB   =39

        0, // CV_BGR2HSV     =40
        0, // CV_RGB2HSV     =41

        0, //                =42
        0, //                =43

        0, // CV_BGR2Lab     =44
        0, // CV_RGB2Lab     =45

        0, // CV_BayerBG2BGR =46
        0, // CV_BayeRGB2BGR =47
        0, // CV_BayerRG2BGR =48
        0, // CV_BayerGR2BGR =49

        0, // CV_BGR2Luv     =50
        0, // CV_RGB2Luv     =51

        0, // CV_BGR2HLS     =52
        0, // CV_RGB2HLS     =53

        0, // CV_HSV2BGR     =54
        0, // CV_HSV2RGB     =55

        0, // CV_Lab2BGR     =56
        0, // CV_Lab2RGB     =57
        0, // CV_Luv2BGR     =58
        0, // CV_Luv2RGB     =59

        0, // CV_HLS2BGR     =60
        0, // CV_HLS2RGB     =61

        0, // CV_BayerBG2BGR_VNG =62
        0, // CV_BayeRGB2BGR_VNG =63
        0, // CV_BayerRG2BGR_VNG =64
        0, // CV_BayerGR2BGR_VNG =65

        0, // CV_BGR2HSV_FULL = 66
        0, // CV_RGB2HSV_FULL = 67
        0, // CV_BGR2HLS_FULL = 68
        0, // CV_RGB2HLS_FULL = 69

        0, // CV_HSV2BGR_FULL = 70
        0, // CV_HSV2RGB_FULL = 71
        0, // CV_HLS2BGR_FULL = 72
        0, // CV_HLS2RGB_FULL = 73

        0, // CV_LBGR2Lab     = 74
        0, // CV_LRGB2Lab     = 75
        0, // CV_LBGR2Luv     = 76
        0, // CV_LRGB2Luv     = 77

        0, // CV_Lab2LBGR     = 78
        0, // CV_Lab2LRGB     = 79
        0, // CV_Luv2LBGR     = 80
        0, // CV_Luv2LRGB     = 81

        BGR2YUV, // CV_BGR2YUV      = 82
        RGB2YUV, // CV_RGB2YUV      = 83
        YUV2BGR, // CV_YUV2BGR      = 84
        YUV2RGB, // CV_YUV2RGB      = 85

        0, // CV_BayerBG2GRAY = 86
        0, // CV_BayeRGB2GRAY = 87
        0, // CV_BayerRG2GRAY = 88
        0, // CV_BayerGR2GRAY = 89

        // YUV 4:2:0 formats family
        0, // CV_YUV2RGB_NV12 = 90,
        0, // CV_YUV2BGR_NV12 = 91,
        0, // CV_YUV2RGB_NV21 = 92,
        0, // CV_YUV2BGR_NV21 = 93,

        0, // CV_YUV2RGBA_NV12 = 94,
        0, // CV_YUV2BGRA_NV12 = 95,
        0, // CV_YUV2RGBA_NV21 = 96,
        0, // CV_YUV2BGRA_NV21 = 97,

        0, // CV_YUV2RGB_YV12 = 98,
        0, // CV_YUV2BGR_YV12 = 99,
        0, // CV_YUV2RGB_IYUV = 100,
        0, // CV_YUV2BGR_IYUV = 101,

        0, // CV_YUV2RGBA_YV12 = 102,
        0, // CV_YUV2BGRA_YV12 = 103,
        0, // CV_YUV2RGBA_IYUV = 104,
        0, // CV_YUV2BGRA_IYUV = 105,

        0, // CV_YUV2GRAY_420 = 106,

        // YUV 4:2:2 formats family
        0, // CV_YUV2RGB_UYVY = 107,
        0, // CV_YUV2BGR_UYVY = 108,
        0, // //CV_YUV2RGB_VYUY = 109,
        0, // //CV_YUV2BGR_VYUY = 110,

        0, // CV_YUV2RGBA_UYVY = 111,
        0, // CV_YUV2BGRA_UYVY = 112,
        0, // //CV_YUV2RGBA_VYUY = 113,
        0, // //CV_YUV2BGRA_VYUY = 114,

        0, // CV_YUV2RGB_YUY2 = 115,
        0, // CV_YUV2BGR_YUY2 = 116,
        0, // CV_YUV2RGB_YVYU = 117,
        0, // CV_YUV2BGR_YVYU = 118,

        0, // CV_YUV2RGBA_YUY2 = 119,
        0, // CV_YUV2BGRA_YUY2 = 120,
        0, // CV_YUV2RGBA_YVYU = 121,
        0, // CV_YUV2BGRA_YVYU = 122,

        0, // CV_YUV2GRAY_UYVY = 123,
        0, // CV_YUV2GRAY_YUY2 = 124,

        // alpha premultiplication
        0, // CV_RGBA2mRGBA = 125,
        0, // CV_mRGBA2RGBA = 126,

        0, // CV_COLORCVT_MAX  = 127
    };

    CV_Assert(code < 128);

    func_t func = funcs[code];

    if (func == 0)
        CV_Error(Error::StsBadFlag, "Unknown/unsupported color conversion code");

    func(src, dst, dcn, stream);
}

} // namespace cann
} // namespace cv