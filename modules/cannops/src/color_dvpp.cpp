// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <iostream>

namespace cv
{
namespace cann
{
void cvt(InputArray _src, OutputArray _dst, hi_pixel_format srcCode, hi_pixel_format dstCode, uint32_t dcn)
{
    Size ssize = _src.size();
    CV_Assert(!ssize.empty());

    Mat src = _src.getMat();
    _dst.create(src.rows, src.cols, CV_MAKE_TYPE(src.depth(), dcn));
    Mat dst = _dst.getMat();

    DvppOperatorRunner op;
    op.Init();
    op.chnId = 0;
    op.stChnAttr = {};
    op.createChannel();

    uint32_t taskID = 0;
    int32_t sizeIn[] = {src.rows, src.cols};
    op.inputPic.picture_format = srcCode;
    op.outputPic.picture_format = dstCode;

    op.setMemAlign(&op.inputPic).setPic(sizeIn, &op.inputPic).addInput(src);
    // std::cout << "op.inputPic = " << op.inputPic.picture_buffer_size << std::endl;

    op.setMemAlign(&op.outputPic).setPic(sizeIn, &op.outputPic).addOutput(dst);
    // std::cout << "op.outputPic = " << op.outputPic.picture_buffer_size << std::endl;

    uint32_t ret = hi_mpi_vpc_convert_color(op.chnId, &op.inputPic, &op.outputPic, &taskID, -1);
    // std::cout << "ret = " << ret << std::endl;
    if (ret != 0)
        CV_Error(Error::StsBadFlag, "failed to convert color");

    uint32_t taskIDResult = taskID;
    op.getResult(dst, taskIDResult);
}

template <typename SRC, typename DST>
inline void BGR2RGB(const SRC& src, DST& dst, uint32_t dcn)
{
    cvt(src, dst, HI_PIXEL_FORMAT_BGR_888, HI_PIXEL_FORMAT_RGB_888, dcn);
}

template <typename SRC, typename DST>
inline void BGR2GRAY(const SRC& src, DST& dst, uint32_t dcn)
{
    cvt(src, dst, HI_PIXEL_FORMAT_BGR_888, HI_PIXEL_FORMAT_YUV_400, dcn);
}
template <typename SRC, typename DST>
inline void GRAY2BGR(const SRC& src, DST& dst, uint32_t dcn)
{
    cvt(src, dst, HI_PIXEL_FORMAT_YUV_400, HI_PIXEL_FORMAT_BGR_888, dcn);
}

template <typename SRC, typename DST>
inline void BGR2BGRA(const SRC& src, DST& dst, uint32_t dcn)
{
    cvt(src, dst, HI_PIXEL_FORMAT_BGR_888, HI_PIXEL_FORMAT_BGRA_8888, dcn);
}
template <typename SRC, typename DST>
inline void BGRA2BGR(const SRC& src, DST& dst, uint32_t dcn)
{
    cvt(src, dst, HI_PIXEL_FORMAT_BGRA_8888, HI_PIXEL_FORMAT_BGR_888, dcn);
}
template <typename SRC, typename DST>
inline void GRAY2BGRA(const SRC& src, DST& dst, uint32_t dcn)
{
    cvt(src, dst, HI_PIXEL_FORMAT_YUV_400, HI_PIXEL_FORMAT_BGRA_8888, dcn);
}
template <typename SRC, typename DST>
inline void BGRA2GRAY(const SRC& src, DST& dst, uint32_t dcn)
{
    cvt(src, dst, HI_PIXEL_FORMAT_BGRA_8888, HI_PIXEL_FORMAT_YUV_400, dcn);
}

template <typename SRC, typename DST>
inline void BGR2RGBA(const SRC& src, DST& dst, uint32_t dcn)
{
    cvt(src, dst, HI_PIXEL_FORMAT_BGR_888, HI_PIXEL_FORMAT_RGBA_8888, dcn);
}
template <typename SRC, typename DST>
inline void RGBA2BGR(const SRC& src, DST& dst, uint32_t dcn)
{
    cvt(src, dst, HI_PIXEL_FORMAT_RGBA_8888, HI_PIXEL_FORMAT_BGR_888, dcn);
}
template <typename SRC, typename DST>
inline void RGBA2GRAY(const SRC& src, DST& dst, uint32_t dcn)
{
    cvt(src, dst, HI_PIXEL_FORMAT_RGBA_8888, HI_PIXEL_FORMAT_YUV_400, dcn);
}

template <typename SRC, typename DST>
inline void BGR2YUV(const SRC& src, DST& dst, uint32_t dcn)
{
    cvt(src, dst, HI_PIXEL_FORMAT_BGR_888, HI_PIXEL_FORMAT_YUV_PACKED_444, dcn);
}
template <typename SRC, typename DST>
inline void BGR2YCrCb(const SRC& src, DST& dst, uint32_t dcn)
{
    cvt(src, dst, HI_PIXEL_FORMAT_BGR_888, HI_PIXEL_FORMAT_YVU_SEMIPLANAR_444, dcn);
}
template <typename SRC, typename DST>
inline void RGB2YCrCb(const SRC& src, DST& dst, uint32_t dcn)
{
    cvt(src, dst, HI_PIXEL_FORMAT_RGB_888, HI_PIXEL_FORMAT_YVU_SEMIPLANAR_444, dcn);
}
template <typename SRC, typename DST>
inline void RGB2GRAY(const SRC& src, DST& dst, uint32_t dcn)
{
    cvt(src, dst, HI_PIXEL_FORMAT_RGB_888, HI_PIXEL_FORMAT_YUV_400, dcn);
}

template <typename SRC, typename DST>
void cvtColorDo(const SRC& src, DST& dst, int code, uint32_t dcn)
{
    typedef void (*dvppFunc_t)(const SRC& src, DST& dst, uint32_t dcn);
    static const dvppFunc_t funcs[] = {
        BGR2BGRA, // CV_BGR2BGRA    =0
        BGRA2BGR, // CV_BGRA2BGR    =1
        BGR2RGBA, // CV_BGR2RGBA    =2
        RGBA2BGR, // CV_RGBA2BGR    =3
        BGR2RGB,  // CV_BGR2RGB     =4
        0, // CV_BGRA2RGBA   =5

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

        0, // CV_BGR2XYZ     =32
        0, // CV_RGB2XYZ     =33
        0, // CV_XYZ2BGR     =34
        0, // CV_XYZ2RGB     =35

        BGR2YCrCb, // CV_BGR2YCrCb   =36
        RGB2YCrCb, // CV_RGB2YCrCb   =37
        0, // CV_YCrCb2BGR   =38
        0, // CV_YCrCb2RGB   =39

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
        0, // CV_RGB2YUV      = 83
        0, // CV_YUV2BGR      = 84
        0, // CV_YUV2RGB      = 85

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

    dvppFunc_t func = funcs[code];

    if (func == 0)
        CV_Error(Error::StsBadFlag, "Unknown/unsupported color conversion code");

    func(src, dst, dcn);
}

// Instantiate templates to avoid confusion in python code generation
void cvtColordvpp(const InputArray src, OutputArray dst, int code, int dstCn, AscendStream& stream)
{
    cvtColorDo(src, dst, code, dstCn);
}

} // namespace cann
} // namespace cv
