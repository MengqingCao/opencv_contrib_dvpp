#ifndef ENABLE_DVPP_INTERFACE
#define ENABLE_DVPP_INTERFACE

#include <acl/acl.h>
#include <acl/dvpp/hi_dvpp.h>
#include <acl/dvpp/hi_media_common.h>
#include "opencv2/core/private.hpp"
#include "opencv2/dvpp_call.hpp"
#include "opencv2/cann.hpp"
#include "opencv2/stream_accessor.hpp"
#include <iostream>

#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr) __builtin_expect(!!(expr), 1)

namespace cv
{
namespace cann
{
static inline void checkAclError(aclError err, const char* file, const int line, const char* func)
{
    if (ACL_SUCCESS != err)
    {
        const char* errMsg = aclGetRecentErrMsg();
        cv::error(cv::Error::StsError, errMsg == nullptr ? "" : errMsg, func, file, line);
    }
}

static inline void checkAclPtr(void* ptr, const char* file, const int line, const char* func)
{
    if (nullptr == ptr)
    {
        const char* errMsg = aclGetRecentErrMsg();
        cv::error(cv::Error::StsError, errMsg == nullptr ? "" : errMsg, func, file, line);
    }
}

#define CV_ACL_SAFE_CALL(expr) checkAclError((expr), __FILE__, __LINE__, CV_Func)
#define CV_ACL_SAFE_CALL_PTR(expr)                     \
    ({                                                 \
        auto ptr = (expr);                             \
        checkAclPtr(ptr, __FILE__, __LINE__, CV_Func); \
        ptr;                                           \
    })

void printDVPPdata(const Mat mat, const hi_vpc_pic_info inputPic)
{
    Mat dst;
    dst.create(mat.size(), mat.type());
    aclrtMemcpy2d(dst.data, mat.step, inputPic.picture_address, mat.step, mat.cols * mat.elemSize(),
                  mat.rows, ACL_MEMCPY_DEVICE_TO_HOST);
    std::cout << dst << std::endl;
}
/******************************Acl Runtime Warpper****************************/
void acldvppMallocWarpper(void** data, size_t size)
{
    CV_ACL_SAFE_CALL(hi_mpi_dvpp_malloc(0, data, size));
}

void acldvppFreeWarpper(void* data) { CV_ACL_SAFE_CALL(hi_mpi_dvpp_free(data)); }

uint32_t DvppOperatorRunner::AlignmentHelper(uint32_t origSize, uint32_t alignment)
{
    if (alignment == 0)
    {
        return 0;
    }
    uint32_t alignmentH = alignment - 1;
    return (origSize + alignmentH) / alignment * alignment;
}
DvppOperatorRunner& DvppOperatorRunner::reset()
{
    uint32_t ret = hi_mpi_vpc_destroy_chn(chnId);
    if (inputPic.picture_address != nullptr)
    {
        hi_mpi_dvpp_free(inputPic.picture_address);
        inputPic.picture_address = nullptr;
    }
    if (outputPic.picture_address != nullptr)
    {
        hi_mpi_dvpp_free(outputPic.picture_address);
        outputPic.picture_address = nullptr;
    }
    return *this;
}
void initDvpp() { hi_mpi_sys_init(); }

void finalizeDvpp() { hi_mpi_sys_exit(); }

// void setBatchInOut(std::vector<cv::Mat>& _src, std::vector<cv::Mat>& _dst, const Rect& rect,
//                    Size dsize, double inv_scale_x, double inv_scale_y, int interpolation,
//                    const int borderType, double* scalarV, int top, int left, int batchNum)
// {
//     // crop info
//     uint32_t x = rect.x, y = rect.y, width = rect.width, height = rect.height;

//     DvppOperatorRunner op;
//     hi_vpc_pic_info inputPic;
//     inputPic.picture_address = nullptr;
//     hi_vpc_pic_info outputPic;
//     outputPic.picture_address = nullptr;

//     uint32_t taskID = 0;
//     int32_t sizeIn[] = {_src[0].rows, _src[0].cols};
//     int32_t dstSize[] = {_dst[0].rows, _dst[0].cols};

//     // set input and output
//     inputPic.picture_format = HI_PIXEL_FORMAT_BGR_888;
//     outputPic.picture_format = HI_PIXEL_FORMAT_BGR_888;
//     uint32_t widthAlignment = 16;
//     uint32_t heightAlignment = 1;
//     uint32_t sizeAlignment = 3;
//     uint32_t sizeNum = 3;

//     // set input
//     inputPic.picture_width = sizeIn[0];
//     inputPic.picture_height = sizeIn[1];
//     inputPic.picture_format = inputPic.picture_format;
//     inputPic.picture_width_stride = ALIGN_UP(sizeIn[0], widthAlignment) * sizeAlignment;
//     inputPic.picture_height_stride = ALIGN_UP(sizeIn[1], heightAlignment);

//     outputPic.picture_width = dstSize[0];
//     outputPic.picture_height = dstSize[1];
//     outputPic.picture_format = outputPic.picture_format;
//     outputPic.picture_width_stride = ALIGN_UP(dstSize[0], widthAlignment) * sizeAlignment;
//     outputPic.picture_height_stride = ALIGN_UP(dstSize[1], heightAlignment);

//     hi_vpc_pic_info* batchInput[batchNum];
//     hi_vpc_crop_region cropRegion = {
//         .top_offset = y, .left_offset = x, .crop_width = width, .crop_height = height};
//     hi_vpc_crop_region_info cropInfo = {.dest_pic_info = op.outputPic, .crop_region = cropRegion};
//     hi_vpc_crop_region_info cropInfos[] = {cropInfo};

//     uint32_t cntCrop = 1;
//     hi_u32 cnt[batchNum];

//     hi_vpc_resize_info resize_info = {
//         .resize_width = dsize.width, .resize_height = dsize.height, .interpolation = interpolation};
//     hi_vpc_crop_resize_border_region crop_resize_make_border_info[batchNum];

//     for (size_t i = 0; i < batchNum; i++)
//     {
//         inputPic.picture_buffer_size = inputPic.picture_width_stride *
//                                        inputPic.picture_height_stride * sizeAlignment / sizeNum;
//         const uint32_t esz = CV_ELEM_SIZE(_src[i].type());
//         size_t step = esz * _src[i].cols;
//         uint32_t ret =
//             hi_mpi_dvpp_malloc(0, &inputPic.picture_address, inputPic.picture_buffer_size);
//         if (ret != HI_SUCCESS)
//             CV_Error(Error::StsBadFlag, "failed to malloc mem for input data");
//         aclrtMemcpy2d(inputPic.picture_address, step, _src[i].data, _src[i].step[0],
//                       inputPic.picture_width_stride, inputPic.picture_height_stride,
//                       ACL_MEMCPY_HOST_TO_DEVICE);
//         batchInput[i] = &inputPic;

//         outputPic.picture_buffer_size = outputPic.picture_width_stride *
//                                        outputPic.picture_height_stride * sizeAlignment / sizeNum;
//         uint32_t ret =
//             hi_mpi_dvpp_malloc(0, &outputPic.picture_address, outputPic.picture_buffer_size);
//         if (ret != HI_SUCCESS)
//             CV_Error(Error::StsBadFlag, "failed to malloc mem for input data");
//         crop_resize_make_border_info[i].dest_pic_info = outputPic;

//         crop_resize_make_border_info[i].crop_region = cropRegion;
//         crop_resize_make_border_info[i].resize_info = resize_info;
//         crop_resize_make_border_info[i].dest_top_offset = top;
//         crop_resize_make_border_info[i].dest_left_offset = left;
//         crop_resize_make_border_info[i].border_type = static_cast<hi_vpc_bord_type>(borderType);
//         // memcpy(crop_resize_make_border_info[i].scalar_value.val, scalarV, sizeof(scalarV));
//         crop_resize_make_border_info[i].scalar_value.val[0] = 1;
//         crop_resize_make_border_info[i].scalar_value.val[1] = 1;
//         crop_resize_make_border_info[i].scalar_value.val[2] = 1;
//         crop_resize_make_border_info[i].scalar_value.val[3] = 1;

//         cnt[i] = 1;
//     }
// }

DvppOperatorRunner& DvppOperatorRunner::createChannel()
{
    uint32_t ret = hi_mpi_vpc_sys_create_chn(&chnId, &stChnAttr);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::setPic(int32_t* size, hi_vpc_pic_info* Pic)
{
    // set input
    Pic->picture_width = size[0];
    Pic->picture_height = size[1];
    Pic->picture_format = Pic->picture_format;
    Pic->picture_width_stride = ALIGN_UP(size[0], widthAlignment) * sizeAlignment;
    Pic->picture_height_stride = ALIGN_UP(size[1], heightAlignment);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::setMemAlign(hi_vpc_pic_info* Pic)
{
    if (Pic->picture_format == HI_PIXEL_FORMAT_BGR_888 ||
        Pic->picture_format == HI_PIXEL_FORMAT_RGB_888 ||
        Pic->picture_format == HI_PIXEL_FORMAT_YUV_PACKED_444)
    {
        widthAlignment = 16;
        heightAlignment = 1;
        sizeAlignment = 3;
        sizeNum = 3;
    }
    else if (Pic->picture_format == HI_PIXEL_FORMAT_YUV_400)
    {
        widthAlignment = 16;
        heightAlignment = 1;
        sizeAlignment = 1;
        sizeNum = 1;
    }
    else if (Pic->picture_format == HI_PIXEL_FORMAT_ARGB_8888 ||
             Pic->picture_format == HI_PIXEL_FORMAT_ABGR_8888 ||
             Pic->picture_format == HI_PIXEL_FORMAT_RGBA_8888 ||
             Pic->picture_format == HI_PIXEL_FORMAT_BGRA_8888)
    {
        widthAlignment = 16;
        heightAlignment = 1;
        sizeAlignment = 4;
        sizeNum = 4;
    }
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::addInput(AscendTensor& tensor)
{
    inputPic.picture_buffer_size =
        inputPic.picture_width_stride * inputPic.picture_height_stride * sizeAlignment / sizeNum;
    hi_mpi_dvpp_malloc(0, &inputPic.picture_address, inputPic.picture_buffer_size);
    aclrtMemcpy(inputPic.picture_address, tensor.dataSize, tensor.data.get(), tensor.dataSize,
                ACL_MEMCPY_DEVICE_TO_DEVICE);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::addInput(AscendMat& mat)
{
    inputPic.picture_buffer_size =
        inputPic.picture_width_stride * inputPic.picture_height_stride * sizeAlignment / sizeNum;

    uint32_t ret = hi_mpi_dvpp_malloc(0, &inputPic.picture_address, inputPic.picture_buffer_size);
    aclrtMemcpy(inputPic.picture_address, inputPic.picture_buffer_size, mat.data.get(),
                inputPic.picture_buffer_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::addInput(Mat& mat)
{
    inputPic.picture_buffer_size =
        inputPic.picture_width_stride * inputPic.picture_height_stride * sizeAlignment / sizeNum;
    const uint32_t esz = CV_ELEM_SIZE(mat.type());
    size_t step = esz * mat.cols;
    uint32_t ret = hi_mpi_dvpp_malloc(0, &inputPic.picture_address, inputPic.picture_buffer_size);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to malloc mem for input data");
    aclrtMemcpy2d(inputPic.picture_address, step, mat.data, mat.step[0],
                  inputPic.picture_width_stride, inputPic.picture_height_stride,
                  ACL_MEMCPY_HOST_TO_DEVICE);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::addOutput(AscendTensor& tensor)
{
    outputPic.picture_buffer_size =
        outputPic.picture_width_stride * outputPic.picture_height_stride * sizeAlignment / sizeNum;
    uint32_t ret = hi_mpi_dvpp_malloc(0, &outputPic.picture_address, outputPic.picture_buffer_size);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::addOutput(AscendMat& mat)
{
    outputPic.picture_buffer_size =
        outputPic.picture_width_stride * outputPic.picture_height_stride * sizeAlignment / sizeNum;
    uint32_t ret = hi_mpi_dvpp_malloc(0, &outputPic.picture_address, outputPic.picture_buffer_size);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::addOutput(Mat& mat)
{
    outputPic.picture_buffer_size =
        outputPic.picture_width_stride * outputPic.picture_height_stride * sizeAlignment / sizeNum;
    uint32_t ret = hi_mpi_dvpp_malloc(0, &outputPic.picture_address, outputPic.picture_buffer_size);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to malloc mem for output data");

    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::getResult(Mat& dst, uint32_t& taskIDResult)
{
    uint32_t ret = hi_mpi_vpc_get_process_result(chnId, taskIDResult, -1);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to get process result.");
    const uint32_t esz = CV_ELEM_SIZE(dst.type());
    size_t step = esz * dst.cols;
    aclrtMemcpy2d(dst.data, dst.step[0], outputPic.picture_address, step,
                  outputPic.picture_width_stride, outputPic.picture_height_stride,
                  ACL_MEMCPY_DEVICE_TO_HOST);
    return *this;
}

DvppOperatorRunner&
DvppOperatorRunner::getResult(std::vector<cv::Mat>& dst, uint32_t& taskIDResult,
                              hi_vpc_crop_resize_border_region* crop_resize_make_border_info,
                              int batchNum)
{
    uint32_t ret = hi_mpi_vpc_get_process_result(chnId, taskIDResult, -1);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to get process result.");
    CV_Assert(batchNum >= 1);
    const uint32_t esz = CV_ELEM_SIZE(dst[0].type());
    size_t step = esz * dst[0].cols;
    for (int i = 0; i < batchNum; i++)
    {
        aclrtMemcpy2d(dst[i].data, dst[i].step[0],
                      crop_resize_make_border_info[i].dest_pic_info.picture_address, step,
                      crop_resize_make_border_info[i].dest_pic_info.picture_width_stride,
                      crop_resize_make_border_info[i].dest_pic_info.picture_height_stride,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    }
    return *this;
}

} // namespace cann
} // namespace cv

#endif // ENABLE_DVPP_INTERFACE