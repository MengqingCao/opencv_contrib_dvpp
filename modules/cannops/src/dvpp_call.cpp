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
#include <memory>
#include <cstdarg>
#include <string>

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

/*****************************************************************************/

/******************************AscendPicDesc****************************/
AscendPicDesc& AscendPicDesc::setMemAlign()
{
    if (Pic.picture_format == HI_PIXEL_FORMAT_BGR_888 ||
        Pic.picture_format == HI_PIXEL_FORMAT_RGB_888 ||
        Pic.picture_format == HI_PIXEL_FORMAT_YUV_PACKED_444)
    {
        widthAlignment = 16;
        heightAlignment = 1;
        sizeAlignment = 3;
        sizeNum = 3;
    }
    else if (Pic.picture_format == HI_PIXEL_FORMAT_YUV_400)
    {
        widthAlignment = 16;
        heightAlignment = 1;
        sizeAlignment = 1;
        sizeNum = 1;
    }
    else if (Pic.picture_format == HI_PIXEL_FORMAT_ARGB_8888 ||
             Pic.picture_format == HI_PIXEL_FORMAT_ABGR_8888 ||
             Pic.picture_format == HI_PIXEL_FORMAT_RGBA_8888 ||
             Pic.picture_format == HI_PIXEL_FORMAT_BGRA_8888)
    {
        widthAlignment = 16;
        heightAlignment = 1;
        sizeAlignment = 4;
        sizeNum = 4;
    }
    return *this;
}

AscendPicDesc& AscendPicDesc::setPic(hi_pixel_format _picture_format)
{
    // set input
    Pic.picture_format = _picture_format;
    setMemAlign();
    Pic.picture_width_stride = ALIGN_UP(Pic.picture_width, widthAlignment) * sizeAlignment;
    Pic.picture_height_stride = ALIGN_UP(Pic.picture_height, heightAlignment);
    Pic.picture_buffer_size =
        Pic.picture_width_stride * Pic.picture_height_stride * sizeAlignment / sizeNum;
    return *this;
}

std::shared_ptr<hi_void> AscendPicDesc::allocate()
{
    Pic.picture_address = nullptr;
    uint32_t ret = hi_mpi_dvpp_malloc(0, &Pic.picture_address, Pic.picture_buffer_size);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to malloc mem on dvpp");

    return std::shared_ptr<hi_void>(Pic.picture_address, [](void* ptr) { hi_mpi_dvpp_free(ptr); });
}

AscendPicDesc::AscendPicDesc(const AscendMat& ascendMat, hi_pixel_format _picture_format)
{
    Pic.picture_width = ascendMat.cols;
    Pic.picture_height = ascendMat.rows;
    setPic(_picture_format);
    data = allocate();
}

AscendPicDesc::AscendPicDesc(const Mat& mat, hi_pixel_format _picture_format)
{
    Pic.picture_width = mat.cols;
    Pic.picture_height = mat.rows;
    setPic(_picture_format);
    data = allocate();
}

/******************************hi_mpi_vpc warppers****************************/
void vpcCropResizeWarpper(hi_vpc_chn chnId, hi_vpc_pic_info& inPic, hi_vpc_pic_info& outPic,
                          int cnt, uint32_t* taskID, const Rect& rect, Size dsize,
                          int interpolation)
{
    hi_vpc_crop_region cropRegion = {.top_offset = rect.y,
                                     .left_offset = rect.x,
                                     .crop_width = rect.width,
                                     .crop_height = rect.height};

    hi_vpc_resize_info resize_info = {
        .resize_width = dsize.width, .resize_height = dsize.height, .interpolation = interpolation};
    hi_vpc_crop_resize_region crop_resize_info[1];
    crop_resize_info[0].dest_pic_info = outPic;
    crop_resize_info[0].crop_region = cropRegion;
    crop_resize_info[0].resize_info = resize_info;
    uint32_t ret = hi_mpi_vpc_crop_resize(chnId, (const hi_vpc_pic_info*)&inPic, crop_resize_info,
                                          cnt, taskID, -1);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to crop and resize image");
}

void setBatchCropResizeMakeBorder(hi_vpc_chn chnId, std::vector<AscendPicDesc>& inPicDesc,
                                  std::vector<AscendPicDesc>& outPicDesc,
                                  hi_vpc_crop_resize_border_region crop_resize_make_border_info[],
                                  uint32_t* taskID, const Rect& rect, Size dsize, int interpolation,
                                  const int borderType, Scalar scalarV, int top, int left,
                                  int batchNum)
{
    hi_vpc_crop_region cropRegion = {.top_offset = rect.y,
                                     .left_offset = rect.x,
                                     .crop_width = rect.width,
                                     .crop_height = rect.height};

    hi_vpc_resize_info resize_info = {
        .resize_width = dsize.width, .resize_height = dsize.height, .interpolation = interpolation};
    for (size_t i = 0; i < batchNum; i++)
    {
        crop_resize_make_border_info[i].dest_pic_info = outPicDesc[i].Pic;
        crop_resize_make_border_info[i].crop_region = cropRegion;
        crop_resize_make_border_info[i].resize_info = resize_info;
        crop_resize_make_border_info[i].dest_top_offset = top;
        crop_resize_make_border_info[i].dest_left_offset = left;
        crop_resize_make_border_info[i].border_type = static_cast<hi_vpc_bord_type>(borderType);
        crop_resize_make_border_info[i].scalar_value.val[0] = scalarV[2];
        crop_resize_make_border_info[i].scalar_value.val[1] = scalarV[1];
        crop_resize_make_border_info[i].scalar_value.val[2] = scalarV[0];
        crop_resize_make_border_info[i].scalar_value.val[3] = scalarV[3];
    }
}

void vpcBatchCropResizeMakeBorderWarpper(hi_vpc_chn chnId, std::vector<AscendPicDesc>& inPicDesc,
                                         std::vector<AscendPicDesc>& outPicDesc, int cnt[],
                                         uint32_t* taskID, const Rect& rect, Size dsize,
                                         int interpolation, const int borderType, Scalar scalarV,
                                         int top, int left, int batchNum)
{
    hi_vpc_crop_region cropRegion = {.top_offset = rect.y,
                                     .left_offset = rect.x,
                                     .crop_width = rect.width,
                                     .crop_height = rect.height};

    hi_vpc_resize_info resize_info = {
        .resize_width = dsize.width, .resize_height = dsize.height, .interpolation = interpolation};
    hi_vpc_crop_resize_border_region crop_resize_make_border_info[batchNum];
    hi_vpc_pic_info* batchIn[batchNum];
    for (size_t i = 0; i < batchNum; i++)
    {
        cnt[i] = 1;
        batchIn[i] = &(inPicDesc[i].Pic);
    }
    setBatchCropResizeMakeBorder(chnId, inPicDesc, outPicDesc, crop_resize_make_border_info, taskID,
                                 rect, dsize, interpolation, borderType, scalarV, top, left,
                                 batchNum);

    uint32_t ret = hi_mpi_vpc_batch_crop_resize_make_border(chnId, (const hi_vpc_pic_info**)batchIn,
                                                            batchNum, crop_resize_make_border_info,
                                                            (hi_u32*)cnt, taskID, -1);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to crop image");
}

void vpcCropResizeMakeBorderWarpper(hi_vpc_chn chnId, std::vector<AscendPicDesc>& inPicDesc,
                                    std::vector<AscendPicDesc>& outPicDesc, int cnt,
                                    uint32_t* taskID, const Rect& rect, Size dsize,
                                    int interpolation, const int borderType, Scalar scalarV,
                                    int top, int left)
{
    hi_vpc_crop_region cropRegion = {.top_offset = rect.y,
                                     .left_offset = rect.x,
                                     .crop_width = rect.width,
                                     .crop_height = rect.height};

    hi_vpc_resize_info resize_info = {
        .resize_width = dsize.width, .resize_height = dsize.height, .interpolation = interpolation};
    hi_vpc_crop_resize_border_region crop_resize_make_border_info[1];

    setBatchCropResizeMakeBorder(chnId, inPicDesc, outPicDesc, crop_resize_make_border_info, taskID,
                                 rect, dsize, interpolation, borderType, scalarV, top, left, 1);
    uint32_t ret =
        hi_mpi_vpc_crop_resize_make_border(chnId, (const hi_vpc_pic_info*)&inPicDesc[0].Pic,
                                           crop_resize_make_border_info, cnt, taskID, -1);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to crop, resize and make border of image");
}

/******************************DvppOperatorRunner****************************/
DvppOperatorRunner& DvppOperatorRunner::reset()
{
    uint32_t ret = hi_mpi_vpc_destroy_chn(chnId);
    inputDesc_.clear();
    outputDesc_.clear();
    holder.clear();
    return *this;
}
void initDvpp() { hi_mpi_sys_init(); }

void finalizeDvpp() { hi_mpi_sys_exit(); }

DvppOperatorRunner& DvppOperatorRunner::createChannel()
{
    uint32_t ret = hi_mpi_vpc_sys_create_chn(&chnId, &stChnAttr);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::addInput(AscendPicDesc& picDesc)
{
    inputDesc_.push_back(picDesc);
    holder.insert(picDesc.data);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::addInput(const AscendMat& mat,
                                                 hi_pixel_format _picture_format)
{
    AscendPicDesc picDesc(mat, _picture_format);
    aclrtMemcpy2d(picDesc.Pic.picture_address, mat.step, mat.data.get(), mat.step,
                  picDesc.Pic.picture_width_stride, picDesc.Pic.picture_height_stride,
                  ACL_MEMCPY_DEVICE_TO_DEVICE);
    // Mat matHost;
    // matHost.create(mat.cols, mat.rows, mat.type());
    // printDVPPdata(matHost, picDesc.Pic);
    return addInput(picDesc);
}

DvppOperatorRunner& DvppOperatorRunner::addInput(const Mat& mat, hi_pixel_format _picture_format)
{
    AscendPicDesc picDesc(mat, _picture_format);
    const uint32_t esz = CV_ELEM_SIZE(mat.type());
    size_t step = esz * mat.cols;
    aclrtMemcpy2d(picDesc.Pic.picture_address, step, mat.data, mat.step[0],
                  picDesc.Pic.picture_width_stride, picDesc.Pic.picture_height_stride,
                  ACL_MEMCPY_HOST_TO_DEVICE);
    return addInput(picDesc);
}

DvppOperatorRunner& DvppOperatorRunner::addBatchInput(const std::vector<cv::Mat>& mats,
                                                      hi_pixel_format _picture_format, int batchNum)
{
    for (int i = 0; i < batchNum; i++)
    {
        AscendPicDesc picDesc(mats[i], _picture_format);
        const uint32_t esz = CV_ELEM_SIZE(mats[i].type());
        size_t step = esz * mats[i].cols;
        aclrtMemcpy2d(picDesc.Pic.picture_address, step, mats[i].data, mats[i].step[0],
                      picDesc.Pic.picture_width_stride, picDesc.Pic.picture_height_stride,
                      ACL_MEMCPY_HOST_TO_DEVICE);
        addInput(picDesc);
    }
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::addOutput(AscendPicDesc& picDesc)
{
    outputDesc_.push_back(picDesc);
    holder.insert(picDesc.data);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::addOutput(AscendMat& mat, hi_pixel_format _picture_format)
{
    AscendPicDesc picDesc(mat, _picture_format);
    return addOutput(picDesc);
}

DvppOperatorRunner& DvppOperatorRunner::addOutput(Mat& mat, hi_pixel_format _picture_format)
{
    AscendPicDesc picDesc(mat, _picture_format);
    return addOutput(picDesc);
}

DvppOperatorRunner& DvppOperatorRunner::addBatchOutput(const std::vector<cv::Mat>& mats,
                                                       hi_pixel_format _picture_format,
                                                       int batchNum)
{
    for (int i = 0; i < batchNum; i++)
    {
        AscendPicDesc picDesc(mats[i], _picture_format);
        const uint32_t esz = CV_ELEM_SIZE(mats[i].type());
        size_t step = esz * mats[i].cols;
        aclrtMemcpy2d(picDesc.Pic.picture_address, step, mats[i].data, mats[i].step[0],
                      picDesc.Pic.picture_width_stride, picDesc.Pic.picture_height_stride,
                      ACL_MEMCPY_HOST_TO_DEVICE);
        addOutput(picDesc);
    }
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::getResult(Mat& dst, uint32_t& taskIDResult)
{
    uint32_t ret = hi_mpi_vpc_get_process_result(chnId, taskIDResult, -1);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to get process result.");
    const uint32_t esz = CV_ELEM_SIZE(dst.type());
    size_t step = esz * dst.cols;

    aclrtMemcpy2d(dst.data, dst.step[0], outputDesc_[0].Pic.picture_address, step,
                  outputDesc_[0].Pic.picture_width_stride, outputDesc_[0].Pic.picture_height_stride,
                  ACL_MEMCPY_DEVICE_TO_HOST);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::getResult(AscendMat& dst, uint32_t& taskIDResult)
{
    uint32_t ret = hi_mpi_vpc_get_process_result(chnId, taskIDResult, -1);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to get process result.");
    // const uint32_t esz = CV_ELEM_SIZE(dst.type());
    // size_t step = esz * dst.cols;
    // aclrtMemcpy2d(dst.data.get(), dst.step, outputDesc_[0].data.get(), step,
    //               outputDesc_[0].Pic.picture_width_stride,
    //               outputDesc_[0].Pic.picture_height_stride, ACL_MEMCPY_DEVICE_TO_DEVICE);
    uint32_t size = dst.rows * dst.cols * dst.elemSize();
    aclrtMemcpy(dst.data.get(), size, outputDesc_[0].Pic.picture_address, size,
                ACL_MEMCPY_DEVICE_TO_DEVICE);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::getResult(std::vector<cv::Mat>& dst, uint32_t& taskIDResult,
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
        aclrtMemcpy2d(dst[i].data, dst[i].step[0], outputDesc_[i].Pic.picture_address, step,
                      outputDesc_[i].Pic.picture_width_stride,
                      outputDesc_[i].Pic.picture_height_stride, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    return *this;
}

} // namespace cann
} // namespace cv

#endif // ENABLE_DVPP_INTERFACE