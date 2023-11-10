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
    hi_mpi_vpc_destroy_chn(chnId);
    hi_mpi_dvpp_free(inputPic.picture_address);
    hi_mpi_dvpp_free(outputPic.picture_address);
    hi_mpi_sys_exit();
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::createChannel()
{
    uint32_t ret = hi_mpi_vpc_sys_create_chn(&chnId, &stChnAttr);
    // std::cout << "hi_mpi_vpc_sys_create_chn " << ret << std::endl;

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
}

DvppOperatorRunner& DvppOperatorRunner::addInput(AscendTensor& tensor)
{
    inputPic.picture_buffer_size =
        inputPic.picture_width_stride * inputPic.picture_height_stride * sizeAlignment / sizeNum;
    // inputPic.picture_buffer_size = tensor.dataSize;
    hi_mpi_dvpp_malloc(0, &inputPic.picture_address, inputPic.picture_buffer_size);
    aclrtMemcpy(inputPic.picture_address, tensor.dataSize, tensor.data.get(), tensor.dataSize,
                ACL_MEMCPY_DEVICE_TO_DEVICE);

    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::addInput(AscendMat& mat)
{
    uint32_t size = mat.rows * mat.cols * mat.elemSize();
    inputPic.picture_buffer_size =
        inputPic.picture_width_stride * inputPic.picture_height_stride * sizeAlignment / sizeNum;

    uint32_t ret = hi_mpi_dvpp_malloc(0, &inputPic.picture_address, inputPic.picture_buffer_size);
    aclrtMemcpy(inputPic.picture_address, inputPic.picture_buffer_size, mat.data.get(),
                inputPic.picture_buffer_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    return *this;
}
DvppOperatorRunner& DvppOperatorRunner::addInput(Mat& mat)
{
    uint32_t size = mat.rows * mat.cols * mat.elemSize();
    inputPic.picture_buffer_size =
        inputPic.picture_width_stride * inputPic.picture_height_stride * sizeAlignment / sizeNum;

    uint32_t ret = hi_mpi_dvpp_malloc(0, &inputPic.picture_address, inputPic.picture_buffer_size);
    aclrtMemcpy(inputPic.picture_address, inputPic.picture_buffer_size, mat.data,
                inputPic.picture_buffer_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::addOutput(AscendTensor& tensor)
{
    outputPic.picture_address = tensor.data.get();
    outputPic.picture_buffer_size =
        outputPic.picture_width_stride * outputPic.picture_height_stride * sizeAlignment / sizeNum;
    uint32_t ret = hi_mpi_dvpp_malloc(0, &outputPic.picture_address, outputPic.picture_buffer_size);

    aclrtMemset(outputPic.picture_address, outputPic.picture_buffer_size, 0,
                outputPic.picture_buffer_size);

    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::addOutput(AscendMat& mat)
{
    outputPic.picture_address = mat.data.get();
    outputPic.picture_buffer_size = mat.rows * mat.cols * mat.elemSize();
    uint32_t ret = hi_mpi_dvpp_malloc(0, &outputPic.picture_address, outputPic.picture_buffer_size);

    aclrtMemset(outputPic.picture_address, outputPic.picture_buffer_size, 0,
                outputPic.picture_buffer_size);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::addOutput(Mat& mat)
{
    outputPic.picture_address = mat.data;
    outputPic.picture_buffer_size = mat.rows * mat.cols * mat.elemSize();
    uint32_t ret = hi_mpi_dvpp_malloc(0, &outputPic.picture_address, outputPic.picture_buffer_size);

    aclrtMemset(outputPic.picture_address, outputPic.picture_buffer_size, 0,
                outputPic.picture_buffer_size);
    return *this;
}

DvppOperatorRunner& DvppOperatorRunner::getResult(Mat& dst, uint32_t& taskIDResult)
{
    hi_mpi_vpc_get_process_result(chnId, taskIDResult, -1);
    aclrtMemcpy(dst.data, outputPic.picture_buffer_size, outputPic.picture_address,
                outputPic.picture_buffer_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    return *this;
}

} // namespace cann
} // namespace cv

#endif // ENABLE_DVPP_INTERFACE