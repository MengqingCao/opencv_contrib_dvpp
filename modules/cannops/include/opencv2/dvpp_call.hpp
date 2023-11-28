// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef ENABLE_DVPP_INTERFACE
    #define ENABLE_DVPP_INTERFACE
#endif // ENABLE_DVPP_INTERFACE

#include <vector>
#include <string>
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <acl/dvpp/hi_dvpp.h>
#include "acl/acl_op.h"
#include "cann_call.hpp"

namespace cv
{
namespace cann
{
void acldvppMallocWarpper(void** data, size_t size);
void acldvppFreeWarpper(void* data);

typedef struct PicDesc
{
    std::string picName;
    int width;
    int height;
} PicDesc;

typedef struct CropPicDesc
{
    std::string picName;
    int left;
    int top;
    int width;
    int height;
} CropPicDesc;

class DvppOperatorRunner
{
private:
    DvppOperatorRunner& addInput(AscendTensor& tensor);
    DvppOperatorRunner& addOutput(AscendTensor& tensor);

public:
    DvppOperatorRunner() {}
    virtual ~DvppOperatorRunner() { reset(); }
    DvppOperatorRunner& addInput(AscendMat& mat);
    DvppOperatorRunner& addOutput(AscendMat& mat);
    DvppOperatorRunner& addInput(Mat& mat);
    DvppOperatorRunner& addOutput(Mat& mat);
    DvppOperatorRunner& getResult(Mat& dst, uint32_t& taskIDResult);
    DvppOperatorRunner& getResult(std::vector<cv::Mat>& dst, uint32_t& taskIDResult,
                                  hi_vpc_crop_resize_border_region* crop_resize_make_border_info,
                                  int batchNum);
    DvppOperatorRunner& getResult(AscendMat& dst, uint32_t& taskIDResult);
    DvppOperatorRunner& addBatchInput(std::vector<cv::Mat>& _src, int batchNum,
                                      hi_pixel_format pixelFormat);
    DvppOperatorRunner& addBatchOutput(std::vector<cv::Mat>& _src, int batchNum,
                                       hi_pixel_format pixelFormat);
    DvppOperatorRunner& setMemAlign(hi_vpc_pic_info* Pic);

    DvppOperatorRunner& reset();
    DvppOperatorRunner& createChannel();

    uint32_t widthAlignment = 1;
    uint32_t heightAlignment = 1;
    uint32_t sizeAlignment = 1;
    uint32_t sizeNum = 1;

    DvppOperatorRunner& Init()
    {
        inputPic.picture_address = nullptr;
        outputPic.picture_address = nullptr;
        return *this;
    }
    DvppOperatorRunner& setPic(int32_t* size, hi_vpc_pic_info* Pic);
    uint32_t AlignmentHelper(uint32_t origSize, uint32_t alignment);
    hi_vpc_pic_info inputPic;
    hi_vpc_pic_info outputPic;
    hi_vpc_pic_info* batchInPic;
    hi_vpc_pic_info* batchOutPic;
    hi_vpc_chn chnId;
    hi_vpc_chn_attr stChnAttr;
    // top;
    // left;
};

} // namespace cann
} // namespace cv
