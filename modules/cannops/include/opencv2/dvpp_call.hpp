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

    DvppOperatorRunner& reset();
    DvppOperatorRunner& createChannel();

    uint32_t widthAlignment;
    uint32_t heightAlignment;
    uint32_t sizeAlignment;
    uint32_t sizeNum;

    DvppOperatorRunner& Init()
    {
        hi_mpi_sys_init();
        inputPic.picture_address = nullptr;
        outputPic.picture_address = nullptr;
    }
    DvppOperatorRunner& setPic(int32_t* size, hi_vpc_pic_info* Pic);
    uint32_t AlignmentHelper(uint32_t origSize, uint32_t alignment);
    hi_vpc_pic_info inputPic;
    hi_vpc_pic_info outputPic;
    hi_vpc_chn chnId;
    hi_vpc_chn_attr stChnAttr;
};

} // namespace cann
} // namespace cv
