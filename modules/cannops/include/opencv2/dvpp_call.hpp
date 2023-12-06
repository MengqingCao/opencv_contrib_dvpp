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
struct AscendPicDesc
{
    const char* name;
    std::shared_ptr<hi_void> data;
    std::vector<int64_t> batchNum;

    size_t widthAlignment = 16;
    size_t heightAlignment = 1;
    size_t sizeAlignment = 3;
    size_t sizeNum = 3;

    hi_vpc_pic_info Pic;
    AscendPicDesc& setMemAlign();
    AscendPicDesc& setPic(hi_pixel_format _picture_format);
    std::shared_ptr<hi_void> allocate();
    AscendPicDesc(){};
    AscendPicDesc(const AscendMat& ascendMat, hi_pixel_format _picture_format);
    AscendPicDesc(const Mat& mat, hi_pixel_format _picture_format);
};

/******************************hi_mpi_vpc warppers****************************/
inline void vpcResizeWarpper(hi_vpc_chn chnId, hi_vpc_pic_info& inPic, hi_vpc_pic_info& outPic,
                             int interpolation, uint32_t* taskID)
{
    uint32_t ret = hi_mpi_vpc_resize(chnId, &inPic, &outPic, 0, 0, interpolation, taskID, -1);
    if (ret != HI_SUCCESS)
        CV_Error(Error::StsBadFlag, "failed to resize image");
}
void vpcCropResizeWarpper(hi_vpc_chn chnId, hi_vpc_pic_info& inPic, hi_vpc_pic_info& outPic,
                          int cnt, uint32_t* taskID, const Rect& rect, Size dsize,
                          int interpolation);

void vpcCropResizeMakeBorderWarpper(hi_vpc_chn chnId, std::vector<AscendPicDesc>& inPicDesc,
                                    std::vector<AscendPicDesc>& outPicDesc, int cnt,
                                    uint32_t* taskID, const Rect& rect, Size dsize,
                                    int interpolation, const int borderType, Scalar scalarV,
                                    int top, int left);
void vpcBatchCropResizeMakeBorderWarpper(hi_vpc_chn chnId, std::vector<AscendPicDesc>& inPicDesc,
                                         std::vector<AscendPicDesc>& outPicDesc, int cnt[],
                                         uint32_t* taskID, const Rect& rect, Size dsize,
                                         int interpolation, const int borderType, Scalar scalarV,
                                         int top, int left, int batchNum);
void vpcCopyMakeBorderWarpper(hi_vpc_chn chnId, hi_vpc_pic_info& inPic, hi_vpc_pic_info& outPic,
                              uint32_t* taskID, int* offsets, int bordertype, Scalar value);
/*****************************************************************************/

class DvppOperatorRunner
{
private:
    DvppOperatorRunner& addInput(AscendPicDesc& picDesc);
    DvppOperatorRunner& addOutput(AscendPicDesc& picDesc);
    std::set<std::shared_ptr<hi_void>> holder;

public:
    DvppOperatorRunner() {}
    virtual ~DvppOperatorRunner() { reset(); }
    DvppOperatorRunner& addInput(const AscendMat& mat);
    DvppOperatorRunner& addOutput(AscendMat& mat);
    DvppOperatorRunner& addInput(const Mat& mat);
    DvppOperatorRunner& addBatchInput(const std::vector<cv::Mat>& mats, int batchNum);
    DvppOperatorRunner& addBatchInput(const std::vector<AscendMat>& mats, int batchNum);
    DvppOperatorRunner& addOutput(Mat& mat);
    DvppOperatorRunner& addBatchOutput(const std::vector<cv::Mat>& mats, int batchNum);
    DvppOperatorRunner& addBatchOutput(const std::vector<AscendMat>& mats, int batchNum);

    DvppOperatorRunner& getResult(Mat& dst, uint32_t& taskIDResult);
    DvppOperatorRunner& getResult(std::vector<cv::Mat>& dst, uint32_t& taskIDResult, int batchNum);
    DvppOperatorRunner& getResult(std::vector<AscendMat>& dst, uint32_t& taskIDResult,
                                  int batchNum);

    DvppOperatorRunner& getResult(AscendMat& dst, uint32_t& taskIDResult);

    DvppOperatorRunner& reset();
    DvppOperatorRunner& createChannel();

    std::vector<AscendPicDesc> inputDesc_;
    std::vector<AscendPicDesc> outputDesc_;

    hi_vpc_chn chnId;
    hi_vpc_chn_attr stChnAttr;
    DvppOperatorRunner& Init()
    {
        chnId = 0;
        stChnAttr = {};
        return *this;
    }
};

} // namespace cann
} // namespace cv
