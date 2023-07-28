// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANNPREPARE_HPP
#define OPENCV_CANNPREPARE_HPP

#include <vector>
#include <acl/acl.h>
#include "opencv2/core.hpp"
#include "opencv2/cann_common.hpp"

namespace cv
{
namespace cann
{
struct CannPreparation
{
    CannPreparation() { opAttr_ = CV_ACL_SAFE_CALL_PTR(aclopCreateAttr()); }

    virtual ~CannPreparation()
    {
        for (auto desc : inputDesc_)
        {
            aclDestroyTensorDesc(desc);
        }

        for (auto desc : outputDesc_)
        {
            aclDestroyTensorDesc(desc);
        }

        for (auto buf : inputBuffers_)
        {
            aclDestroyDataBuffer(buf);
        }

        for (auto buf : outputBuffers_)
        {
            aclDestroyDataBuffer(buf);
        }

        aclopDestroyAttr(opAttr_);
    }

    std::vector<aclDataBuffer*> inputBuffers_;
    std::vector<aclDataBuffer*> outputBuffers_;
    std::vector<aclTensorDesc*> inputDesc_;
    std::vector<aclTensorDesc*> outputDesc_;
    aclopAttr* opAttr_;
};

#define CANN_PREPARE_ADD_ATTR(var, type, ...)                           \
    do                                                                  \
    {                                                                   \
        CV_ACL_SAFE_CALL(aclopSetAttr##type(var.opAttr_, __VA_ARGS__)); \
    } while (0)

#define CANN_PREPARE_INPUTDESC(var, ...)                                     \
    do                                                                       \
    {                                                                        \
        auto _rPtr = CV_ACL_SAFE_CALL_PTR(aclCreateTensorDesc(__VA_ARGS__)); \
        if (_rPtr != nullptr)                                                \
            var.inputDesc_.push_back(_rPtr);                                 \
    } while (0)

#define CANN_PREPARE_OUTPUTDESC(var, ...)                                    \
    do                                                                       \
    {                                                                        \
        auto _rPtr = CV_ACL_SAFE_CALL_PTR(aclCreateTensorDesc(__VA_ARGS__)); \
        if (_rPtr != nullptr)                                                \
            var.outputDesc_.push_back(_rPtr);                                \
    } while (0)

#define CANN_PREPARE_INPUTBUFFER(var, ...)                                   \
    do                                                                       \
    {                                                                        \
        auto _rPtr = CV_ACL_SAFE_CALL_PTR(aclCreateDataBuffer(__VA_ARGS__)); \
        if (_rPtr != nullptr)                                                \
            var.inputBuffers_.push_back(_rPtr);                              \
    } while (0)

#define CANN_PREPARE_OUTPUTBUFFER(var, ...)                                  \
    do                                                                       \
    {                                                                        \
        auto _rPtr = CV_ACL_SAFE_CALL_PTR(aclCreateDataBuffer(__VA_ARGS__)); \
        if (_rPtr != nullptr)                                                \
            var.outputBuffers_.push_back(_rPtr);                             \
    } while (0)

aclDataType getACLType(int opencvdepth);

} // namespace cann
} // namespace cv

#endif // OPENCV_CANNPREPARE_HPP
