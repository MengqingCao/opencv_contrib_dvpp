// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANN_COMMON_HPP
#define OPENCV_CANN_COMMON_HPP

#include <acl/acl.h>

namespace cv
{
namespace cann
{
static inline void checkAclError(aclError err, const char* file, const int line, const char* func)
{
    if (ACL_SUCCESS != err)
    {
        const char* errMsg = aclGetRecentErrMsg();
        cv::error(cv::Error::AscendApiCallError, errMsg == nullptr ? "" : errMsg, func, file, line);
    }
}

static inline void checkAclPtr(void* ptr, const char* file, const int line, const char* func)
{
    if (nullptr == ptr)
    {
        const char* errMsg = aclGetRecentErrMsg();
        cv::error(cv::Error::AscendApiCallError, errMsg == nullptr ? "" : errMsg, func, file, line);
    }
}

} // namespace cann
} // namespace cv

#define CV_ACL_SAFE_CALL(expr) cv::cann::checkAclError((expr), __FILE__, __LINE__, CV_Func)
#define CV_ACL_SAFE_CALL_PTR(expr)                               \
    ({                                                           \
        auto ptr = (expr);                                       \
        cv::cann::checkAclPtr(ptr, __FILE__, __LINE__, CV_Func); \
        ptr;                                                     \
    })

#endif // OPENCV_CANN_COMMON_HPP
