// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANN_STREAM_ACCESSOR_HPP
#define OPENCV_CANN_STREAM_ACCESSOR_HPP

#include <acl/acl.h>
#include "opencv2/cann.hpp"

namespace cv
{
namespace cann
{

//! @addtogroup cann_struct
//! @{

/** @brief Class that enables getting aclrtAclStream from cann::AclStream
 */
struct AclStreamAccessor
{
    CV_EXPORTS static aclrtStream getStream(const AclStream& stream);
    CV_EXPORTS static AclStream wrapStream(aclrtStream stream);
};

/** @brief Class that enables getting aclrtAclEvent from cann::AclEvent
 */
struct AclEventAccessor
{
    CV_EXPORTS static aclrtEvent getEvent(const AclEvent& event);
    CV_EXPORTS static AclEvent wrapEvent(aclrtEvent event);
};

//! @} cann_struct

} // namespace cann
} // namespace cv

#endif // OPENCV_CANN_STREAM_ACCESSOR_HPP
