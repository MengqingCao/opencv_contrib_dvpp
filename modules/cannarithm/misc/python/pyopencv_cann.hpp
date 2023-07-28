// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifdef HAVE_OPENCV_CORE

#include "opencv2/cann.hpp"

typedef std::vector<cann::AclMat> vector_AclMat;
typedef cann::AclMat::Allocator AclMat_Allocator;

CV_PY_TO_CLASS(cann::AclMat);
CV_PY_TO_CLASS(cann::AclStream);

CV_PY_TO_CLASS_PTR(cann::AclMat);
CV_PY_TO_CLASS_PTR(cann::AclMat::Allocator);

CV_PY_FROM_CLASS(cann::AclMat);
CV_PY_FROM_CLASS(cann::AclStream);

CV_PY_FROM_CLASS_PTR(cann::AclMat::Allocator);

#endif
