// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANNCALL_HPP
#define OPENCV_CANNCALL_HPP

#include <vector>
#include <acl/acl.h>
#include "opencv2/cann.hpp"

namespace cv
{
namespace cann
{
struct AclAttribute
{
    virtual ~AclAttribute() = default;
    virtual void addAttr(aclopAttr* opAttr) = 0;
};

#define DEFINE_ATTR(FUNC, TYPE)                                                              \
    class Acl##FUNC##Attribute : public AclAttribute                                         \
    {                                                                                        \
        const char* name;                                                                    \
        TYPE value;                                                                          \
                                                                                             \
    public:                                                                                  \
        Acl##FUNC##Attribute(const char* _name, TYPE _value) : name(_name), value(_value){}; \
        void addAttr(aclopAttr* opAttr) override                                             \
        {                                                                                    \
            CV_ACL_SAFE_CALL(aclopSetAttr##FUNC(opAttr, name, value));                       \
        }                                                                                    \
    }

DEFINE_ATTR(Float, float);
DEFINE_ATTR(String, const char*);

static std::vector<AclAttribute*> emptyattr;
void aclOneInput(const AclMat& src, AclMat& dst, const char* op,
                 AclStream& stream = AclStream::Null(),
                 std::vector<AclAttribute*>& attrs = emptyattr);

void aclTwoInputs(const AclMat& src1, const AclMat& src2, AclMat& dst, const char* op,
                  AclStream& stream = AclStream::Null());

void transNCHWToNHWC(const AclMat& src, AclMat& dst, AclStream& stream = AclStream::Null());

} // namespace cann
} // namespace cv

#endif // OPENCV_CANNCALL_HPP
