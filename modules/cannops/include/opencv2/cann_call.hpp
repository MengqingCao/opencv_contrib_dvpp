// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANNOPS_CANN_CALL_HPP
#define OPENCV_CANNOPS_CANN_CALL_HPP

#include <vector>
#include <set>
#include <string>
#include <acl/acl_base.h>
#include "opencv2/cann.hpp"

class aclopAttr;

namespace cv
{
namespace cann
{
struct AclAttribute
{
    virtual ~AclAttribute() = default;
    virtual void addAttr(aclopAttr* opAttr) = 0;
};

#define DEFINE_ATTR_DECLEAR(FUNC, TYPE)                                                      \
    class Acl##FUNC##Attribute : public AclAttribute                                         \
    {                                                                                        \
        const char* name;                                                                    \
        TYPE value;                                                                          \
                                                                                             \
    public:                                                                                  \
        Acl##FUNC##Attribute(const char* _name, TYPE _value) : name(_name), value(_value){}; \
        void addAttr(aclopAttr* opAttr) CV_OVERRIDE;                                         \
    }

#define DEFINE_ATTR_LIST_DECLEAR(FUNC, TYPE)                               \
    class AclList##FUNC##Attribute : public AclAttribute                   \
    {                                                                      \
        const char* name;                                                  \
        TYPE value;                                                        \
        int num;                                                           \
                                                                           \
    public:                                                                \
        AclList##FUNC##Attribute(const char* _name, int _num, TYPE _value) \
            : name(_name), value(_value), num(_num){};                     \
        void addAttr(aclopAttr* opAttr) CV_OVERRIDE;                       \
    }

DEFINE_ATTR_DECLEAR(Float, float);
DEFINE_ATTR_DECLEAR(String, const char*);
DEFINE_ATTR_DECLEAR(Int, int);
DEFINE_ATTR_DECLEAR(Bool, bool);
DEFINE_ATTR_LIST_DECLEAR(Int, int64_t*);

#undef DEFINE_ATTR_DECLEAR
#undef DEFINE_ATTR_LIST_DECLEAR

class AscendStream::Impl
{
public:
    aclrtStream stream;
    bool ownStream;
    std::set<std::shared_ptr<uchar>> tensorHolders;
    Impl();
    explicit Impl(aclrtStream stream);
    void AddTensorHolder(const std::shared_ptr<uchar>& tensorData);
};
class AscendEvent::Impl
{
public:
    aclrtEvent event;
    bool ownEvent;

    Impl();
    explicit Impl(aclrtEvent event);
    ~Impl();
};
struct AscendTensor
{
    std::string name;
    std::shared_ptr<uchar> data;
    size_t dataSize;
    std::vector<int64_t> dims;
    aclDataType dtype;
    aclFormat format;
    AscendTensor(){};
    AscendTensor(std::shared_ptr<uchar> _data, size_t _dataSize, int64_t* _dims, size_t _dimSize,
                 aclDataType _dtype, std::string _name = "", aclFormat _format = ACL_FORMAT_ND);
    AscendTensor(std::shared_ptr<uchar> _data, size_t _dataSize, std::vector<int64_t>& _dims,
                 aclDataType _dtype, std::string _name = "", aclFormat _format = ACL_FORMAT_ND)
        : name(_name), data(_data), dataSize(_dataSize), dims(_dims), dtype(_dtype),
          format(_format){};
    AscendTensor(const NpuMat& npuMat, std::string _name = "", aclFormat format = ACL_FORMAT_ND);
};
void aclrtMallocWarpper(void** data, size_t size);
void aclrtFreeWarpper(void* data);

void aclrtMemcpyWarpper(std::shared_ptr<uchar>& dst, size_t offset, const void* src, size_t size,
                        AscendStream& stream);
void aclrtMemcpyWarpper(void* dst, const std::shared_ptr<uchar>& src, size_t offset, size_t size,
                        AscendStream& stream);
void aclrtMemcpyWarpper(std::shared_ptr<uchar>& dst, size_t dstOffset,
                        const std::shared_ptr<uchar>& src, size_t srcOffset, size_t size,
                        AscendStream& stream);
void aclrtMemcpy2dWarpper(std::shared_ptr<uchar>& dst, size_t offset, size_t dpitch,
                          const void* src, size_t spitch, size_t width, size_t length,
                          AscendStream& stream);
void aclrtMemcpy2dWarpper(void* dst, size_t dpitch, const std::shared_ptr<uchar>& src,
                          size_t offset, size_t spitch, size_t width, size_t length,
                          AscendStream& stream);
void aclrtMemsetWarpper(std::shared_ptr<uchar>& ptr, int32_t value, size_t count,
                        AscendStream& stream);

static std::vector<AclAttribute*> emptyattr;
void callAscendOperator(const char* op, std::vector<AscendTensor>& srcs,
                        std::vector<AscendTensor>& dsts, AscendStream& stream,
                        std::vector<AclAttribute*>& attrs = emptyattr);
void callAscendOperator(const NpuMat& src, NpuMat& dst, const char* op, AscendStream& stream,
                        std::vector<AclAttribute*>& attrs = emptyattr);
void callAscendOperator(const NpuMat& src1, const NpuMat& src2, NpuMat& dst, const char* op,
                        AscendStream& stream, std::vector<AclAttribute*>& attrs = emptyattr);
void callAscendOperator(const NpuMat* srcs, size_t srcCount, NpuMat& dst, const char* op,
                        AscendStream& stream, std::vector<AclAttribute*>& attrs = emptyattr);
void callAscendOperator(const NpuMat& src, const Scalar& sc, bool inv, NpuMat& dst, const char* op,
                        AscendStream& stream, std::vector<AclAttribute*>& attrs = emptyattr);
void callAscendOperator(const NpuMat& src, NpuMat* dsts, const size_t dstCount, const char* op,
                        AscendStream& stream, std::vector<AclAttribute*>& attrs = emptyattr);
std::shared_ptr<uchar> mallocAndUpload(void* data, size_t size, AscendStream& stream,
                                       NpuMat::Allocator* allocator = NpuMat::defaultAllocator());
} // namespace cann
} // namespace cv

#endif // OPENCV_CANNOPS_CANN_CALL_HPP
