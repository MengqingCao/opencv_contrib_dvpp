// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include "precomp.hpp"
#include "opencv2/core/private.hpp"
namespace cv
{
namespace cann
{
/*******************************Acl Error Checker*****************************/
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

#define CV_ACL_SAFE_CALL(expr) checkAclError((expr), __FILE__, __LINE__, CV_Func)
#define CV_ACL_SAFE_CALL_PTR(expr)                     \
    ({                                                 \
        auto ptr = (expr);                             \
        checkAclPtr(ptr, __FILE__, __LINE__, CV_Func); \
        ptr;                                           \
    })

/*****************************Acl Operator Attribute**************************/
#define DEFINE_ATTR_BODY(FUNC)                                     \
    void Acl##FUNC##Attribute::addAttr(aclopAttr* opAttr)          \
    {                                                              \
        CV_ACL_SAFE_CALL(aclopSetAttr##FUNC(opAttr, name, value)); \
    }

#define DEFINE_ATTR_LIST_BODY(FUNC)                                         \
    void AclList##FUNC##Attribute::addAttr(aclopAttr* opAttr)               \
    {                                                                       \
        CV_ACL_SAFE_CALL(aclopSetAttrList##FUNC(opAttr, name, num, value)); \
    }

DEFINE_ATTR_BODY(Float);
DEFINE_ATTR_BODY(String);
DEFINE_ATTR_BODY(Int);
DEFINE_ATTR_BODY(Bool);
DEFINE_ATTR_LIST_BODY(Int);

#undef DEFINE_ATTR_BODY
#undef DEFINE_ATTR_LIST_BODY

/******************************Acl Runtime Warpper****************************/
void aclrtMallocWarpper(void** data, size_t size)
{
    CV_ACL_SAFE_CALL(aclrtMalloc(data, size, ACL_MEM_MALLOC_HUGE_FIRST));
}

void aclrtFreeWarpper(void* data) { CV_ACL_SAFE_CALL(aclrtFree(data)); }
// TODO should define dstMax?
void aclrtMemcpyWarpper(std::shared_ptr<uchar>& dst, size_t offset, const void* src, size_t size,
                        AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(
            aclrtMemcpy(dst.get() + offset, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
    else
    {
        CV_ACL_SAFE_CALL(aclrtMemcpyAsync(dst.get() + offset, size, src, size,
                                          ACL_MEMCPY_HOST_TO_DEVICE, rawStream));
        if (offset == 0)
            stream.addTensorHolder(dst);
    }
}

void aclrtMemcpyWarpper(void* dst, const std::shared_ptr<uchar>& src, size_t offset, size_t size,
                        AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(
            aclrtMemcpy(dst, size, src.get() + offset, size, ACL_MEMCPY_DEVICE_TO_HOST));
    else
    {
        CV_ACL_SAFE_CALL(aclrtMemcpyAsync(dst, size, src.get() + offset, size,
                                          ACL_MEMCPY_DEVICE_TO_HOST, rawStream));
        if (offset == 0)
            stream.addTensorHolder(src);
    }
}

void aclrtMemcpyWarpper(std::shared_ptr<uchar>& dst, size_t dstOffset,
                        const std::shared_ptr<uchar>& src, size_t srcOffset, size_t size,
                        AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtMemcpy(dst.get() + dstOffset, size, src.get() + srcOffset, size,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE));
    else
    {
        CV_ACL_SAFE_CALL(aclrtMemcpyAsync(dst.get() + dstOffset, size, src.get() + srcOffset, size,
                                          ACL_MEMCPY_DEVICE_TO_DEVICE, rawStream));
        if (srcOffset == 0)
            stream.addTensorHolder(src);
        if (dstOffset == 0)
            stream.addTensorHolder(dst);
    }
}

void aclrtMemcpy2dWarpper(std::shared_ptr<uchar>& dst, size_t offset, size_t dpitch,
                          const void* src, size_t spitch, size_t width, size_t length,
                          AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtMemcpy2d(dst.get() + offset, dpitch, src, spitch, width, length,
                                       ACL_MEMCPY_HOST_TO_DEVICE));
    else
    {
        CV_ACL_SAFE_CALL(aclrtMemcpy2dAsync(dst.get() + offset, dpitch, src, spitch, width, length,
                                            ACL_MEMCPY_HOST_TO_DEVICE, rawStream));
        stream.addTensorHolder(dst);
    }
}

void aclrtMemcpy2dWarpper(void* dst, size_t dpitch, const std::shared_ptr<uchar>& src,
                          size_t offset, size_t spitch, size_t width, size_t length,
                          AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtMemcpy2d(dst, dpitch, src.get() + offset, spitch, width, length,
                                       ACL_MEMCPY_DEVICE_TO_HOST));
    else
    {
        CV_ACL_SAFE_CALL(aclrtMemcpy2dAsync(dst, dpitch, src.get() + offset, spitch, width, length,
                                            ACL_MEMCPY_DEVICE_TO_HOST, rawStream));
        stream.addTensorHolder(src);
    }
}

void aclrtMemsetWarpper(std::shared_ptr<uchar>& ptr, int32_t value, size_t count,
                        AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtMemset(ptr.get(), count, value, count));
    else
    {
        CV_ACL_SAFE_CALL(aclrtMemsetAsync(ptr.get(), count, value, count, rawStream));
        stream.addTensorHolder(ptr);
    }
}

/**************************Acl attribute preparation**************************/
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

#define CANN_PREPARE_INPUTDESC(var, name, ...)                               \
    do                                                                       \
    {                                                                        \
        auto _rPtr = CV_ACL_SAFE_CALL_PTR(aclCreateTensorDesc(__VA_ARGS__)); \
        if (_rPtr != nullptr)                                                \
        {                                                                    \
            if (name != nullptr and strlen(name) != 0)                       \
                aclSetTensorDescName(_rPtr, name);                           \
            var.inputDesc_.push_back(_rPtr);                                 \
        }                                                                    \
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

/********************************Ascend Tensor********************************/
static inline aclDataType getACLType(int opencvdepth)
{
    switch (opencvdepth)
    {
        case CV_8S:
            return ACL_INT8;
        case CV_16S:
            return ACL_INT16;
        case CV_8U:
            return ACL_UINT8;
        case CV_16U:
            return ACL_UINT16;
        case CV_32S:
            return ACL_INT32;
        case CV_32F:
            return ACL_FLOAT;
        case CV_64F:
            return ACL_DOUBLE;
        case CV_16F:
            return ACL_FLOAT16;
        default:
            return ACL_DT_UNDEFINED;
    }
}

AscendTensor::AscendTensor(std::shared_ptr<uchar> _data, size_t _dataSize, int64_t* _dims,
                           size_t _dimSize, aclDataType _dtype, std::string _name,
                           aclFormat _format)
    : name(_name), data(_data), dataSize(_dataSize), dtype(_dtype), format(_format)
{
    dims.assign(_dims, _dims + _dimSize);
}

AscendTensor::AscendTensor(const NpuMat& npuMat, std::string _name, aclFormat _format)
    : name(_name), format(_format)
{
    data = npuMat.data;
    // Ascend can't process with gaps in matrix.
    CV_Assert(npuMat.isContinuous());
    dataSize = npuMat.rows * npuMat.cols * npuMat.elemSize();

    switch (_format)
    {
        case ACL_FORMAT_NHWC:
        case ACL_FORMAT_ND:
            dims.resize(4);
            // Batch, default = 1.
            dims[0] = 1;
            // Default OpenCV image format = NHWC.
            dims[1] = npuMat.rows;
            dims[2] = npuMat.cols;
            dims[3] = npuMat.channels();
            break;
        case ACL_FORMAT_NCHW:
            dims.resize(4);
            dims[0] = 1;
            dims[1] = npuMat.channels();
            dims[2] = npuMat.rows;
            dims[3] = npuMat.cols;
            break;
        default:
            CV_Error(Error::StsBadArg, "Unknown/unsupported matrix format");
    }

    dtype = getACLType(npuMat.depth());
}

/**********************************Device*************************************/
void setDevice(int device_id)
{
    aclrtContext context;
    CV_ACL_SAFE_CALL(aclrtSetDevice(device_id));
    CV_ACL_SAFE_CALL(aclrtCreateContext(&context, device_id));
}

void resetDevice() { CV_ACL_SAFE_CALL(aclrtResetDevice(getDevice())); }

int32_t getDevice()
{
    int32_t deviceId;
    CV_ACL_SAFE_CALL(aclrtGetDevice(&deviceId));
    return deviceId;
}

void initAcl() { CV_ACL_SAFE_CALL(aclInit(nullptr)); }

void finalizeAcl() { CV_ACL_SAFE_CALL(aclFinalize()); }

class DefaultDeviceInitializer
{
public:
    DefaultDeviceInitializer();
    ~DefaultDeviceInitializer();

    AscendStream& getNullAscendStream(int deviceId);

private:
    std::vector<Ptr<AscendStream>> streams_;
    Mutex streams_mtx_;
};

DefaultDeviceInitializer::DefaultDeviceInitializer() {}

DefaultDeviceInitializer::~DefaultDeviceInitializer() { streams_.clear(); }

AscendStream& DefaultDeviceInitializer::getNullAscendStream(int deviceId)
{
    AutoLock lock(streams_mtx_);

    if (streams_.empty())
    {
        uint32_t deviceCount;
        CV_ACL_SAFE_CALL(aclrtGetDeviceCount(&deviceCount));

        if (deviceCount > 0)
            streams_.resize(deviceCount);
    }

    CV_DbgAssert(deviceId >= 0 && deviceId < static_cast<int>(streams_.size()));

    if (streams_[deviceId].empty())
    {
        aclrtStream stream = nullptr;
        Ptr<AscendStream::Impl> impl = makePtr<AscendStream::Impl>(stream);
        streams_[deviceId] = Ptr<AscendStream>(new AscendStream(impl));
    }

    return *streams_[deviceId];
}

DefaultDeviceInitializer initializer;

/***********************************Event*************************************/
AscendEvent::Impl::Impl() : event(nullptr), ownEvent(true)
{
    CV_ACL_SAFE_CALL(aclrtCreateEvent(&event));
}

AscendEvent::Impl::Impl(aclrtEvent e) : event(e), ownEvent(false) {}

AscendEvent::Impl::~Impl()
{
    if (event && ownEvent)
    {
        CV_ACL_SAFE_CALL(aclrtDestroyEvent(event));
    }
}

aclrtEvent AscendEventAccessor::getEvent(const AscendEvent& event) { return event.impl_->event; }

AscendEvent AscendEventAccessor::wrapEvent(aclrtEvent event)
{
    return AscendEvent(makePtr<AscendEvent::Impl>(event));
}

AscendEvent::AscendEvent() { impl_ = makePtr<Impl>(); }

void AscendEvent::record(AscendStream& stream)
{
    CV_ACL_SAFE_CALL(aclrtRecordEvent(impl_->event, AscendStreamAccessor::getStream(stream)));
}

void AscendEvent::waitForComplete() const { CV_ACL_SAFE_CALL(aclrtSynchronizeEvent(impl_->event)); }

/************************************Stream***********************************/
void AscendStream::Impl::AddTensorHolder(const std::shared_ptr<uchar>& tensorData)
{
    tensorHolders.insert(tensorData);
}

AscendStream::Impl::Impl() : stream(nullptr), ownStream(true)
{
    CV_ACL_SAFE_CALL(aclrtCreateStream(&stream));
}

AscendStream::Impl::Impl(aclrtStream s) : stream(s), ownStream(false) {}

aclrtStream AscendStreamAccessor::getStream(const AscendStream& stream)
{
    return stream.impl_->stream;
}

AscendStream AscendStreamAccessor::wrapStream(aclrtStream stream)
{
    return AscendStream(makePtr<AscendStream::Impl>(stream));
}

AscendStream wrapStream(size_t AscendStreamAddress)
{
    return AscendStreamAccessor::wrapStream(reinterpret_cast<aclrtStream>(AscendStreamAddress));
}

AscendStream::AscendStream() { impl_ = makePtr<Impl>(); }

void AscendStream::waitForCompletion()
{
    CV_ACL_SAFE_CALL(aclrtSynchronizeStream(impl_->stream));
    impl_->tensorHolders.clear();
}

void AscendStream::waitAscendEvent(const AscendEvent& event)
{
    CV_ACL_SAFE_CALL(aclrtStreamWaitEvent(impl_->stream, AscendEventAccessor::getEvent(event)));
}

AscendStream& AscendStream::Null()
{
    const uint32_t deviceId = getDevice();
    return initializer.getNullAscendStream(deviceId);
}

void AscendStream::addTensorHolder(const std::shared_ptr<uchar>& holder)
{
    impl_->AddTensorHolder(holder);
}

/********************************Operator caller******************************/
std::shared_ptr<uchar> mallocAndUpload(void* data, size_t size, AscendStream& stream,
                                       NpuMat::Allocator* allocator)
{
    std::shared_ptr<uchar> ptr = allocator->allocate(size);
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);

    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtMemcpy(ptr.get(), size, data, size, ACL_MEMCPY_HOST_TO_DEVICE));
    else
        CV_ACL_SAFE_CALL(
            aclrtMemcpyAsync(ptr.get(), size, data, size, ACL_MEMCPY_HOST_TO_DEVICE, rawStream));
    return ptr;
}

void callAscendOperator(const char* op, std::vector<AscendTensor>& srcs,
                        std::vector<AscendTensor>& dsts, AscendStream& stream,
                        std::vector<AclAttribute*>& attrs)
{
    CannPreparation prepare;
    for (AclAttribute* attr : attrs)
    {
        attr->addAttr(prepare.opAttr_);
    }

    for (const AscendTensor& src : srcs)
    {
        CANN_PREPARE_INPUTDESC(prepare, src.name.c_str(), src.dtype, src.dims.size(),
                               &src.dims.at(0), src.format);
        CANN_PREPARE_INPUTBUFFER(prepare, src.data.get(), src.dataSize);
    }

    for (const AscendTensor& dst : dsts)
    {
        CANN_PREPARE_OUTPUTDESC(prepare, dst.dtype, dst.dims.size(), &dst.dims.at(0), dst.format);
        CANN_PREPARE_OUTPUTBUFFER(prepare, dst.data.get(), dst.dataSize);
    }

    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);

    CV_ACL_SAFE_CALL(aclopCompileAndExecute(
        op, prepare.inputDesc_.size(), prepare.inputDesc_.data(), prepare.inputBuffers_.data(),
        prepare.outputDesc_.size(), prepare.outputDesc_.data(), prepare.outputBuffers_.data(),
        prepare.opAttr_, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, rawStream));
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtSynchronizeStream(rawStream));
    else
    {
        for (const AscendTensor& src : srcs)
        {
            stream.addTensorHolder(src.data);
        }
        for (const AscendTensor& dst : dsts)
        {
            stream.addTensorHolder(dst.data);
        }
    }
}

void callAscendOperator(const NpuMat& src, NpuMat& dst, const char* op, AscendStream& stream,
                        std::vector<AclAttribute*>& attrs)
{
    std::vector<AscendTensor> srcTensors, dstTensors;
    srcTensors.emplace_back(src);
    dstTensors.emplace_back(dst);
    callAscendOperator(op, srcTensors, dstTensors, stream, attrs);
}

void callAscendOperator(const NpuMat& src1, const NpuMat& src2, NpuMat& dst, const char* op,
                        AscendStream& stream, std::vector<AclAttribute*>& attrs)
{
    std::vector<AscendTensor> srcTensors, dstTensors;
    srcTensors.emplace_back(src1);
    srcTensors.emplace_back(src2);
    dstTensors.emplace_back(dst);
    callAscendOperator(op, srcTensors, dstTensors, stream, attrs);
}

void callAscendOperator(const NpuMat* srcs, const size_t srcCount, NpuMat& dst, const char* op,
                        AscendStream& stream, std::vector<AclAttribute*>& attrs)
{
    std::vector<AscendTensor> srcTensors, dstTensors;
    for (size_t i = 0; i < srcCount; i++)
    {
        srcTensors.emplace_back(srcs[i]);
    }
    dstTensors.emplace_back(dst);
    callAscendOperator(op, srcTensors, dstTensors, stream, attrs);
}

void callAscendOperator(const NpuMat& src, const Scalar& sc, bool inv, NpuMat& dst, const char* op,
                        AscendStream& stream, std::vector<AclAttribute*>& attrs)
{
    uchar rawData[32];
    cv::scalarToRawData(sc, rawData, src.type(), 0);
    std::shared_ptr<uchar> scPtr = mallocAndUpload(rawData, src.elemSize(), stream);

    int64_t dims[] = {1, 1, 1, src.channels()};
    AscendTensor scalarTensor(scPtr, src.elemSize(), dims, sizeof(dims) / sizeof(dims[0]),
                              getACLType(src.depth()));

    std::vector<AscendTensor> srcTensors, dstTensors;

    srcTensors.emplace_back(src);
    srcTensors.push_back(scalarTensor);

    if (inv)
        std::swap(srcTensors[0], srcTensors[1]);

    dstTensors.emplace_back(dst);
    callAscendOperator(op, srcTensors, dstTensors, stream, attrs);
}

void callAscendOperator(const NpuMat& src, NpuMat* dsts, const size_t dstCount, const char* op,
                        AscendStream& stream, std::vector<AclAttribute*>& attrs)
{
    std::vector<AscendTensor> srcTensors, dstTensors;
    srcTensors.emplace_back(src);
    for (size_t i = 0; i < dstCount; i++)
    {
        dstTensors.emplace_back(dsts[i]);
    }
    callAscendOperator(op, srcTensors, dstTensors, stream, attrs);
}

} // namespace cann
} // namespace cv
