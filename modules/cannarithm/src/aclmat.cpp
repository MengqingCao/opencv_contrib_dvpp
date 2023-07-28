// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace
{
/********************************************AclMat********************************************/
class DefaultAllocator : public cv::cann::AclMat::Allocator
{
public:
    bool allocate(cv::cann::AclMat* mat, int rows, int cols, size_t elemSize) CV_OVERRIDE;
    void free(cv::cann::AclMat* mat) CV_OVERRIDE;
};

bool DefaultAllocator::allocate(cv::cann::AclMat* mat, int rows, int cols, size_t elemSize)
{
    CV_ACL_SAFE_CALL(
        aclrtMalloc((void**)(&mat->data), elemSize * cols * rows, ACL_MEM_MALLOC_HUGE_FIRST));

    mat->step = cols * elemSize;
    mat->refcount = (int*)cv::fastMalloc(sizeof(int));

    return true;
}

void DefaultAllocator::free(cv::cann::AclMat* mat)
{
    aclrtFree(mat->datastart);
    cv::fastFree(mat->refcount);
}

DefaultAllocator cannDefaultAllocator;
cv::cann::AclMat::Allocator* g_defaultAllocator = &cannDefaultAllocator;
} // namespace

namespace cv
{
namespace cann
{
AclMat::Allocator* AclMat::defaultAllocator() { return g_defaultAllocator; }

void AclMat::setDefaultAllocator(AclMat::Allocator* allocator)
{
    CV_Assert(allocator != 0);
    g_defaultAllocator = allocator;
}

// TODO: this function is copied from matrix.cpp, which is a local symbol there and can be
// refreneced.
static int updateContinuityFlag(int flags, int dims, const int* size, const size_t* step)
{
    int i, j;
    for (i = 0; i < dims; i++)
    {
        if (size[i] > 1)
            break;
    }

    uint64 t = (uint64)size[std::min(i, dims - 1)] * CV_MAT_CN(flags);
    for (j = dims - 1; j > i; j--)
    {
        t *= size[j];
        if (step[j] * size[j] < step[j - 1])
            break;
    }

    if (j <= i && t == (uint64)(int)t)
        return flags | Mat::CONTINUOUS_FLAG;
    return flags & ~Mat::CONTINUOUS_FLAG;
}

void AclMat::updateContinuityFlag()
{
    int sz[] = {rows, cols};
    size_t steps[] = {step, elemSize()};
    flags = cv::cann::updateContinuityFlag(flags, 2, sz, steps);
}

AclMat::AclMat(int rows_, int cols_, int type_, void* data_, size_t step_)
    : flags(Mat::MAGIC_VAL + (type_ & Mat::TYPE_MASK)), rows(rows_), cols(cols_), step(step_),
      data((uchar*)data_), refcount(0), datastart((uchar*)data_), dataend((const uchar*)data_),
      allocator(defaultAllocator())
{
    size_t minstep = cols * elemSize();

    if (step == Mat::AUTO_STEP)
    {
        step = minstep;
    }
    else
    {
        if (rows == 1)
            step = minstep;

        CV_DbgAssert(step >= minstep);
    }

    dataend += step * (rows - 1) + minstep;
    updateContinuityFlag();
}

AclMat::AclMat(Size size_, int type_, void* data_, size_t step_)
    : flags(Mat::MAGIC_VAL + (type_ & Mat::TYPE_MASK)), rows(size_.height), cols(size_.width),
      step(step_), data((uchar*)data_), refcount(0), datastart((uchar*)data_),
      dataend((const uchar*)data_), allocator(defaultAllocator())
{
    size_t minstep = cols * elemSize();

    if (step == Mat::AUTO_STEP)
    {
        step = minstep;
    }
    else
    {
        if (rows == 1)
            step = minstep;

        CV_DbgAssert(step >= minstep);
    }

    dataend += step * (rows - 1) + minstep;
    updateContinuityFlag();
}

void AclMat::create(int _rows, int _cols, int _type)
{
    CV_DbgAssert(_rows >= 0 && _cols >= 0);

    _type &= Mat::TYPE_MASK;

    if (rows == _rows && cols == _cols && type() == _type && data)
        return;

    if (data)
        release();

    if (_rows > 0 && _cols > 0)
    {
        flags = Mat::MAGIC_VAL + _type;
        rows = _rows;
        cols = _cols;

        const size_t esz = elemSize();

        bool allocSuccess = allocator->allocate(this, rows, cols, esz);

        if (!allocSuccess)
        {
            // custom allocator fails, try default allocator
            allocator = defaultAllocator();
            allocSuccess = allocator->allocate(this, rows, cols, esz);
            CV_Assert(allocSuccess);
        }

        if (esz * cols == step)
            flags |= Mat::CONTINUOUS_FLAG;

        datastart = data;
        dataend = data + step * (rows - 1) + cols * esz;

        if (refcount)
            *refcount = 1;
    }
}

void AclMat::upload(InputArray arr)
{
    Mat mat = arr.getMat();
    CV_DbgAssert(!mat.empty());
    create(mat.rows, mat.cols, mat.type());
    CV_ACL_SAFE_CALL(aclrtMemcpy2d(data, step, mat.data, mat.step[0], cols * elemSize(), rows,
                                   ACL_MEMCPY_HOST_TO_DEVICE));
}

void AclMat::upload(InputArray arr, AclStream& _stream)
{
    Mat mat = arr.getMat();
    CV_DbgAssert(!mat.empty());
    create(mat.rows, mat.cols, mat.type());
    aclrtStream stream = AclStreamAccessor::getStream(_stream);
    CV_ACL_SAFE_CALL(aclrtMemcpy2dAsync(data, step, mat.data, mat.step[0], cols * elemSize(), rows,
                                        ACL_MEMCPY_HOST_TO_DEVICE, stream));
}

void AclMat::download(OutputArray _dst) const
{
    CV_DbgAssert(!empty());

    _dst.create(size(), type());
    Mat dst = _dst.getMat();
    CV_ACL_SAFE_CALL(aclrtMemcpy2d(dst.data, dst.step[0], data, step, cols * elemSize(), rows,
                                   ACL_MEMCPY_DEVICE_TO_HOST));
}

void AclMat::download(OutputArray _dst, AclStream& _stream) const
{
    CV_DbgAssert(!empty());

    _dst.create(size(), type());
    Mat dst = _dst.getMat();
    aclrtStream stream = AclStreamAccessor::getStream(_stream);
    CV_ACL_SAFE_CALL(aclrtMemcpy2dAsync(dst.data, dst.step[0], data, step, cols * elemSize(), rows,
                                        ACL_MEMCPY_DEVICE_TO_HOST, stream));
}

AclMat::AclMat(int rows_, int cols_, int type_, Scalar& s_, AclMat::Allocator* allocator_)
    : flags(0), rows(rows_), cols(cols_), step(0), data(0), refcount(0), datastart(0), dataend(0),
      allocator(allocator_)
{
    create(rows_, cols_, type_);
    setTo(s_);
}

AclMat::AclMat(Size size_, int type_, Scalar& s_, AclMat::Allocator* allocator_)
    : flags(0), rows(size_.height), cols(size_.width), step(0), data(0), refcount(0), datastart(0),
      dataend(0), allocator(allocator_)
{
    create(size_.height, size_.width, type_);
    setTo(s_);
}

AclMat& AclMat::setTo(Scalar s_) { return setTo(s_, AclStream::Null()); }

AclMat& AclMat::setTo(Scalar s_, AclStream& stream_)
{
    size_t totalBytes = (size_t)rows * cols * elemSize();
    if (totalBytes == 0)
        return *this;

    CV_ACL_SAFE_CALL(aclrtMemset(data, totalBytes, 0, totalBytes));

    Mat scMat(1, 1, type(), s_);
    AclMat scAclMat;
    scAclMat.upload(scMat);

    AclMat dst(rows, cols, type());
    // TODO use AssignAdd to avoid memcpy, or use broadcase.
    aclTwoInputs(*this, scAclMat, dst, "Add", stream_);
    swap(dst);

    return *this;
}

void AclMat::convertTo(AclMat& dst, int rtype) const { convertTo(dst, rtype, AclStream::Null()); }

void AclMat::convertTo(AclMat& dst, int _rtype, AclStream& _stream) const
{
    int cn = channels();
    dst.create(rows, cols, CV_MAKE_TYPE(_rtype, cn));
    aclOneInput(*this, dst, "Cast", _stream);
}

void AclMat::expandTo(CV_OUT AclMat& dst, int chs) const { expandTo(dst, chs, AclStream::Null()); }

void AclMat::expandTo(CV_OUT AclMat& dst, int chs, AclStream& stream) const
{
    CV_Assert(channels() == 1);

    // TODO use inplace expand.
    AclMat NCHW_mat;
    NCHW_mat.create(rows, cols, CV_MAKE_TYPE(depth(), chs));

    aclrtStream rawStream = AclStreamAccessor::getStream(stream);
    size_t expandsize = rows * step * chs;
    uchar* dataptr = (uchar*)NCHW_mat.data;
    for (int ch = 0; ch < chs; ch++)
    {
        if (rawStream == nullptr)
        {
            CV_ACL_SAFE_CALL(
                aclrtMemcpy(dataptr, expandsize, data, rows * step, ACL_MEMCPY_DEVICE_TO_DEVICE));
        }
        else
        {
            CV_ACL_SAFE_CALL(aclrtMemcpyAsync(dataptr, expandsize, data, rows * step,
                                              ACL_MEMCPY_DEVICE_TO_DEVICE, rawStream));
        }

        dataptr += (step * rows);
    }

    dst.create(rows, cols, CV_MAKE_TYPE(depth(), chs));

    transNCHWToNHWC(NCHW_mat, dst, stream);
}

AclStream wrapStream(size_t aclStreamAddress)
{
    return AclStreamAccessor::wrapStream(reinterpret_cast<aclrtStream>(aclStreamAddress));
}

static AclMat getAclMat(InputArray arr)
{
    _InputArray::KindFlag k = arr.kind();
    if (k == _InputArray::ACL_MAT)
    {
        const cann::AclMat* a_mat = (const cann::AclMat*)arr.getObj();
        return *a_mat;
    }

    if (k == _InputArray::NONE)
        return cann::AclMat();

    CV_Error(cv::Error::StsNotImplemented, "getAclMat is available only for cann::AclMat");
}

AclMat getInputMat(InputArray _src)
{
    AclMat src;
    if (_src.kind() == _InputArray::ACL_MAT)
    {
        src = getAclMat(_src);
    }
    else if (!_src.empty())
    {
        src.upload(_src);
    }
    return src;
}

AclMat getInputMat(InputArray _src, AclStream& stream)
{
    AclMat src;
    if (_src.kind() == _InputArray::ACL_MAT)
    {
        src = getAclMat(_src);
    }
    else if (!_src.empty())
    {
        aclrtStream rawStream = AclStreamAccessor::getStream(stream);
        if (rawStream == nullptr)
        {
            src.upload(_src);
        }
        else
        {
            src.upload(_src, stream);
        }
    }
    return src;
}

AclMat getOutputMat(OutputArray _dst, int rows, int cols, int type)
{
    AclMat dst;
    if (_dst.kind() == _InputArray::ACL_MAT)
    {
        ((cann::AclMat*)(_dst.getObj()))->create(rows, cols, type);
        dst = getAclMat(_dst);
    }
    else
    {
        dst.create(rows, cols, type);
    }
    return dst;
}

void syncOutput(const AclMat& dst, OutputArray _dst)
{
    if (_dst.kind() != _InputArray::ACL_MAT)
    {
        dst.download(_dst);
    }
}

/********************************************Device********************************************/

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

    AclStream& getNullAclStream(int deviceId);

private:
    std::vector<Ptr<AclStream>> streams_;
    Mutex streams_mtx_;
};

DefaultDeviceInitializer::DefaultDeviceInitializer() {}

DefaultDeviceInitializer::~DefaultDeviceInitializer() { streams_.clear(); }

AclStream& DefaultDeviceInitializer::getNullAclStream(int deviceId)
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
        Ptr<AclStream::Impl> impl = makePtr<AclStream::Impl>(stream);
        streams_[deviceId] = Ptr<AclStream>(new AclStream(impl));
    }

    return *streams_[deviceId];
}

DefaultDeviceInitializer initializer;

/********************************************AclEvent********************************************/
class AclEvent::Impl
{
public:
    aclrtEvent event;
    bool ownEvent;

    Impl();
    explicit Impl(aclrtEvent event);

    ~Impl();
};

AclEvent::Impl::Impl() : event(nullptr), ownEvent(true)
{
    CV_ACL_SAFE_CALL(aclrtCreateEvent(&event));
}

AclEvent::Impl::Impl(aclrtEvent e) : event(e), ownEvent(false) {}

AclEvent::Impl::~Impl()
{
    if (event && ownEvent)
    {
        CV_ACL_SAFE_CALL(aclrtDestroyEvent(event));
    }
}

aclrtEvent AclEventAccessor::getEvent(const AclEvent& event) { return event.impl_->event; }

AclEvent AclEventAccessor::wrapEvent(aclrtEvent event)
{
    return AclEvent(makePtr<AclEvent::Impl>(event));
}

AclEvent::AclEvent() { impl_ = makePtr<Impl>(); }

void AclEvent::record(AclStream& stream)
{
    CV_ACL_SAFE_CALL(aclrtRecordEvent(impl_->event, AclStreamAccessor::getStream(stream)));
}

void AclEvent::waitForComplete() const { CV_ACL_SAFE_CALL(aclrtSynchronizeEvent(impl_->event)); }

/******************************************AclStream********************************************/
struct AsyncThdArgs
{
    bool isExit;
    void* context;
    pthread_mutex_t mutex;
    AsyncThdArgs() : isExit(false), context(nullptr), mutex(PTHREAD_MUTEX_INITIALIZER) {}
};

class AclStream::Impl
{
public:
    aclrtStream stream;
    bool ownStream;
    AsyncThdArgs asyncThdArgs;
    pthread_t asyncThdId;

    void bindThread();
    void addToAsyncRelease(const AclMat& mat);

    Impl();
    explicit Impl(aclrtStream stream);

    ~Impl();
};

AclStream::Impl::Impl() : stream(nullptr), ownStream(true), asyncThdId(0)
{
    CV_ACL_SAFE_CALL(aclrtCreateStream(&stream));
}

AclStream::Impl::Impl(aclrtStream s) : stream(s), ownStream(false), asyncThdId(0) {}

AclStream::Impl::~Impl()
{
    if (stream && ownStream)
    {
        aclrtSynchronizeStream(stream);
        if (asyncThdId != 0)
        {
            asyncThdArgs.isExit = true;
            CV_ACL_SAFE_CALL(aclrtUnSubscribeReport(asyncThdId, stream));
            (void)pthread_join(asyncThdId, nullptr);
        }
        CV_ACL_SAFE_CALL(aclrtDestroyStream(stream));
    }
}

static void* processReportLoop(void* args_)
{
    AsyncThdArgs* args = (AsyncThdArgs*)args_;
    CV_ACL_SAFE_CALL(aclrtSetCurrentContext(args->context));

    // Wait for subscribe.
    pthread_mutex_lock(&args->mutex);
    pthread_mutex_unlock(&args->mutex);

    while (!args->isExit)
    {
        aclError ret = aclrtProcessReport(-1);
        // Skip error check if exiting. aclrtProcessReport will report an timeout error when
        // unsubscribing.
        if (!args->isExit)
            CV_ACL_SAFE_CALL(ret);
    }

    return (nullptr);
}

void AclStream::Impl::bindThread()
{
    // Only one thread will created. Lock for parallelling.
    pthread_mutex_lock(&asyncThdArgs.mutex);
    if (asyncThdId == 0)
    {
        CV_ACL_SAFE_CALL(aclrtGetCurrentContext(&asyncThdArgs.context));
        (void)pthread_create(&asyncThdId, nullptr, processReportLoop, &asyncThdArgs);
        CV_ACL_SAFE_CALL(aclrtSubscribeReport(asyncThdId, stream));
    }
    pthread_mutex_unlock(&asyncThdArgs.mutex);
}

static void releaseAclMatCB(void* releaseHandle)
{
    if (releaseHandle == nullptr)
        return;
    AclMat* mat = (AclMat*)releaseHandle;
    delete mat;
}

void AclStream::Impl::addToAsyncRelease(const AclMat& mat)
{
    if (stream != nullptr)
    {
        if (asyncThdId == 0)
            bindThread();
        AclMat* releaseHandle = new AclMat(mat);
        CV_ACL_SAFE_CALL(
            aclrtLaunchCallback(releaseAclMatCB, releaseHandle, ACL_CALLBACK_BLOCK, stream));
    }
}

aclrtStream AclStreamAccessor::getStream(const AclStream& stream) { return stream.impl_->stream; }

AclStream AclStreamAccessor::wrapStream(aclrtStream stream)
{
    return AclStream(makePtr<AclStream::Impl>(stream));
}

AclStream::AclStream() { impl_ = makePtr<Impl>(); }

void AclStream::waitForCompletion() { CV_ACL_SAFE_CALL(aclrtSynchronizeStream(impl_->stream)); }

void AclStream::waitAclEvent(const AclEvent& event)
{
    CV_ACL_SAFE_CALL(aclrtStreamWaitEvent(impl_->stream, AclEventAccessor::getEvent(event)));
}

AclStream& AclStream::Null()
{
    const uint32_t deviceId = getDevice();
    return initializer.getNullAclStream(deviceId);
}

void AclStream::addToAsyncRelease(const AclMat& mat) { impl_->addToAsyncRelease(mat); }

} // namespace cann
} // namespace cv
