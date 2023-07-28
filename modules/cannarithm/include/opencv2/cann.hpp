// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANN_HPP
#define OPENCV_CANN_HPP

#include "opencv2/core.hpp"

/**
  @defgroup cann Ascend-accelerated Computer Vision
  @{
    @defgroup canncore Core part
    @{
      @defgroup cann_struct Data Structures
      @defgroup cann_init Initializeation and Information
    @}
  @}
 */

namespace cv
{
namespace cann
{
class AclStream;

//! @addtogroup cann_struct
//! @{

//===================================================================================
// AclMat
//===================================================================================

/** @brief Base storage class for NPU memory with reference counting.
 * AclMat class has a similar interface with Mat and AclMat, and work on [Ascend
 * NPU](https://www.hiascend.com/) backend.
 * @sa Mat cuda::GpuMat
 */

class CV_EXPORTS_W AclMat
{
public:
    class CV_EXPORTS_W Allocator
    {
    public:
        virtual ~Allocator() {}

        // allocator must fill data, step and refcount fields
        virtual bool allocate(AclMat* mat, int rows, int cols, size_t elemSize) = 0;
        virtual void free(AclMat* mat) = 0;
    };

    /**
     * @brief Create default allocator for AclMat. This allocator alloc memory from device for
     * specific size.
     */
    CV_WRAP static AclMat::Allocator* defaultAllocator();

    /**
     * @brief Set allocator for AclMat.
     * @param allocator
     */
    CV_WRAP static void setDefaultAllocator(AclMat::Allocator* allocator);

    //! default constructor
    CV_WRAP explicit AclMat(AclMat::Allocator* allocator_ = AclMat::defaultAllocator());

    //! constructs AclMat of the specified size and type
    CV_WRAP AclMat(int rows, int cols, int type,
                   AclMat::Allocator* allocator = AclMat::defaultAllocator());
    //! constructs AclMat of the specified size and type
    CV_WRAP AclMat(Size size, int type, AclMat::Allocator* allocator = AclMat::defaultAllocator());

    //! constructs AclMat and fills it with the specified value s
    CV_WRAP AclMat(int rows, int cols, int type, Scalar& s,
                   AclMat::Allocator* allocator = AclMat::defaultAllocator());
    //! constructs AclMat and fills it with the specified value s
    CV_WRAP AclMat(Size size, int type, Scalar& s,
                   AclMat::Allocator* allocator = AclMat::defaultAllocator());

    //! copy constructor
    CV_WRAP AclMat(const AclMat& m);

    //! constructor for AclMat headers pointing to user-allocated data
    AclMat(int rows, int cols, int type, void* data, size_t step = Mat::AUTO_STEP);
    //! constructor for AclMat headers pointing to user-allocated data
    AclMat(Size size, int type, void* data, size_t step = Mat::AUTO_STEP);

    //! builds AclMat from host memory (Blocking call)
    CV_WRAP explicit AclMat(InputArray arr,
                            AclMat::Allocator* allocator = AclMat::defaultAllocator());

    //! assignment operators
    AclMat& operator=(const AclMat& m);

    //! destructor - calls release()
    ~AclMat();

    //! sets some of the AclMat elements to s (Blocking call)
    CV_WRAP AclMat& setTo(Scalar s);
    //! sets some of the AclMat elements to s (Non-Blocking call)
    CV_WRAP AclMat& setTo(Scalar s, AclStream& stream);

    //! swaps with other smart pointer
    CV_WRAP void swap(AclMat& mat);

    //! allocates new AclMat data unless the AclMat already has specified size and type
    CV_WRAP void create(int rows, int cols, int type);

    //! upload host memory data to AclMat (Blocking call)
    CV_WRAP void upload(InputArray arr);
    //! upload host memory data to AclMat (Non-Blocking call)
    CV_WRAP void upload(InputArray arr, AclStream& stream);

    //! download data from AclMat to host (Blocking call)
    CV_WRAP void download(OutputArray dst) const;
    //! download data from AclMat to host (Non-Blocking call)
    CV_WRAP void download(OutputArray dst, AclStream& stream) const;

    //! converts AclMat to another datatype (Blocking call)
    CV_WRAP void convertTo(CV_OUT AclMat& dst, int rtype) const;

    //! converts AclMat to another datatype (Non-Blocking call)
    CV_WRAP void convertTo(CV_OUT AclMat& dst, int rtype, AclStream& stream) const;

    //! decreases reference counter, deallocate the data when reference counter reaches 0
    CV_WRAP void release();

    //! returns element size in bytes
    CV_WRAP size_t elemSize() const;

    //! returns the size of element channel in bytes
    CV_WRAP size_t elemSize1() const;

    //! returns element type
    CV_WRAP int type() const;

    //! returns element type
    CV_WRAP int depth() const;

    //! returns number of channels
    CV_WRAP int channels() const;

    //! returns step/elemSize1()
    CV_WRAP size_t step1() const;

    //! returns AclMat size : width == number of columns, height == number of rows
    CV_WRAP Size size() const;

    //! returns true if AclMat data is NULL
    CV_WRAP bool empty() const;

    //! internal use method: updates the continuity flag
    CV_WRAP void updateContinuityFlag();

    //! expand one channel mat to multi-channels (Blocking call)
    //! @note, source mat must only have one channel, copy value to all channels.
    CV_WRAP void expandTo(CV_OUT AclMat& dst, int channels) const;

    //! expand one channel mat to multi-channels (Non-Blocking call)
    //! @note, source mat must only have one channel, copy value to all channels.
    CV_WRAP void expandTo(CV_OUT AclMat& dst, int channels, AclStream& stream) const;

    /*! includes several bit-fields:
     - the magic signature
     - continuity flag
     - depth
     - number of channels
     */
    int flags;

    //! the number of rows and columns
    int rows, cols;

    //! a distance between successive rows in bytes; includes the gap if any
    CV_PROP size_t step;

    //! pointer to the data
    uchar* data;

    //! pointer to the reference counter;
    //! when AclMat points to user-allocated data, the pointer is NULL
    int* refcount;

    //! helper fields used in locateROI and adjustROI
    uchar* datastart;
    const uchar* dataend;

    //! allocator
    Allocator* allocator;
};

class AclStream;
class AclStreamAccessor;
class AclEvent;
class AclEventAccessor;
class DefaultDeviceInitializer;

//===================================================================================
// AclStream
//===================================================================================

/** @brief In AscendCL Stream(AclStream) is a task queue. Stream is used to manage the parallelism
 * of tasks. The tasks inside a Stream are executed sequentially, that is, the Stream executes
 * sequentially according to the sent tasks; the tasks in different Streams are executed in
 * parallel.
 *
 * All Non-blocking functions should pass parameter stream, These function returns immediately after
 * the task is submitted. Caller should wait stream until completion.
 *
 * Blocking functions implicityly use the default stream, and synchronize stream before function
 * return.
 * @sa cuda::Stream
 */

// TODO: Stream is defined in namespace cuda, and pybind code does not use a namespace of stream,
// change stream name to AclStream to avoid confilct.
class CV_EXPORTS_W AclStream
{
public:
    CV_WRAP AclStream();

    //! blocks the current CPU thread until all operations in the stream are complete.
    CV_WRAP void waitForCompletion();

    //! blocks the current CPU thread until event trigger.
    CV_WRAP void waitAclEvent(const cv::cann::AclEvent& event);

    /**
     * @brief return default AclStream object for default Acl stream.
     */
    CV_WRAP static AclStream& Null();

    // acl symbols CANNOT used in any hpp files. Use a inner class to avoid acl symbols defined in
    // hpp.
    class Impl;

    // add temporary mat for async release.
    void addToAsyncRelease(const AclMat& mat);

private:
    Ptr<Impl> impl_;
    AclStream(const Ptr<Impl>& impl);

    friend class AclStreamAccessor;
    friend class DefaultDeviceInitializer;
};

/**
 * @brief AclEvent to synchronize between different streams.
 */
class CV_EXPORTS_W AclEvent
{
public:
    CV_WRAP AclEvent();

    //! records an event
    CV_WRAP void record(AclStream& stream = AclStream::Null());

    //! waits for an event to complete
    CV_WRAP void waitForComplete() const;

    class Impl;

private:
    Ptr<Impl> impl_;
    AclEvent(const Ptr<Impl>& impl);

    friend class AclEventAccessor;
};

/** @brief Bindings overload to create a Stream object from the address stored in an existing CANN
 * Runtime API stream pointer (aclrtStream).
 * @param aclStreamAddress Memory address stored in a CANN Runtime API stream pointer
 * (aclrtStream). The created Stream object does not perform any allocation or deallocation and simply
 * wraps existing raw CANN Runtime API stream pointer.
 * @note Overload for generation of bindings only, not exported or intended for use internally fro C++.
 */
CV_EXPORTS_W AclStream wrapStream(size_t aclStreamAddress);

//! @} cann_struct

//===================================================================================
// Initialization & Info
//===================================================================================

//! @addtogroup cann_init
//! @{

//! Get Ascend matrix object from Input array, upload matrix memory if need. (Blocking call)
AclMat getInputMat(InputArray src);
//! Get Ascend matrix object from Input array, upload matrix memory if need. (Non-Blocking call)
AclMat getInputMat(InputArray src, AclStream& stream);

//! Get Ascend matrix object from Output array, upload matrix memory if need.
AclMat getOutputMat(OutputArray dst, int rows, int cols, int type);

//! Sync output matrix to Output array, download matrix memory if need.
void syncOutput(const AclMat& dst, OutputArray _dst);

/**
 * @brief Choose Ascend npu device.
 */
CV_EXPORTS_W void setDevice(int device);

/**
 * @brief Clear all context created in current Ascend device.
 */
CV_EXPORTS_W void resetDevice();

/**
 * @brief Get current Ascend device.
 */
CV_EXPORTS_W int32_t getDevice();

/**
 * @brief init AscendCL.
 */
CV_EXPORTS_W void initAcl();

/**
 * @brief finalize AscendCL.
 * @note finalizeAcl only can be called once for a process. Call this function after all AscendCL
 * options finished.
 */
CV_EXPORTS_W void finalizeAcl();

//! @} cann_init

} // namespace cann
} // namespace cv

#include "opencv2/cann.inl.hpp"

#endif /* OPENCV_CANN_HPP */
