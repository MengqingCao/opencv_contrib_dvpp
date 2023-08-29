// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <iostream>

namespace
{
class DefaultAllocator : public cv::cann::NpuMat::Allocator
{
public:
    std::shared_ptr<uchar> allocate(size_t size) CV_OVERRIDE;
    bool allocate(cv::cann::NpuMat* mat, int rows, int cols, size_t elemSize) CV_OVERRIDE;
};

std::shared_ptr<uchar> DefaultAllocator::allocate(size_t size)
{
    uchar* data;
    cv::cann::aclrtMallocWarpper((void**)(&data), size);
    return std::shared_ptr<uchar>(data, [](void* ptr) { cv::cann::aclrtFreeWarpper(ptr); });
}

bool DefaultAllocator::allocate(cv::cann::NpuMat* mat, int rows, int cols, size_t elemSize)
{
    mat->data = allocate(elemSize * cols * rows);
    mat->step = cols * elemSize;

    return true;
}

DefaultAllocator cannDefaultAllocator;
cv::cann::NpuMat::Allocator* g_defaultAllocator = &cannDefaultAllocator;
} // namespace

namespace cv
{
namespace cann
{
NpuMat::Allocator* NpuMat::defaultAllocator() { return g_defaultAllocator; }

void NpuMat::setDefaultAllocator(NpuMat::Allocator* allocator)
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

void NpuMat::updateContinuityFlag()
{
    int sz[] = {rows, cols};
    size_t steps[] = {step, elemSize()};
    flags = cv::cann::updateContinuityFlag(flags, 2, sz, steps);
}

void NpuMat::create(int _rows, int _cols, int _type)
{
    CV_DbgAssert(_rows >= 0 && _cols >= 0);

    _type &= Mat::TYPE_MASK;

    if (rows == _rows && cols == _cols && type() == _type && data)
        return;

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

        datastart = data.get();
        dataend = data.get() + step * (rows - 1) + cols * esz;
    }
}

void NpuMat::upload(InputArray arr) { upload(arr, AscendStream::Null()); }

void NpuMat::upload(InputArray arr, AscendStream& stream)
{
    Mat mat = arr.getMat();
    CV_DbgAssert(!mat.empty());
    create(mat.rows, mat.cols, mat.type());
    aclrtMemcpy2dWarpper(data, 0, step, mat.data, mat.step[0], cols * elemSize(), rows, stream);
}

void NpuMat::download(OutputArray dst) const { download(dst, AscendStream::Null()); }

void NpuMat::download(OutputArray _dst, AscendStream& stream) const
{
    CV_DbgAssert(!empty());

    _dst.create(size(), type());
    Mat dst = _dst.getMat();
    aclrtMemcpy2dWarpper(dst.data, dst.step[0], data, 0, step, cols * elemSize(), rows, stream);
}

NpuMat::NpuMat(int rows_, int cols_, int type_, Scalar& s_, NpuMat::Allocator* allocator_)
    : flags(0), rows(rows_), cols(cols_), step(0), datastart(0), dataend(0), allocator(allocator_)
{
    create(rows_, cols_, type_);
    setTo(s_);
}

NpuMat::NpuMat(Size size_, int type_, Scalar& s_, NpuMat::Allocator* allocator_)
    : flags(0), rows(size_.height), cols(size_.width), step(0), datastart(0), dataend(0),
      allocator(allocator_)
{
    create(size_.height, size_.width, type_);
    setTo(s_);
}

NpuMat::NpuMat(InputArray _m, const Rect& roi) : NpuMat(_m, roi, AscendStream::Null()) {}

NpuMat::NpuMat(InputArray _m, const Rect& roi, AscendStream& stream)
    : rows(roi.height), cols(roi.width), allocator(defaultAllocator())
{
    NpuMat m = getInputMat(_m, stream);
    step = m.step;
    data = m.data;
    flags = m.flags;
    CV_Assert(0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y &&
              0 <= roi.height && roi.y + roi.height <= m.rows);
    size_t esz = CV_ELEM_SIZE(flags);
    size_t sizeMem = esz * roi.width * roi.height * m.channels();
    size_t offset = roi.y * m.step + roi.x * esz;

    void* dst = malloc(sizeMem);
    size_t dpitch = roi.width * esz;
    std::shared_ptr<uchar> dstDevice = allocator->allocate(sizeMem);
    aclrtMemcpy2dWarpper(dst, dpitch, data, offset, step, dpitch, roi.height, stream);
    aclrtMemcpy2dWarpper(dstDevice, 0, dpitch, dst, dpitch, dpitch, roi.height, stream);
    data = dstDevice;
    step = dpitch;
    free(dst);
    updateContinuityFlag();
}

NpuMat& NpuMat::setTo(const Scalar& sc) { return setTo(sc, AscendStream::Null()); }

NpuMat& NpuMat::setTo(const Scalar& sc, AscendStream& stream)
{
    size_t totalBytes = (size_t)rows * cols * elemSize();
    if (totalBytes == 0)
        return *this;

    aclrtMemsetWarpper(data, 0, totalBytes, stream);

    NpuMat dst(rows, cols, type());
    // TODO use AssignAdd to avoid memcpy, or use broadcase.
    callAscendOperator(*this, sc, false, dst, "Add", stream);
    swap(dst);

    return *this;
}

NpuMat& NpuMat::setTo(float sc) { return setTo(sc, AscendStream::Null()); }

NpuMat& NpuMat::setTo(float sc, AscendStream& stream)
{
    size_t totalBytes = (size_t)rows * cols * elemSize();
    if (totalBytes == 0)
        return *this;

    aclrtMemsetWarpper(data, 0, totalBytes, stream);

    NpuMat dst(rows, cols, type());
    adds(*this, sc, dst, stream);
    swap(dst);

    return *this;
}

void NpuMat::convertTo(NpuMat& dst, int rtype) const
{
    convertTo(dst, rtype, AscendStream::Null());
}

void NpuMat::convertTo(NpuMat& dst, int _rtype, AscendStream& _stream) const
{
    int cn = channels();
    dst.create(rows, cols, CV_MAKE_TYPE(_rtype, cn));
    callAscendOperator(*this, dst, "Cast", _stream);
}

static NpuMat getNpuMat(InputArray arr)
{
    _InputArray::KindFlag k = arr.kind();
    if (k == _InputArray::NPU_MAT)
    {
        const cann::NpuMat* n_mat = (const cann::NpuMat*)arr.getObj();
        return *n_mat;
    }

    if (k == _InputArray::NONE)
        return cann::NpuMat();

    CV_Error(cv::Error::StsNotImplemented, "getNpuMat is available only for cann::NpuMat");
}

NpuMat getInputMat(InputArray _src, AscendStream& stream)
{
    NpuMat src;
    if (_src.kind() == _InputArray::NPU_MAT)
    {
        src = getNpuMat(_src);
    }
    else if (!_src.empty())
    {
        src.upload(_src, stream);
    }
    return src;
}

NpuMat getOutputMat(OutputArray _dst, int rows, int cols, int type, AscendStream& stream)
{
    CV_UNUSED(stream);
    NpuMat dst;
    if (_dst.kind() == _InputArray::NPU_MAT)
    {
        ((cann::NpuMat*)(_dst.getObj()))->create(rows, cols, type);
        dst = getNpuMat(_dst);
    }
    else
    {
        dst.create(rows, cols, type);
    }
    return dst;
}

void syncOutput(const NpuMat& dst, OutputArray _dst, AscendStream& stream)
{
    if (_dst.kind() != _InputArray::NPU_MAT)
    {
        dst.download(_dst, stream);
    }
}
} // namespace cann
} // namespace cv
