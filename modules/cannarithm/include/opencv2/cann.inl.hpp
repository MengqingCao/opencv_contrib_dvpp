// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANNINL_HPP
#define OPENCV_CANNINL_HPP

#include "opencv2/cann.hpp"

namespace cv
{
namespace cann
{
inline AclMat::AclMat(AclMat::Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0),
      allocator(allocator_)
{
}

inline AclMat::AclMat(int rows_, int cols_, int type_, AclMat::Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0),
      allocator(allocator_)
{
    if (rows_ > 0 && cols_ > 0)
        create(rows_, cols_, type_);
}

inline AclMat::AclMat(Size size_, int type_, AclMat::Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0),
      allocator(allocator_)
{
    if (size_.height > 0 && size_.width > 0)
        create(size_.height, size_.width, type_);
}

inline AclMat::AclMat(InputArray arr, AclMat::Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0),
      allocator(allocator_)
{
    upload(arr);
}

inline AclMat::AclMat(const AclMat& m)
    : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount),
      datastart(m.datastart), dataend(m.dataend), allocator(m.allocator)
{
    if (refcount)
        CV_XADD(refcount, 1);
}

inline AclMat::~AclMat() { release(); }

inline AclMat& AclMat::operator=(const AclMat& m)
{
    if (this != &m)
    {
        AclMat temp(m);
        swap(temp);
    }

    return *this;
}

inline void AclMat::swap(AclMat& b)
{
    std::swap(flags, b.flags);
    std::swap(rows, b.rows);
    std::swap(cols, b.cols);
    std::swap(step, b.step);
    std::swap(data, b.data);
    std::swap(datastart, b.datastart);
    std::swap(dataend, b.dataend);
    std::swap(refcount, b.refcount);
    std::swap(allocator, b.allocator);
}

inline void AclMat::release()
{
    CV_DbgAssert(allocator != 0);

    if (refcount && CV_XADD(refcount, -1) == 1)
        allocator->free(this);

    dataend = data = datastart = 0;
    step = rows = cols = 0;
    refcount = 0;
}

inline size_t AclMat::elemSize() const { return CV_ELEM_SIZE(flags); }

inline size_t AclMat::elemSize1() const { return CV_ELEM_SIZE1(flags); }

inline int AclMat::type() const { return CV_MAT_TYPE(flags); }

inline int AclMat::depth() const { return CV_MAT_DEPTH(flags); }

inline int AclMat::channels() const { return CV_MAT_CN(flags); }

inline size_t AclMat::step1() const { return step / elemSize1(); }

inline Size AclMat::size() const { return Size(cols, rows); }

inline bool AclMat::empty() const { return data == 0; }

inline AclStream::AclStream(const Ptr<AclStream::Impl>& impl) : impl_(impl) {}

inline AclEvent::AclEvent(const Ptr<AclEvent::Impl>& impl) : impl_(impl) {}
} // namespace cann
} // namespace cv

#endif // OPENCV_CANNINL_HPP
