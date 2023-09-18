// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANNOPS_CANN_PRIVATE_HPP
#define OPENCV_CANNOPS_CANN_PRIVATE_HPP
#include "opencv2/cann.hpp"

namespace cv
{
namespace cann
{
void adds(const AscendMat& arr, float scalar, AscendMat& dst, AscendStream& stream);
void muls(const AscendMat& arr, float scalar, AscendMat& dst, AscendStream& stream);
void transData(const AscendMat& src, AscendMat& dst, const char* from, const char* to,
               AscendStream& stream);
void transpose(const AscendMat& src, int64_t* perm, AscendMat& dst, AscendStream& stream);
void flip(const AscendMat& src, std::vector<int32_t>& asixs, AscendMat& dst, AscendStream& stream);
void merge(const AscendMat* src, size_t n, AscendMat& dst, AscendStream& stream);
void split(const AscendMat& src, AscendMat* dst, AscendStream& stream);

double threshold(AscendMat& src, AscendMat& dst, double thresh, double maxval, int type,
                 AscendStream& stream);
} // namespace cann
} // namespace cv

#endif // OPENCV_CANNOPS_CANN_PRIVATE_HPP