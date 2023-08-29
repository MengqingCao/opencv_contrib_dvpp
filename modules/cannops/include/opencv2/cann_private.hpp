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
void adds(const NpuMat& arr, float scalar, NpuMat& dst, AscendStream& stream);
void muls(const NpuMat& arr, float scalar, NpuMat& dst, AscendStream& stream);
void transData(const NpuMat& src, NpuMat& dst, const char* from, const char* to,
               AscendStream& stream);
void transpose(const NpuMat& src, int64_t* perm, NpuMat& dst, AscendStream& stream);
void flip(const NpuMat& src, std::vector<int32_t>& asixs, NpuMat& dst, AscendStream& stream);
void merge(const NpuMat* src, size_t n, NpuMat& dst, AscendStream& stream);
void split(const NpuMat& src, NpuMat* dst, AscendStream& stream);

double threshold(NpuMat& src, NpuMat& dst, double thresh, double maxval, int type,
                 AscendStream& stream);
} // namespace cann
} // namespace cv

#endif // OPENCV_CANNOPS_CANN_PRIVATE_HPP