// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANNARITHM_HPP
#define OPENCV_CANNARITHM_HPP

#include "opencv2/cann.hpp"

namespace cv
{
namespace cann
{

/**
  @addtogroup cann
  @{
    @defgroup cannarithm Operations on Matrices
    @{
        @defgroup cannarithm_elem Per-element Operations
    @}
  @}
 */

//! @addtogroup cannarithm_elem
//! @{

/** @brief Computes a matrix-matrix or matrix-scalar sum.
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar. Matrix should have the same size and type as src1 .
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param dtype Optional depth of the output array.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::add cuda::add
 */
CV_EXPORTS_W void add(InputArray src1, InputArray src2, OutputArray dst,
                      InputArray mask = noArray(), int dtype = -1,
                      AclStream& stream = AclStream::Null());
// This code should not be compiled nor analyzed by doxygen. This interface only for python binding
// code generation. add(InputArray, InputArray ...) can accept Scalar as its parametr.(Scalar -> Mat
// -> InputArray)
#ifdef NEVER_DEFINED
CV_EXPORTS_W void add(InputArray src1, Scalar src2, OutputArray dst, InputArray mask = noArray(),
                      int dtype = -1, AclStream& stream = AclStream::Null());
CV_EXPORTS_W void add(Scalar src1, InputArray src2, OutputArray dst, InputArray mask = noArray(),
                      int dtype = -1, AclStream& stream = AclStream::Null());
#endif

/** @brief Computes a matrix-matrix or matrix-scalar difference.
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar. Matrix should have the same size and type as src1 .
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param dtype Optional depth of the output array.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::subtract cuda::subtract
 */
CV_EXPORTS_W void subtract(InputArray src1, InputArray src2, OutputArray dst,
                           InputArray mask = noArray(), int dtype = -1,
                           AclStream& stream = AclStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void subtract(InputArray src1, Scalar src2, OutputArray dst,
                           InputArray mask = noArray(), int dtype = -1,
                           AclStream& stream = AclStream::Null());
CV_EXPORTS_W void subtract(Scalar src1, InputArray src2, OutputArray dst,
                           InputArray mask = noArray(), int dtype = -1,
                           AclStream& stream = AclStream::Null());
#endif

/** @brief Computes a matrix-matrix or matrix-scalar per-element product.
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar. Matrix should have the same size and type as src1 .
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param scale Optional scale factor.
 * @param dtype Optional depth of the output array.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::multiply cuda::multiply
 */
CV_EXPORTS_W void multiply(InputArray src1, InputArray src2, OutputArray dst, float scale,
                           int dtype = -1, AclStream& stream = AclStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void multiply(InputArray src1, Scalar src2, OutputArray dst, float scale,
                           int dtype = -1, AclStream& stream = AclStream::Null());
CV_EXPORTS_W void multiply(Scalar src1, InputArray src2, OutputArray dst, float scale,
                           int dtype = -1, AclStream& stream = AclStream::Null());
#endif

/** @brief Computes a matrix-matrix or matrix-scalar division.
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar. Matrix should have the same size and type as src1 .
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param scale Optional scale factor.
 * @param dtype Optional depth of the output array.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::divide cuda::divide
 */
CV_EXPORTS_W void divide(InputArray src1, InputArray src2, OutputArray dst, float scale,
                         int dtype = -1, AclStream& stream = AclStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void divide(InputArray src1, Scalar src2, OutputArray dst, float scale, int dtype = -1,
                         AclStream& stream = AclStream::Null());
CV_EXPORTS_W void divide(Scalar src1, InputArray src2, OutputArray dst, float scale, int dtype = -1,
                         AclStream& stream = AclStream::Null());
#endif

/** @brief Performs a per-element bitwise conjunction of two matrices (or of matrix and scalar).
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar.
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::bitwise_and cuda::bitwise_and
 */
CV_EXPORTS_W void bitwise_and(InputArray src1, InputArray src2, OutputArray dst,
                              InputArray mask = noArray(), AclStream& stream = AclStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void bitwise_and(InputArray src1, Scalar src2, OutputArray dst,
                              InputArray mask = noArray(), AclStream& stream = AclStream::Null());
CV_EXPORTS_W void bitwise_and(Scalar src1, InputArray src2, OutputArray dst,
                              InputArray mask = noArray(), AclStream& stream = AclStream::Null());
#endif

/** @brief Performs a per-element bitwise disjunction of two matrices (or of matrix and scalar).
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar.
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::bitwise_or cuda::bitwise_or
 */
CV_EXPORTS_W void bitwise_or(InputArray src1, InputArray src2, OutputArray dst,
                             InputArray mask = noArray(), AclStream& stream = AclStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void bitwise_or(InputArray src1, Scalar src2, OutputArray dst,
                             InputArray mask = noArray(), AclStream& stream = AclStream::Null());
CV_EXPORTS_W void bitwise_or(Scalar src1, InputArray src2, OutputArray dst,
                             InputArray mask = noArray(), AclStream& stream = AclStream::Null());
#endif

/** @brief Performs a per-element bitwise exclusive or operation of two matrices (or of matrix and
 * scalar).
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar.
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::bitwise_xor cuda::bitwise_xor
 */
CV_EXPORTS_W void bitwise_xor(InputArray src1, InputArray src2, OutputArray dst,
                              InputArray mask = noArray(), AclStream& stream = AclStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void bitwise_xor(InputArray src1, Scalar src2, OutputArray dst,
                              InputArray mask = noArray(), AclStream& stream = AclStream::Null());
CV_EXPORTS_W void bitwise_xor(Scalar src1, InputArray src2, OutputArray dst,
                              InputArray mask = noArray(), AclStream& stream = AclStream::Null());
#endif

//! @} cannarithm_elem

} // namespace cann
} // namespace cv

#endif /* OPENCV_CANNARITHM_HPP */
