// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv
{
namespace cann
{
static inline aclFormat getAclFormat(const char* type)
{
    if (strcmp(type, "NCHW") == 0)
    {
        return ACL_FORMAT_NCHW;
    }
    else if (strcmp(type, "NHWC") == 0)
    {
        return ACL_FORMAT_NHWC;
    }
    else
    {
        CV_Error(Error::StsBadArg, "Unknown/unsupported matrix format");
    }
}

void transData(const NpuMat& src, NpuMat& dst, const char* from, const char* to,
               AscendStream& stream)
{
    AclStringAttribute fromAttr("src_format", from);
    AclStringAttribute toAttr("dst_format", to);
    std::vector<AclAttribute*> attrs{&fromAttr, &toAttr};

    std::vector<AscendTensor> srcTensors, dstTensors;
    srcTensors.emplace_back(src, "", getAclFormat(from));
    dstTensors.emplace_back(dst, "", getAclFormat(to));
    callAscendOperator("TransData", srcTensors, dstTensors, stream, attrs);
}

void merge(const NpuMat* src, size_t n, NpuMat& dst, AscendStream& stream)
{
    if (src == nullptr || n < 2)
        return;

    int depth = src->depth();
    int rows = src->rows;
    int cols = src->cols;

    // all matrix must have same size and type
    for (size_t i = 1; i < n; i++)
    {
        CV_Assert(src[i].depth() == depth && src[i].channels() == 1);
        CV_Assert(src[i].rows == rows && src[i].cols == cols);
    }

    AclIntAttribute concatDim("concat_dim", 3);
    std::vector<AclAttribute*> attrs{&concatDim};

    std::vector<AscendTensor> srcTensors, dstTensors;

    for (size_t i = 0; i < n; i++)
    {
        srcTensors.emplace_back(src[i], "x" + std::to_string(i));
    }
    dstTensors.emplace_back(dst);

    callAscendOperator("ConcatD", srcTensors, dstTensors, stream, attrs);
}

void merge(const NpuMat* src, size_t n, OutputArray _dst, AscendStream& stream)
{
    NpuMat dst = getOutputMat(_dst, src->rows, src->cols, CV_MAKE_TYPE(src->depth(), n), stream);
    merge(src, n, dst, stream);
    syncOutput(dst, _dst, stream);
}

void merge(const std::vector<NpuMat>& src, OutputArray dst, AscendStream& stream)
{
    merge(&src[0], src.size(), dst, stream);
}

void split(const NpuMat& src, NpuMat* dst, AscendStream& stream)
{
    if (src.empty() || dst == nullptr)
        return;

    int cn = src.channels();
    AclIntAttribute splitDim("split_dim", 3);
    AclIntAttribute numSplit("num_split", cn);

    for (int i = 0; i < cn; i++)
        dst[i].create(src.rows, src.cols, CV_MAKE_TYPE(src.depth(), 1));

    std::vector<AclAttribute*> attrs{&splitDim, &numSplit};

    callAscendOperator(src, dst, cn, "SplitD", stream, attrs);
}

void split(InputArray _src, NpuMat* dst, AscendStream& stream)
{
    NpuMat src = getInputMat(_src, stream);
    split(src, dst, stream);
}

void split(InputArray _src, std::vector<NpuMat>& dst, AscendStream& stream)
{
    NpuMat src = getInputMat(_src, stream);
    dst.resize(src.channels());
    split(_src, &dst[0], stream);
}

void transpose(const NpuMat& src, int64_t* perm, NpuMat& dst, AscendStream& stream)
{
    AclListIntAttribute permAttr("perm", 4, perm);
    std::vector<AclAttribute*> attrs{&permAttr};

    std::vector<AscendTensor> srcTensors, dstTensors;
    srcTensors.emplace_back(src);
    dstTensors.emplace_back(dst);
    callAscendOperator("TransposeD", srcTensors, dstTensors, stream, attrs);
}

void transpose(InputArray _src, OutputArray _dst, AscendStream& stream)
{
    NpuMat src = getInputMat(_src, stream);

    NpuMat dst = getOutputMat(_dst, src.cols, src.rows, src.type(), stream);

    int64_t perm[] = {0, 2, 1, 3};
    transpose(src, perm, dst, stream);
    syncOutput(dst, _dst, stream);
}

void flip(const NpuMat& src, std::vector<int32_t>& asixs, NpuMat& dst, AscendStream& stream)
{
    size_t dataSize = asixs.size() * sizeof(int32_t);
    std::shared_ptr<uchar> axisPtr = mallocAndUpload(&asixs.at(0), dataSize, stream);

    int64_t dims[] = {(int64_t)asixs.size()};
    AscendTensor asixTensor(axisPtr, dataSize, dims, 1, ACL_INT32);

    std::vector<AscendTensor> srcTensors, dstTensors;
    srcTensors.emplace_back(src);
    srcTensors.push_back(std::move(asixTensor));
    dstTensors.emplace_back(dst);
    callAscendOperator("ReverseV2", srcTensors, dstTensors, stream, emptyattr);
}

void flip(InputArray _src, OutputArray _dst, int flipCode, AscendStream& stream)
{
    NpuMat src = getInputMat(_src, stream);
    NpuMat dst = getOutputMat(_dst, src.rows, src.cols, src.type(), stream);

    std::vector<int32_t> asix;
    if (flipCode == 0)
    {
        asix.push_back(1);
    }
    else if (flipCode > 0)
    {
        asix.push_back(2);
    }
    else
    {
        asix.push_back(1);
        asix.push_back(2);
    }
    flip(src, asix, dst, stream);
    syncOutput(dst, _dst, stream);
}

void rotate(InputArray _src, OutputArray _dst, int rotateMode, AscendStream& stream)
{
    CV_Assert(_src.dims() <= 2);
    NpuMat src = getInputMat(_src, stream), dst, tempMat;
    switch (rotateMode)
    {
        case ROTATE_90_CLOCKWISE:
        {
            dst = getOutputMat(_dst, src.cols, src.rows, src.type(), stream);
            transpose(src, tempMat, stream);
            flip(tempMat, dst, 1, stream);
            break;
        }
        case ROTATE_180:
        {
            dst = getOutputMat(_dst, src.rows, src.cols, src.type(), stream);
            flip(src, dst, -1, stream);
            break;
        }
        case ROTATE_90_COUNTERCLOCKWISE:
        {
            dst = getOutputMat(_dst, src.cols, src.rows, src.type(), stream);
            transpose(_src, tempMat, stream);
            flip(tempMat, dst, 0, stream);
            break;
        }
        default:
            break;
    }
    syncOutput(dst, _dst, stream);
}

} // namespace cann
} // namespace cv
