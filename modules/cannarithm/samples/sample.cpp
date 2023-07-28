// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

//g++ -o sample sample.cpp -I opencv/include/opencv4/ -L opencv/build/install/lib/ -l opencv_cannarithm -l opencv_core -l opencv_imgcodecs

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cann.hpp>
#include <opencv2/cann_arithm.hpp>

int main()
{
    cv::Mat img = cv::imread("/path/to/img");

    cv::cann::initAcl();
    cv::cann::setDevice(0);

    cv::cann::AclMat aclMat = cv::cann::AclMat();
    aclMat.upload(img);

    cv::cann::AclMat aclMatSum;
    cv::cann::add(aclMat, aclMat, aclMatSum);
    cv::Mat imgResult;
    aclMatSum.download(imgResult);
    std::cout<<imgResult<<std::endl;

    cv::cann::resetDevice();
    cv::cann::finalizeAcl();

    return 0;
}
