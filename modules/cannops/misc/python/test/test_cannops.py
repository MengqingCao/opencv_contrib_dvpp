# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

from tests_common import NewOpenCVTests
import cv2 as cv
import numpy as np


def genMask(mask, listx, listy):
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if (row in listx and col in listx) or (row in listy and col in listy):
                mask[row][col] = 1
    mask = mask.astype(np.uint8)
    return mask


mask = np.zeros((5, 5))
listx = [0, 1]
listy = [1, 2]
mask = genMask(mask, listx, listy)


class cannop_test(NewOpenCVTests):
    def test_ascend(self):
        cv.cann.initAcl()
        cv.cann.getDevice()
        cv.cann.setDevice(0)
        stream = cv.cann.AscendStream_Null()
        cv.cann.wrapStream(id(stream))
        cv.cann.resetDevice()

    def test_arithmetic(self):
        npMat1 = np.random.random((5, 5, 3)).astype(int)
        npMat2 = np.random.random((5, 5, 3)).astype(int)
        cv.cann.setDevice(0)

        self.assertTrue(np.allclose(cv.cann.add(
            npMat1, npMat2), cv.add(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.subtract(
            npMat1, npMat2), cv.subtract(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.multiply(
            npMat1, npMat2, scale=2), cv.multiply(npMat1, npMat2, scale=2)))
        self.assertTrue(np.allclose(cv.cann.divide(
            npMat1, npMat2, scale=2), cv.divide(npMat1, npMat2, scale=2)))

        # mask
        self.assertTrue(np.allclose(cv.cann.add(
            npMat1, npMat2, mask=mask), cv.add(npMat1, npMat2, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.subtract(
            npMat1, npMat2, mask=mask), cv.subtract(npMat1, npMat2, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.multiply(npMat1, npMat2, scale=2),
                                    cv.multiply(npMat1, npMat2, scale=2)))
        self.assertTrue(np.allclose(cv.cann.divide(npMat1, npMat2, scale=2),
                                    cv.divide(npMat1, npMat2, scale=2)))
        self.assertTrue(np.allclose(cv.cann.addWeighted(npMat1, 2, npMat2, 4, 3),
                                    cv.addWeighted(npMat1, 2, npMat2, 4, 3)))

        # stream
        stream = cv.cann.AscendStream()
        matDst = cv.cann.add(npMat1, npMat2, stream=stream)
        stream.waitForCompletion()
        self.assertTrue(np.allclose(matDst, cv.add(npMat1, npMat2)))
        matDst = cv.cann.add(npMat1, npMat2, mask=mask, stream=stream)
        stream.waitForCompletion()
        self.assertTrue(np.allclose(matDst, cv.add(npMat1, npMat2, mask=mask)))
        matDst = cv.cann.subtract(npMat1, npMat2, mask=mask, stream=stream)
        stream.waitForCompletion()
        self.assertTrue(np.allclose(
            matDst, cv.subtract(npMat1, npMat2, mask=mask)))

        cv.cann.resetDevice()

    def test_logical(self):
        npMat1 = np.random.random((5, 5, 3)).astype(np.uint16)
        npMat2 = np.random.random((5, 5, 3)).astype(np.uint16)
        cv.cann.setDevice(0)

        self.assertTrue(np.allclose(cv.cann.bitwise_or(npMat1, npMat2),
                                    cv.bitwise_or(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_or(
            npMat1, npMat2), cv.bitwise_or(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_and(npMat1, npMat2),
                                    cv.bitwise_and(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_and(
            npMat1, npMat2), cv.bitwise_and(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_xor(npMat1, npMat2),
                                    cv.bitwise_xor(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_xor(
            npMat1, npMat2), cv.bitwise_xor(npMat1, npMat2)))
        self.assertTrue(np.allclose(cv.cann.bitwise_not(npMat1),
                                    cv.bitwise_not(npMat1)))
        self.assertTrue(np.allclose(
            cv.cann.bitwise_not(npMat1), cv.bitwise_not(npMat1)))
        self.assertTrue(np.allclose(cv.cann.bitwise_and(npMat1, npMat2, mask=mask),
                                    cv.bitwise_and(npMat1, npMat2, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.bitwise_or(npMat1, npMat2, mask=mask),
                                    cv.bitwise_or(npMat1, npMat2, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.bitwise_not(npMat1, mask=mask),
                                    cv.bitwise_not(npMat1, mask=mask)))
        self.assertTrue(np.allclose(cv.cann.bitwise_xor(npMat1, npMat2, mask=mask),
                                    cv.bitwise_xor(npMat1, npMat2, mask=mask)))
        cv.cann.resetDevice()

    def test_imgproc(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cv.cann.setDevice(0)

        self.assertTrue(np.allclose(
            cv.cann.merge(cv.cann.split(npMat)), npMat))

        self.assertTrue(np.allclose(
            cv.cann.transpose(npMat), cv.transpose(npMat)))

        flipMode = [0, 1, -1]
        for fMode in flipMode:
            self.assertTrue(np.allclose(cv.cann.flip(
                npMat, fMode), cv.flip(npMat, fMode)))

        rotateMode = [0, 1, 2]
        for rMode in rotateMode:
            self.assertTrue(np.allclose(cv.cann.rotate(
                npMat, rMode), cv.rotate(npMat, rMode)))

        cvtModeC1 = [cv.COLOR_GRAY2BGR, cv.COLOR_GRAY2BGRA]
        cvtModeC3 = [cv.COLOR_BGR2GRAY, cv.COLOR_BGRA2BGR, cv.COLOR_BGR2RGBA, cv.COLOR_RGBA2BGR,
                     cv.COLOR_BGR2RGB, cv.COLOR_BGRA2RGBA, cv.COLOR_RGB2GRAY, cv.COLOR_BGRA2GRAY,
                     cv.COLOR_RGBA2GRAY, cv.COLOR_BGR2BGRA, cv.COLOR_BGR2YUV, cv.COLOR_RGB2YUV,
                     cv.COLOR_YUV2BGR, cv.COLOR_YUV2RGB, cv.COLOR_BGR2YCrCb, cv.COLOR_RGB2YCrCb,
                     cv.COLOR_YCrCb2BGR, cv.COLOR_YCrCb2RGB, cv.COLOR_BGR2XYZ, cv.COLOR_RGB2XYZ,
                     cv.COLOR_XYZ2BGR, cv.COLOR_XYZ2RGB,]
        for cvtM in cvtModeC3:
            self.assertTrue(np.allclose(cv.cann.cvtColor(
                npMat, cvtM), cv.cvtColor(npMat, cvtM), 1))
        npMatC1 = (np.random.random((128, 128, 1)) * 255).astype(np.uint8)
        for cvtM in cvtModeC1:
            self.assertTrue(np.allclose(cv.cann.cvtColor(
                npMatC1, cvtM), cv.cvtColor(npMatC1, cvtM), 1))

        threshType = [cv.THRESH_BINARY, cv.THRESH_BINARY_INV,
                      cv.THRESH_TRUNC, cv.THRESH_TOZERO, cv.THRESH_TOZERO_INV]
        for tType in threshType:
            cvRet, cvThresh = cv.threshold(
                npMat.astype(np.uint8), 127, 255, tType)
            cannRet, cannThresh = cv.cann.threshold(
                npMat.astype(np.float32), 127, 255, tType)
            self.assertTrue(np.allclose(cvThresh, cannThresh))
            self.assertTrue(np.allclose(cvRet, cannRet))
        cv.cann.resetDevice()


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
