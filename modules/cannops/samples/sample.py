# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

import numpy as np
import cv2

img = cv2.imread("/path/to/img")

cv2.cann.initAcl()
cv2.cann.setDevice(0)

npuMat = cv2.cann.NpuMat()
npuMat.upload(img)

npuMatSum = cv2.cann.add(npuMat, npuMat)
imgResult = npuMatSum.download()
print(imgResult)

cv2.cann.finalizeAcl()
