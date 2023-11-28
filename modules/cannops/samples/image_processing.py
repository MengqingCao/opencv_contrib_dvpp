# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description='This is a sample for image processing with Ascend NPU.')
parser.add_argument('image', help='path to input image')
parser.add_argument('output', help='path to output image')
args = parser.parse_args()

# read input image and generate guass noise
#! [input_noise]
img = cv2.imread(args.image, cv2.IMREAD_COLOR)
# Generate gauss noise that will be added into the input image
gaussNoise = np.random.normal(0, 25,(img.shape[0], img.shape[1], img.shape[2])).astype(img.dtype)
#! [input_noise]

# setup cann
#! [setup]
cv2.cann.initAcl()
cv2.cann.setDevice(0)
cv2.cann.initDvpp()
#! [setup]

#! [image-process]
# add gauss noise to the image
output = cv2.cann.add(img, gaussNoise)
# rotate the image with a certain mode (0, 1 and 2, correspond to rotation of 90, 180
# and 270 degrees clockwise respectively)
output = cv2.cann.rotate(output, 0)
# flip the image with a certain mode (0, positive and negative number, correspond to flipping
# around the x-axis, y-axis and both axes respectively)
output = cv2.cann.flip(output, 0)

w_off, h_off, crop_w, crop_h = 250, 250, 512, 512
roi = [w_off, h_off, crop_w, crop_h]
dstSize = np.array([256, 256])
scalarV = np.array([230.0, 10.0, 10.0]).astype(np.double)
# scalarV = [30.0, 50.0, 200.0]

# output1 = cv2.cann.CropResizeMakeBorder(img, roi, dstSize, 0, 0, 0, 0, scalarV, 128, 64)
output1 = cv2.cann.CropResize(img, roi, dstSize, 0, 0, 0, 0, scalarV)

# print(output1)
output = img[w_off: w_off+crop_w, h_off: h_off+crop_h]
# # cv.resize(npMat.astype(np.float32), dstSize, 0, 0, 3)
output = cv2.resize(output, dstSize, 0, 0, 1)

# #! [image-process]

cv2.imwrite(args.output, output1)

#! [tear-down-cann]
cv2.cann.finalizeDvpp()
cv2.cann.finalizeAcl()
#! [tear-down-cann]
