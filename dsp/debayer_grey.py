# Python script to debayer an unknown format image

import cv2 as cv
import numpy as np
import os
BayerPatternMap = {"RGGB": cv.COLOR_BAYER_RG2RGB,
                  "BGGR": cv.COLOR_BAYER_BG2RGB,
                  "GRBG": cv.COLOR_BAYER_GR2RGB,
                  "GBRG": cv.COLOR_BAYER_GB2RGB,
                  "EGBRG": cv.COLOR_BAYER_GB2BGR_EA,
                  }

def debayer_raw(inFile, width=3856, height=2176, linesCaptured = 2201, inPattern = "GBRG"):
    # Read the raw MIPI-CSI2 streamed image but since we don't know the data-type
    # of pixels, read the file-size for hint
    bufferSize = os.path.getsize(inFile)
    pixelLineLength = bufferSize / linesCaptured
    if pixelLineLength.is_integer():
        bytesPerPixel = int(pixelLineLength / width)
        padding = pixelLineLength - (bytesPerPixel * width)
        if padding < 0:
            raise ValueError(f"Invalid padding calculated: {padding}")
        else:
            print(f"Input Buffer size: {bufferSize}, Pixel line length: {pixelLineLength}, Bytes per pixel: {bytesPerPixel}, Padding: {padding}")
    mipiBuffer = np.fromfile(inFile, dtype=np.uint16)
    imgFrame = mipiBuffer.reshape(linesCaptured, int(pixelLineLength / bytesPerPixel))
    imgMat = imgFrame[:height, :width]
    imgMat = (imgMat << 8) | (imgMat >> 8)
    #imgMat = imgMat << 4
    # Convert to float for normalization
    #imgMatFloat = imgMat.astype(np.float32)
    # Save as gray scale image
    debayered = cv.cvtColor(imgMat, BayerPatternMap["EGBRG"])
    grayImg = cv.cvtColor(debayered, cv.COLOR_BGR2GRAY)
    normalizedImg = cv.normalize(grayImg, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    return normalizedImg

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python debayer.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    output_image = debayer_raw(image_path)
    cv.imwrite("debayered_gray.png", output_image)