# Python script to debayer an unknown format image

import cv2 as cv
import numpy as np
import os
BayerPatternMap = {"RGGB": cv.COLOR_BAYER_RG2BGR,
                  "BGGR": cv.COLOR_BAYER_BG2BGR,
                  "GRBG": cv.COLOR_BAYER_GR2BGR,
                  "GBRG": cv.COLOR_BAYER_GB2BGR}

def debayer_raw(inFile, width=3856, height=2176, linesCaptured = 2201, inPattern = "RGGB"):
    # Read the raw MIPI-CSI2 streamed image but since we don't know the data-type
    # of pixel, read the full buffer size and tinker with 10 or 12-bit MIPI-CSI2 packed formats
    mipiBuffer = np.fromfile(inFile, dtype=np.uint8)
    bufferStride = len(mipiBuffer) / linesCaptured
    if bufferStride > width:
        print(f"Packed Pixel format detected buffer stride: {bufferStride} given width: {width}")
    else:
        raise ValueError(f"Insufficient input buffer stride: {bufferStride}")
    
    imgFrame = mipiBuffer.reshape(linesCaptured, int(bufferStride))
    # For 12 bit MIPI packed
    pixelLineLength = width * 1.5
    if pixelLineLength > bufferStride:
        raise ValueError(f"Insufficient buffer stride for 12-bit packed format: {bufferStride} required: {pixelLineLength}")
    imgMat = imgFrame[:height, :int(pixelLineLength)]
    b0 = imgMat[:, 0::3].astype(np.uint16)
    b1 = imgMat[:, 1::3].astype(np.uint16)
    b2 = imgMat[:, 2::3].astype(np.uint16)
    evenPixels = (b0 << 4) | (b1 & 0x0F)
    oddPixels = (b2 << 4) | (b1 >> 4)
    retImg = np.empty((height, width), dtype=np.uint16)
    retImg[:, 0::2] = evenPixels
    retImg[:, 1::2] = oddPixels
    retImg = (retImg << 8) | (retImg >> 8)
    retImg8 = (retImg << 4).astype(np.uint8)
    # Debayer the image
    debayered_image = cv.cvtColor(retImg8, BayerPatternMap[inPattern])
    return debayered_image

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python debayer.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    output_image = debayer_raw(image_path)
    cv.imwrite("debayered_packed12_output.png", output_image)