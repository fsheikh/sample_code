# Python script to debayer an unknown format image

import cv2 as cv
import numpy as np
import os

# Dictionary to cycle through typical OpenCV Bayer patterns
BayerPatternMap = {"RGGB": cv.COLOR_BAYER_RG2BGR,
                  "BGGR": cv.COLOR_BAYER_BG2BGR,
                  "GRBG": cv.COLOR_BAYER_GR2BGR,
                  "GBRG": cv.COLOR_BAYER_GB2BGR,
                  "EGBRG": cv.COLOR_BAYER_GB2BGR_EA
                  }

def unpack(inFile, width=3856, height=2176, linesCaptured = 2201):
    # Read the raw MIPI-CSI2 streamed image but since we don't know the data-type
    # of pixels, read the file-size for hint and coorrelate with expected resolution
    bufferSize = os.path.getsize(inFile)
    bufferStride = bufferSize / linesCaptured
    if bufferStride.is_integer():
        bytesPerPixel = int(bufferStride / width)
        padding = bufferStride - (bytesPerPixel * width)
        if padding < 0:
            raise ValueError(f"Invalid padding calculated: {padding}")
        else:
            print(f"Buffer size: {bufferSize}, Stride: {bufferStride}, BPP: {bytesPerPixel}, Padding: {padding}")

    # This function expects 2 bytes per pixel (16-bit images)
    # since raw data buffer is way too large for single-byte pixels
    if bytesPerPixel != 2:
        raise ValueError(f"Unsupported bytes per pixel: {bytesPerPixel}")
    
    mipiBuffer = np.fromfile(inFile, dtype=np.uint16)
    imgFrame = mipiBuffer.reshape(linesCaptured, int(bufferStride / bytesPerPixel))
    imgMat = imgFrame[:height, :width]
    # It looks like LSB byte arrives first on the wire(little-endian format)
    unpackedMat = (imgMat << 8) | (imgMat >> 8)
    return unpackedMat

def debayer(unpackedMat, inPattern = "EGBRG"):
    filteredImg = cv.cvtColor(unpackedMat, BayerPatternMap[inPattern])
    debayeredImg = cv.normalize(filteredImg, None, alpha=0, beta=2047, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

    return debayeredImg

def white_balance(debayeredImg, RGain=2.1, GGain=1.0, BGain=1.2):
    b,g,r = cv.split(debayeredImg)

    rBalanced = cv.convertScaleAbs(r, alpha=RGain, beta=0)
    gBalanced = cv.convertScaleAbs(g, alpha=GGain, beta=0)
    bBalanced = cv.convertScaleAbs(b, alpha=BGain, beta=0)
    balancedImg = cv.merge([bBalanced, gBalanced, rBalanced])
    return balancedImg

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python debayer.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print("Starting ISP pipeline unpack-->debayer-->white balance ...")
    unpackedImg = unpack(image_path)
    debayeredImg = debayer(unpackedImg, inPattern="EGBRG")
    balancedImg = white_balance(debayeredImg)

    print("Writing output images...")
    cv.imwrite("debayered_output.png", balancedImg)

    # Optional: Generate a gray image
    grayImg = cv.cvtColor(debayeredImg, cv.COLOR_BGR2GRAY)
    cv.imwrite("debayered_gray.png", grayImg)
    print("Done! Check results visually!")