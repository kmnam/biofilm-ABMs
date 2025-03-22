"""
Stitch together a sequence of frames (.jpg) into a video (.avi). 

Authors:
    Kee-Myoung Nam

Last updated:
    3/22/2025
"""

import sys
import numpy as np
from PIL import Image
import cv2

#########################################################################
# Stitch the images together and export as an .avi file
filenames = sys.argv[1:]
outfilename = filenames[0][:-9] + '.avi'
image_filenames = [filename[:-4] + '_frame.jpg' for filename in filenames]
width = None
height = None
with Image.open(image_filenames[0]) as image:
    width, height = image.size
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
fps = 20
video = cv2.VideoWriter(
    outfilename, fourcc, fps, (width, height), isColor=True
)
for image_filename in image_filenames:
    with Image.open(image_filename) as image:
        video.write(np.array(image)[:, :, ::-1])    # Switch from RGB to BGR

