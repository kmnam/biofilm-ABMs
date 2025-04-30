"""
Stitch together a sequence of frames (.jpg) into a video (.avi). 

Authors:
    Kee-Myoung Nam

Last updated:
    4/29/2025
"""

import sys
import numpy as np
from PIL import Image
import cv2

#########################################################################
suffix = sys.argv[1]
filenames = sys.argv[2:-1]
outfilename = sys.argv[-1]
if suffix == 'None':
    image_filenames = filenames
else:
    image_filenames = [
        filename[:-4] + '_{}.jpg'.format(suffix) for filename in filenames
    ]

# Stitch the images together and export as an .avi file
#
# First get the dimensions of the first image (assuming they are the same
# for all frames)
width = None
height = None
with Image.open(image_filenames[0]) as image:
    width, height = image.size

# Stitch the images together 
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
fps = 20
video = cv2.VideoWriter(
    outfilename, fourcc, fps, (width, height), isColor=True
)
for image_filename in image_filenames:
    with Image.open(image_filename) as image:
        video.write(np.array(image)[:, :, ::-1])    # Switch from RGB to BGR

