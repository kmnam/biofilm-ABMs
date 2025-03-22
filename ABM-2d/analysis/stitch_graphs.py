"""
Stitch together a sequence of graph plots (.jpg) into a video (.avi). 

Authors:
    Kee-Myoung Nam

Last updated:
    3/22/2025
"""

import sys
import os
import numpy as np
from PIL import Image
import cv2

#########################################################################
# Stitch the images together and export as an .avi file
filenames = sys.argv[1:]
outfilename = '_'.join(filenames[0][:-4].split('_')[:-1]) + '_graph.avi'
image_filenames = []
for filename in filenames:
    image_filenames.append(
        os.path.join(
            'graphs', os.path.basename(os.path.dirname(filename)),
            os.path.basename(filename[:-4]) + '_graph_plot.jpg'
        )
    )
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

