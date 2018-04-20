#!/usr/bin/env python3

from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import pylab
import scipy
import sys
from scipy import ndimage
from scipy import misc

image = misc.imread(sys.argv[1], flatten=True)
image = misc.imresize(image, (200, 450))

sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
edge_detect = np.array([[1,0,-1],[0,0,0],[-1,0,1]])
gaussian_blur = (1/16) * np.array([[1,2,1],[2,4,2],[1,2,1]])

sharpened = ndimage.convolve(image, sharpen)
edges = ndimage.convolve(image, edge_detect)
blurred = ndimage.convolve(image, gaussian_blur)

misc.imsave('originial.png', image)
misc.imsave('sharpened.png', sharpened)
misc.imsave('edges.png', edges)
misc.imsave('blurred.png', blurred)
