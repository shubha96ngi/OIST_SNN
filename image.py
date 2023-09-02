import numpy as np
import cv2 
gray_imgs = cv2.imread('gray_0.png')
#gray_imgs = cv2.cvtColor(gray_imgs, cv2.COLOR_BGR2GRAY)

import torch
from skimage import img_as_float
im1=[]
for i in range(1):
   # im= cv2.cvtColor(gray_imgs, cv2.COLOR_BGR2GRAY)
    im1.append(np.array(img_as_float(gray_imgs)))


img1 = np.array([im1])
imd = torch.from_numpy(img1) #[:,:,:,1])
imd.shape
# perform automatic thresholding
import skimage
blurred_image = skimage.filters.gaussian(gray_imgs, sigma=1.0)
t = skimage.filters.threshold_otsu(blurred_image)

from snntorch import spikegen
spike_data = spikegen.latency(imd, num_steps=70, tau=5, threshold=0.1,clip=True)

import tensorflow as tf 
pil_img = tf.keras.preprocessing.image.array_to_img(np.squeeze(spike_data[1]))
plt.imshow(pil_img)

import snntorch.spikeplot as splt 
num_steps=70
import matplotlib.pyplot as plt
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
#splt.raster(spike_data[:,0].reshape(num_steps,-1)[:,15].unsqueeze(1), ax, s=25, c="black")
splt.raster(np.squeeze(spike_data[1]),ax,c='black')
plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
#plt.imshow('gray_imgs)
plt.show()
