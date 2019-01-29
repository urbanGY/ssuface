import numpy as np
import tensorflow as tf
import cv2 as cv

def _read_list_image(input):
    return [cv.imread(input[x]) for x in range(1847)]

image_list = []
label_list = []
for i in range(1,1848):
    name = 'C:/Users/sfsfk/Desktop/develope/tensorflow/ssuface/data/pencilCase/test'
    name = name + str(i) + '.jpg'
    image_list.append(name)
    label_list.append(1)
    
img = np.stack(_read_list_image(image_list))
