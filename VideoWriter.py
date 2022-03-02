import cv2
import numpy as np
import glob

img_array = []
i=0
while i<914:
    img=cv2.imread("Images for MDR/Processed/Processed Image " + str(i) + '.jpg')
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    i+=1
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('ProcessedVideoFinal.avi', fourcc, 60, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()