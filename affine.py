import cv2
import numpy as np

img=cv2.imread("./119.jpg")
shape=img.shape
height=shape[0]
width=shape[1]
print(shape)
cv2.imshow("src",img)

#定义mat_src
mat_src=np.float32([[150,150],[height-100,0],[0,width-100]])
mat_dst=np.float32([[0,0],[height-1,0],[0,width-1]])
affine_mat=cv2.getAffineTransform(mat_src,mat_dst)
dst=cv2.warpAffine(img,affine_mat,(width,height))
print(dst.shape)
cv2.imshow("dst",dst)
cv2.waitKey(0)
