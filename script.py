import cv2 #opencv
import numpy as np
import scipy.fftpack as fftpack
import os

def rgb_to_yuv(img):
    """OpenCV u and v Channels are bugged so this function was adapted from: https://stackoverflow.com/questions/43983265/rgb-to-yuv-conversion-and-accessing-y-u-and-v-channels
    The function receives the image as a parameter and returns it splitted in it's proper y, u and v channels """   
    lut_u = np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)
    lut_v = np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    u = cv2.LUT(u, lut_u)
    v = cv2.LUT(v, lut_v)
    return y,u,v



"""The following code was inspired by the https://github.com/changhsinlee/software-for-science/tree/master/2019-04-11-jpeg-algorithm jpeg compressor"""
if __name__ == '__main__':
    files = os.listdir("images")
    for file in files:
        name=file.split(".")
        image = cv2.imread("images/"+file)
        y,u,v= rgb_to_yuv(image)
        cv2.imwrite('y.png', y)
        cv2.imwrite("u.png", u)
        cv2.imwrite("v.png", v)                
        cv2.imwrite(name[0]+"_compressed."+name[1], image)
