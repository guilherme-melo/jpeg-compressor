from typing import Counter, final
import cv2 #opencv
import numpy as np
import scipy.fftpack as fftpack
import os
from matplotlib import pyplot as plt
import time

prev_16 = lambda x: x >> 4 << 4

def encode_list_py(zigzag_vector_list):
    """Encodes the message from the zigzag vector"""
    full_list_1 = []
    full_list_2 = []
    full_list_3 = []    
    for zigzag_vector_1 in zigzag_vector_list:
        for zigzag_vector_2 in zigzag_vector_1:
            for zigzag_vector in zigzag_vector_2:
                encoded_list = []
                i = 0
                while (i <= len(zigzag_vector)-1):
                    count = 1
                    ch = zigzag_vector[i]
                    j = i
                    while (j < len(zigzag_vector)-1): 
                        if (zigzag_vector[j] == zigzag_vector[j + 1]): 
                            count = count + 1
                            j = j + 1
                        else: 
                            break
                    encoded_list.append([count, ch])
                    i = j + 1
                full_list_1.append(encoded_list)   
            full_list_2.append(full_list_1)
            full_list_1 = []
        full_list_3.append(full_list_2)
        full_list_2 = []
    return full_list_3

def image_to_zigzag(matrix_b4):
    """Transforms the image into a zigzag vector"""
    list_final0 = []
    list_final1 = []
    list_final2 = []
    matrix_t = matrix_b4.transpose(0,2,4,1,3)
    count=0
    for element in matrix_t:
        for element_2 in element:
            for matrix in element_2:
                sp = matrix.shape
                list_enc = [0]*((sp[0]*sp[1])-1)
                length = sp[0]
                i,j=0,0
                case = 0
                count = 1
                count_ward = 1
                mid_ward = False 
                for l in range(len(list_enc)):
                    if not mid_ward:
                        if count == length:
                            count -=2
                            if length % 2 == 0:
                                case = 6
                            else:
                                case = 4
                            count_ward = count
                            mid_ward = True
                        if case == 0:
                            j+=1
                            case += 1
                        elif case == 1:
                            if count_ward != 0:
                                i+=1
                                j-=1
                                count_ward -= 1
                            if count_ward == 0:
                                case += 1
                                count += 1
                                count_ward = count
                        elif case == 2:
                            i+=1
                            case += 1
                        elif case == 3:
                            if count_ward != 0:
                                i-=1
                                j+=1
                                count_ward -= 1
                            if count_ward == 0:
                                case = 0
                                count += 1
                                count_ward = count

                        if not mid_ward: 
                            list_enc[l] = matrix[i][j]
                    if mid_ward:
                        if case == 4:
                            i+=1
                            case += 1
                        elif case == 5:
                            if count_ward != 0:
                                i+=1
                                j-=1
                                count_ward -= 1
                            if count_ward == 0:
                                case += 1
                                count -= 1
                                count_ward = count
                        elif case == 6:
                            j+=1
                            case += 1
                        elif case == 7:
                            if count_ward != 0:
                                i-=1
                                j+=1
                                count_ward -= 1
                            if count_ward == 0:
                                case = 4
                                count -= 1
                                count_ward = count
                        list_enc[l] = matrix[i][j]
                list_enc.insert(0, matrix[0][0])
                list_final0.append(list_enc)
                list_enc = []
            list_final1.append(list_final0)
            list_final0 = []
        list_final2.append(list_final1)
        list_final1 = []
            
    return list_final2


def rgb_to_yuv(img):
    """OpenCV u and v Channels are bugged so this function was adapted from: https://stackoverflow.com/questions/43983265/rgb-to-yuv-conversion-and-accessing-y-u-and-v-channels
    The function receives the image as a parameter and returns it splitted in it's proper y, u and v channels """   
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, u, v = cv2.split(img_yuv)
    return y,u,v

def yuv_to_rgb(y,u,v):
    """OpenCV u and v Channels are bugged so this function was adapted from: https://stackoverflow.com/questions/43983265/rgb-to-yuv-conversion-and-accessing-y-u-and-v-channels
    The function receives the image as a parameter and returns it splitted in it's proper y, u and v channels """   
    image = cv2.merge((y,u,v))
    img_bgr = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
    return img_bgr


def encode_dct(orig, bx, by):
    """Encodes the image using the DCT and returns the encoded image"""
    #orig -= 128
    new_shape = (
        orig.shape[0] // bx * bx,
        orig.shape[1] // by * by,
        3
    )
    new = orig[
        :new_shape[0],
        :new_shape[1]
    ].reshape((
        new_shape[0] // bx,
        bx,
        new_shape[1] // by,
        by,
        3
    ))
    return fftpack.dctn(new, axes=[1,3], norm='ortho')


def quantization(image):
    """Quantizes the image using the quantization matrix"""
    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],[12, 12, 14, 19, 26, 58, 60, 55],[14, 13, 16, 24, 40, 57, 69, 56],[14, 17, 22, 29, 51, 87, 80, 62],[18, 22, 37, 56, 68, 109, 103, 77],[24, 36, 55, 64, 81, 104, 113, 92],[49, 64, 78, 87, 103, 121, 120, 101],[72, 92, 95, 98, 112, 100, 103, 99]])
    #quantization_matrix = np.array([[16,12,14,14,18,24,49,72],[11,12,13,17,22,35,64,92],[10,14,16,22,37,55,78,95],[16,19,24,29,56,64,87,98],[24,26,40,51,68,81,103,112],[40,58,57,87,109,104,121,100],[51,60,69,80,103,113,120,103],[61,55,56,62,77,92,101,99]])
    quantized = np.round(image / quantization_matrix[np.newaxis,:, np.newaxis,:,np.newaxis])
    image
    return quantized


def decode_list_py(encoded_list_list):
    final_list = []
    final_list_2 = []
    final_list_3 = []
    for encoded_list_1 in encoded_list_list:
        for encoded_list_2 in encoded_list_1:
            for encoded_list in encoded_list_2:
                decoded_list = []
                i = 0
                j = 0
                while (i <= len(encoded_list) - 1):
                    run_count = int(encoded_list[i][0])
                    run_word = encoded_list[i][1]
                    for j in range(run_count):
                        decoded_list.append(run_word)
                        j = j + 1
                    i = i + 1
                final_list.append(decoded_list)
            final_list_2.append(final_list)
            final_list = []
        final_list_3.append(final_list_2)
        final_list_2 = []

    return final_list_3

def zigzag_to_image(list_encoded_b4):
    """Transforms the zigzag vector into an image"""
    matrix_final_1 = []
    matrix_final_2 = []
    matrix_final_3 = []
    for list_encoded_1 in list_encoded_b4:
        for list_encoded_2 in list_encoded_1:
            for list_encoded in list_encoded_2:
                i,j=0,0
                length = int(len(list_encoded)**0.5)
                matrix = np.zeros((length,length))
                case = 0
                count = 1
                count_ward = 1
                mid_ward = False
                matrix[0][0] = list_encoded[0]
                list_encoded.remove(list_encoded[0])
                for k in list_encoded:
                    if not mid_ward:
                        if count == length:            
                            if length % 2 == 0:
                                case = 6
                            else:
                                case = 4
                            count -=2
                            count_ward = count
                            mid_ward = True
                        if case == 0:
                            j+=1
                            case += 1
                        elif case == 1:
                            if count_ward != 0:
                                i+=1
                                j-=1
                                count_ward -= 1
                            if count_ward == 0:
                                case += 1
                                count += 1
                                count_ward = count
                        elif case == 2:
                            i+=1
                            case += 1
                        elif case == 3:
                            if count_ward != 0:
                                i-=1
                                j+=1
                                count_ward -= 1
                            if count_ward == 0:
                                case = 0
                                count += 1
                                count_ward = count

                        if not mid_ward: 
                            matrix[i][j]=int(k)
                    if mid_ward:
                        if case == 4:
                            i+=1
                            case += 1
                        elif case == 5:
                            if count_ward != 0:
                                i+=1
                                j-=1
                                count_ward -= 1
                            if count_ward == 0:
                                case += 1
                                count -= 1
                                count_ward = count
                        elif case == 6:
                            j+=1
                            case += 1
                        elif case == 7:
                            if count_ward != 0:
                                i-=1
                                j+=1
                                count_ward -= 1
                            if count_ward == 0:
                                case = 4
                                count -= 1
                                count_ward = count
                        matrix[i][j]=int(k)
                matrix_final_1.append(matrix)
            matrix_final_2.append(matrix_final_1)
            matrix_final_1 = []
        matrix_final_3.append(matrix_final_2)
        matrix_final_2 = []
    matrix_final_3 = np.array(matrix_final_3)
    matrix_final = matrix_final_3.transpose(0,3,1,4,2)
    return matrix_final

def decode_dct(orig, bx, by):
    dec = fftpack.idctn(orig, axes=[1,3], norm='ortho'
    ).reshape((
        orig.shape[0]*bx,
        orig.shape[2]*by,
        3
    )) 
    #dec += 128
    return dec

def dequantization(image):
    """Dequantizes the image using the quantization matrix"""
    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],[12, 12, 14, 19, 26, 58, 60, 55],[14, 13, 16, 24, 40, 57, 69, 56],[14, 17, 22, 29, 51, 87, 80, 62],[18, 22, 37, 56, 68, 109, 103, 77],[24, 36, 55, 64, 81, 104, 113, 92],[49, 64, 78, 87, 103, 121, 120, 101],[72, 92, 95, 98, 112, 100, 103, 99]])
    #dequantization_matrix = np.array([[16,12,14,14,18,24,49,72],[11,12,13,17,22,35,64,92],[10,14,16,22,37,55,78,95],[16,19,24,29,56,64,87,98],[24,26,40,51,68,81,103,112],[40,58,57,87,109,104,121,100],[51,60,69,80,103,113,120,103],[61,55,56,62,77,92,101,99]])

    dequantized = image * quantization_matrix[np.newaxis,:, np.newaxis,:,np.newaxis]

    return dequantized
    
"""The following code was inspired by the https://github.com/changhsinlee/software-for-science/tree/master/2019-04-11-jpeg-algorithm jpeg compressor"""

"""File-Encoder"""

if __name__ == '__main__':
    files = os.listdir("images")
    compressed_files = []
    for file in files:
        name=file.split(".")
        image = cv2.imread("images/"+file)
        y,u,v= rgb_to_yuv(image)
        u=np.delete(u,list(range(0,u.shape[0],2)),axis=0)
        v=np.delete(v,list(range(0,v.shape[0],2)),axis=0)   
        u=np.repeat(u, 2, axis=0)
        v=np.repeat(v, 2, axis=0)
        im = yuv_to_rgb(y,u,v)
        im_encoded = encode_dct(im,8,8)
        im_quantized = quantization(im_encoded)
        zigzag = image_to_zigzag(im_quantized)
        compressed = encode_list_py(zigzag) 
        compressed_files.append([compressed,name])
"""File-Decoder"""

if __name__ == '__main__':
    for file in compressed_files:  
        #orig_size = file[0].shape
        decoded_list = decode_list_py(file[0])
        im_decoded = zigzag_to_image(decoded_list)
        image = dequantization(im_decoded)
        image = decode_dct(image,8,8)
        cv2.imwrite(file[1][0]+"_compressed.png", image)

