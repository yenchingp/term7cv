import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


img_dir = "/home/zach/PycharmProjects/term7cv/dataset/objects/train_4"
img_dim = (128, 128)


def image_comparison_GS(input_dir: str, img_dim: tuple):
    error_list = {}
    
    img_file_list = os.listdir(input_dir)
    ref_img_name = img_file_list[0]

    ref_img = cv2.imread(input_dir + "/" + ref_img_name)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_img = cv2.resize(ref_img, img_dim, interpolation=cv2.INTER_AREA)
    
    for comp_img_name in img_file_list:
        comp_img = cv2.imread(input_dir + "/" + comp_img_name)
        comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2GRAY)
        comp_img = cv2.resize(comp_img, img_dim, interpolation=cv2.INTER_AREA)

        error_list[comp_img_name] = calc_error(ref_img, comp_img, img_dim)
    
    return error_list

def calc_error(img1, img2, img_dim):
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err/(float(img_dim[0] * img_dim[1]))
    msre = np.sqrt(mse)

    return mse


error_list = image_comparison_GS(img_dir, img_dim)

for k, v in error_list.items():
    if v < 20:
        print(k, v)
