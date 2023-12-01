import os
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

images_dir = '/Users/kaavi/Documents/GitHub/term7cv/dataset/SKU110K_fixed/images'
annotations_dir = '/Users/kaavi/Documents/GitHub/term7cv/dataset/SKU110K_fixed/annotations'
headers = ['img_name','x1','y1','x2','y2','class','img_w','img_h']

data_set = 'train'
annotation_set_df = pd.read_csv(annotations_dir + f'/annotations_{data_set}.csv', names=headers)

output_path = 'dataset/objects'

for img_name in os.listdir(images_dir):
    if data_set not in img_name:
        continue

    img_path = images_dir + '/' + img_name
    annotation_img = annotation_set_df[annotation_set_df.iloc[:, 0] == img_name]

    img = Image.open(img_path)
    output_folder = output_path + '/'+ img_name.split(".")[0]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for idx, row in annotation_img.iterrows():
        x1, y1, x2, y2 = row[1], row[2], row[3], row[4]

        obj = img.crop((x1, y1, x2, y2))
        obj.save(output_folder + f'/obj_{idx}.jpg')

