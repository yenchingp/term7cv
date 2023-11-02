import os

data_set = 'test'

for img in os.listdir('dataset/SKU110K_fixed/images'):
    if data_set in img:
        num = img.split("_")[1].split(".")[0]
        print(num)