# Inventory Management CV Project SUTD

## 50.035 CV 2023 Group 14

This is used for SUTD 50.035 Computer Vision project, inventory management using SKU-110K dataset(dense object detection)

All the final code is in "GUI.py" and "GUI_utils.py". Code in other folders are from our testing and evaluation phase, and are provided for reference only.
Some test images are provided in "GUI_image_test/raw_img", with some of their respective outputs in "GUI_image_test/annotated_img".

To use, first make sure all necessary packages are installed from "requirements.txt".
Once done, simply run "GUI.py". Note that an internet connection is needed to download model weights.

For bandwidth value, a smaller value leads to more clusters being estimated. Values of 0.2 - 0.8 work well for most of the images in our dataset.

Google drive link for full inventory detection materials: https://drive.google.com/drive/folders/1LlX6jSBj3hqiEa7b0CXNkRZVwMz3Z4r9?usp=drive_link
