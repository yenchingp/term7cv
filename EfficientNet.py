from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

def extract_features_from_array(img_array, model):
    # Preprocess the image for the model's expected input
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    # Extract and return features
    features = model.predict(img_preprocessed)
    return features.squeeze()

def extract_features_for_folder(folder_path, model):
    feature_list = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.jpg'):  # Check if the file is a JPEG image
            file_path = os.path.join(folder_path, file_name)
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            features = extract_features_from_array(img_array, model)
            feature_list.append(features)
    return np.array(feature_list)

folder_path = '/Users/kaavi/Documents/GitHub/term7cv/dataset/objects/train_8'

features = extract_features_for_folder(folder_path, model)

print("Feature extraction complete. Number of images processed:", len(features))
