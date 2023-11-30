from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

def extract_features_from_array(img_array, model):
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    features = model.predict(img_preprocessed)
    return features.squeeze()

def extract_features_for_folder(folder_path, model):
    feature_list = []
    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.jpg'): 
            file_path = os.path.join(folder_path, file_name)
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            features = extract_features_from_array(img_array, model)
            feature_list.append(features)
    return np.array(feature_list)

def calculate_centroids(features, cluster_labels):
    centroids = []
    for label in np.unique(cluster_labels):
        members = features[cluster_labels == label]
        centroid = np.mean(members, axis=0)
        centroids.append(centroid)
    return np.array(centroids)

folder_path = '/Users/kaavi/Documents/GitHub/term7cv/dataset/objects/train_8'

features = extract_features_for_folder(folder_path, model)
print("Feature extraction complete. Number of images processed:", len(features))

cluster_labels = np.random.randint(0, 10, len(features))  # Mock labels

centroids = calculate_centroids(features, cluster_labels)

print("Calculated centroids for each cluster:")
for i, centroid in enumerate(centroids):
    print(f"Centroid {i}:")
    print(centroid)
