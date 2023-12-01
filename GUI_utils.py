########################################
######      UTILS FOR PART 1      ######
########################################

import os
import csv
from ultralytics import YOLO
from PIL import Image


def detect_objects(img_path: str, img_name: str, output_path: str):
    # Load a model
    model = YOLO('Models/yolov8_best.pt')  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(source=img_path)

    i=0 # keep track of output
    for r in results:
        im_array = r.plot(labels=False)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

        annotated_img = f'{output_path}/{img_name.split(".")[0]}_annotated.jpg'
        im.save(annotated_img)

        annotations_file = f'{output_path}/{img_name.split(".")[0]}_annotations.csv'
        with open(annotations_file, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            boxes = results[i].boxes.xyxy.tolist()
            for box in boxes:
                writer.writerow(box)
                i+=1

    return True


def extract_objects(img_path: str, img_name: str, output_path: str):
    raw_img = Image.open(img_path)
    
    obj_output_path = f'{output_path}/{img_name.split(".")[0]}_objects'
    if not os.path.exists(obj_output_path):
        os.mkdir(obj_output_path)

    annotations_file = f'{output_path}/{img_name.split(".")[0]}_annotations.csv'
    with open(annotations_file, 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            obj = raw_img.crop((x1, y1, x2, y2))
            obj.save(f'{obj_output_path}/obj_{idx}.jpg')

    return True



########################################
######      UTILS FOR PART 2      ######
########################################

import umap
import shutil
import numpy as np
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications import vgg19


def extract_vgg19_features(img_dir):
    # Load Model
    model = vgg19.VGG19(weights='imagenet', include_top=False)

    # Feature Extraction
    features_dict = {}
    for img_name in os.listdir(img_dir):
        image_path = os.path.join(img_dir, img_name)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check to ensure only images are processed


            # Load and preprocess the image
            img = image.load_img(image_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = vgg19.preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            # Extract features from the desired layer (e.g., block5_pool)
            layer_name = 'block5_pool'  # You can choose a different layer
            intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
            features_dict[img_name] = intermediate_layer_model.predict(img)

    return features_dict


def dim_reduction_umap(features_dict):
    feature_list = [features for features in features_dict.values()]
    feature_matrix = np.array(feature_list).reshape(len(feature_list), -1) # Reshape to (n_samples, n_features) if necessary

    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(feature_matrix)

    reduced_features = {name: emb for name, emb in zip(features_dict.keys(), embedding)}

    return reduced_features


def clustering_AP(reduced_features, img_dir):
    # 'reduced_features' to list of feature coordinates for clustering
    X = np.array(list(reduced_features.values()))

    af = AffinityPropagation().fit(X)
    af_labels = af.labels_

    cluster_dir_base = os.path.join(img_dir, 'clusters')
    os.makedirs(cluster_dir_base, exist_ok=True)

    for filename in os.listdir(cluster_dir_base):
        file_path = os.path.join(cluster_dir_base, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    inventory_count = {}

    for img_name, cluster_label in zip(reduced_features.keys(), af_labels):
        if cluster_label == -1:
            # -1 for noise points
            cluster_path = os.path.join(cluster_dir_base, 'noise')
        else:
            cluster_path = os.path.join(cluster_dir_base, f'cluster_{cluster_label}')
        
        os.makedirs(cluster_path, exist_ok=True)
        
        source = os.path.join(img_dir, img_name)
        destination = os.path.join(cluster_path, img_name)
        shutil.copy(source, destination)
        
        inventory_count[cluster_label] = inventory_count.get(cluster_label, 0) + 1

    return inventory_count