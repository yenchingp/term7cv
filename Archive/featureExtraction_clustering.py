from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
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

# Calculate centroids for given cluster labels
def calculate_centroids(features, cluster_labels):
    centroids = []
    for label in np.unique(cluster_labels):
        members = features[cluster_labels == label]
        centroid = np.mean(members, axis=0)
        centroids.append(centroid)
    return np.array(centroids)

def merge_clusters(global_centroids, batch_centroids, features, cluster_labels, threshold=0.5):
    batch_centroids = {}
    if len(global_centroids) == 0:  
        global_centroids = batch_centroids.copy()
        return global_centroids, cluster_labels, features

    # mapping -> batch cluster indices to global cluster indices
    new_global_label_mapping = {}
    for batch_cluster_id, batch_centroid in batch_centroids.items():
        if batch_cluster_id == -1:
            continue

        # Compute distance -> batch centroid to global centroids
        distances = cdist([batch_centroid], list(global_centroids.values()), metric='euclidean').flatten()
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]

        # If closest global centroid -> within the threshold -> map to the same global cluster
        if min_distance < threshold:
            global_cluster_id = list(global_centroids.keys())[min_distance_index]
            new_global_label_mapping[batch_cluster_id] = global_cluster_id
        else:
            # If not -> new global cluster ID
            new_global_cluster_id = max(global_centroids.keys()) + 1
            global_centroids[new_global_cluster_id] = batch_centroid
            new_global_label_mapping[batch_cluster_id] = new_global_cluster_id

    # Re-assign cluster labels for features based on mapping
    new_cluster_labels = np.array([
        new_global_label_mapping[label] if label in new_global_label_mapping else label
        for label in cluster_labels
    ])

    # Recalculate global centroids after merge to include new batch features
    for new_global_id, global_centroid in global_centroids.items():
        # Find the batch indices that have been mapped to the current global ID
        assigned_batch_indices = np.where(new_cluster_labels == new_global_id)[0]
        if len(assigned_batch_indices) > 0:
            # Update centroid with new features that belong to this global cluster
            assigned_features = features[assigned_batch_indices]
            updated_centroid = np.mean(np.vstack((global_centroid, assigned_features)), axis=0)
            global_centroids[new_global_id] = updated_centroid

    return global_centroids, new_cluster_labels, features

def process_batches(folder_path, model, eps_value, min_samples_value, distance_threshold):
    global_centroids = {}
    all_features = []
    all_cluster_labels = []
    
    batch_folders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('train_') and os.path.isdir(os.path.join(folder_path, f))]
    print(batch_folders)
    
    for batch_index, batch_folder in enumerate(batch_folders):
        batch_files = [os.path.join(batch_folder, f) for f in os.listdir(batch_folder) if f.lower().endswith('.jpg')]
        print(batch_files)
        
        if not batch_files:
            continue
        
        features = extract_features_for_folder(batch_folder, model)
        all_features.extend(features)
        
        # DBSCAN + merge clusters
        if len(features) > 0:
            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value, metric='euclidean')
            batch_cluster_labels = dbscan.fit_predict(features)
            batch_centroids = calculate_centroids(features, batch_cluster_labels)
            
            global_centroids, new_global_labels, _ = merge_clusters(global_centroids, batch_centroids, features, batch_cluster_labels, threshold=distance_threshold)
            
            # Update global cluster labels with the new labels for this batch
            all_cluster_labels.extend(new_global_labels + max(all_cluster_labels + [-1]) + 1) 
        else:
            all_cluster_labels.extend([-1] * len(features))
            
    return global_centroids, all_cluster_labels, all_features

folder_path = '/Users/kaavi/Documents/GitHub/term7cv/dataset/objects' 
eps_value = 0.5  # DBSCAN eps parameter value
min_samples_value = 5  # DBSCAN min_samples parameter value
distance_threshold = 0.5  # Distance threshold for considering two centroids to be in the same cluster

global_centroids, global_cluster_labels, global_features = process_batches(folder_path, model, eps_value, min_samples_value, distance_threshold)

# 'global_centroids' contains the merged centroids from all batches
# 'global_cluster_labels' contains the global labels for each feature in all batches
# 'global_features' is a list of all features extracted from all batches

def manage_inventory(global_cluster_labels):
    unique, counts = np.unique(global_cluster_labels, return_counts=True)
    
    inventory = {obj_type: count for obj_type, count in zip(unique, counts) if obj_type != -1}
    return inventory

inventory = manage_inventory(global_cluster_labels)
print(global_cluster_labels)

print("Inventory of object instances:")
for object_id, count in inventory.items():
    print(f"Object ID {object_id}: Count {count}")