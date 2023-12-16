import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import mobilenet, resnet, densenet, efficientnet, vgg19
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth, AffinityPropagation
import hdbscan
from sklearn import metrics
import umap


def load_and_preprocess_image(img_path, preprocess_input_func, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img = preprocess_input_func(img)
    return np.expand_dims(img, axis=0)


def create_feature_model(model_fn, preprocess_input_func):
    base_model = model_fn(weights='imagenet', include_top=False)
    feature_model = Model(inputs=base_model.input, outputs=base_model.output)
    return feature_model, preprocess_input_func


def extract_features_from_folder(folder_path, model_fn, preprocess_input_func):
    feature_model, preprocess_input = create_feature_model(model_fn, preprocess_input_func)
    features = []
    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, img_name)
            img = load_and_preprocess_image(img_path, preprocess_input)
            feature = feature_model.predict(img)
            features.append(feature.flatten())
    return np.array(features)


models = {
    'mobilenet': (mobilenet.MobileNet, mobilenet.preprocess_input),
    'resnet': (resnet.ResNet50, resnet.preprocess_input),
    'densenet': (densenet.DenseNet121, densenet.preprocess_input),
    'efficientnet': (efficientnet.EfficientNetB0, efficientnet.preprocess_input),
    'vgg19': (vgg19.VGG19, vgg19.preprocess_input),
}


for i in range(100):
    print(i)

    if i in [40]:
        continue

    img_dir = f'dataset/objects/train_{i}'

    all_features = {}
    for model_name, (model_fn, preprocess_input_func) in models.items():
        all_features[model_name] = extract_features_from_folder(img_dir, model_fn, preprocess_input_func)


    reduced_features = {}
    for model_name, features in all_features.items():
        features = StandardScaler().fit_transform(features)
        
        pca = PCA(n_components=0.9)
        reduced_features[(model_name, 'pca')] = pca.fit_transform(features)

        tsne = TSNE(n_components=2)
        reduced_features[(model_name, 'tsne')] = tsne.fit_transform(features)

        umap_reducer = umap.UMAP(n_neighbors=15, n_components=2)
        reduced_features[(model_name, 'umap')] = umap_reducer.fit_transform(features)

        reduced_features[(model_name, '-')] = features


    results = {}
    for (model_name, reduction_technique), features in reduced_features.items():
        # DBSCAN
        db = DBSCAN(eps=0.5, min_samples=5).fit(features)
        db_labels = db.labels_

        # HDBSCAN
        hdb = hdbscan.HDBSCAN(min_cluster_size=5).fit(features)
        hdb_labels = hdb.labels_

        # Mean Shift
        bandwidth = estimate_bandwidth(features, quantile=0.2)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(features)
        ms_labels = ms.labels_

        # Affinity Propagation
        af = AffinityPropagation().fit(features)
        af_labels = af.labels_

        # Store results
        for algorithm, labels in [('dbscan', db_labels), ('hdbscan', hdb_labels), 
                                ('meanshift', ms_labels), ('affinity', af_labels)]:
            key = (model_name, reduction_technique, algorithm)
            silhouette = metrics.silhouette_score(features, labels) if len(set(labels)) > 1 else -1
            results[key] = {'silhouette_score': silhouette}


    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    results_df.columns = ['Model', 'Reduction Technique', 'Algorithm', 'Silhouette Score']  #, 'Accuracy'
    results_df.sort_values(by='Silhouette Score', ascending=False, inplace=True)

    results_df.reset_index(drop=True, inplace=True)
    results_df.index += 1

    results_df['Rank'] = results_df.index

    print(results_df.head())

    try:
        results_df.head().to_csv('permutation_results.csv', mode='a', index=False, header=False)

    except:
        results_df.head().to_csv('permutation_results.csv', mode='w+', index=False, header=True)