import numpy as np
from ast import literal_eval
from sklearn.cluster import KMeans, DBSCAN
from collections import defaultdict
import hdbscan
from tqdm import tqdm


def create_mean_vector(domain_vectors_txt):

    print("Computing mean vector for", domain_vectors_txt, "...")
    print()
    vector_list = []
    with open('Datasets/Temp/' + domain_vectors_txt, 'r') as f:
        for line in f:
            data = line.split(", ", 1)
            if len(data) == 2 and data[1] != "<UNK>\n":
                vector = np.array(literal_eval(data[1]))
                vector_list.append(vector)


    mean_vector = np.mean(vector_list, axis=0)

    print('Mean Vector', mean_vector, "\n")
    print("Finished computing mean. \n")

    return mean_vector, vector_list


def kmeans_clustering(vector_list):
    """
    This algorithm aims to partition n data points into k clusters in which each data point belongs
    to the cluster with the nearest mean. It's simple and fast, but requires specifying the number
    of clusters upfront and assumes that clusters are spherical.
    """
    print("Computing clusters ... \n")

    kmeans = KMeans(n_clusters=5, n_init=10)
    kmeans.fit(vector_list)
    labels = kmeans.labels_

    clusters = defaultdict(list)
    for label, vector in zip(labels, vector_list):
        clusters[label].append(vector)

    for label in tqdm(clusters):
        clusters[label] = np.mean(clusters[label], axis=0)

    for elem in clusters:
        print(f"Cluster {elem}, {clusters[elem]}\n")

    print("Finished computing clusters. \n")

    return clusters

