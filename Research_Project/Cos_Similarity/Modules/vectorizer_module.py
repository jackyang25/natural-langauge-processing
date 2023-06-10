from gensim.models import KeyedVectors
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import pickle
from tqdm import tqdm
import shutil
import os


def calculate_vectors(model_type, word_list, vocabulary_file):

    if model_type.lower() == 'glove':
        model = KeyedVectors.load_word2vec_format('Vector_Models/GloVe/glove.6B.50d.txt', binary=False,
                                                  no_header=True)
    elif model_type.lower() == 'google':
        model = KeyedVectors.load_word2vec_format('Vector_Models/Google/GoogleNews-vectors-negative300.bin',
                                                  binary=True) # case sensitive
    else:
        raise ValueError('Invalid model type.')

    word_vectors = {}
    unknown_words = []

    for word in tqdm(word_list):
        try:
            word_to_use = word.lower() if model_type.lower() != 'google' else word
            word_vectors[word_to_use] = model[word_to_use]
        except KeyError:
            unknown_words.append(word_to_use)

    output_file = vocabulary_file.replace('vocabulary', 'vectors')
    output_file_base = output_file.split('vectors')[0]
    output_file = output_file_base + 'vectors.pkl'

    with open('Datasets/Temp/' + output_file, 'wb') as f:
        pickle.dump(word_vectors, f)

    output_file = output_file_base + 'vectors.txt'
    with open('Datasets/Temp/' + output_file, 'w') as f:
        f.write(f'{model_type} \n')
        for word, vector in word_vectors.items():
            f.write(f'{word}, {vector.tolist()}\n')

        for word in unknown_words:
            f.write(f'{word}, <UNK>\n')

    output_directory = 'Training_Data/New/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(output_directory + output_file, 'w') as f:
        f.write(f'{model_type} \n')
        for word, vector in word_vectors.items():
            f.write(f'{word}, {vector.tolist()}\n')

        for word in unknown_words:
            f.write(f'{word}, <UNK>\n')

    return word_vectors, unknown_words


def convert_file(file_path):
    unique_words = set()

    with open(file_path, 'r') as f:
        for line in f:
            for word in line.rstrip().split():
                unique_words.add(word)

    word_list = list(unique_words)
    return word_list


def reduce_dimensions(word_vectors, dim_reduction_method, output_file_base):
    vector_matrix = np.array(list(word_vectors.values()))

    for n_components in [2, 3]:

        if dim_reduction_method.lower() == 'pca':
            pca = PCA(n_components=n_components)
            reduced_vectors = pca.fit_transform(vector_matrix)
            output_file = output_file_base + f'vectors_{n_components}D_PCA.txt'
        elif dim_reduction_method.lower() == 'tsne':
            tsne = TSNE(n_components=n_components)
            reduced_vectors = tsne.fit_transform(vector_matrix)
            output_file = output_file_base + f'vectors_{n_components}D_tSNE.txt'
        elif dim_reduction_method.lower() == 'umap':
            umap = UMAP(n_components=n_components)
            reduced_vectors = umap.fit_transform(vector_matrix)
            output_file = output_file_base + f'vectors_{n_components}D_UMAP.txt'
        else:
            raise ValueError('Invalid dimensionality reduction method.')

        with open('Training_Data/New/Analytics/' + output_file, 'w') as f:
            for word, vector in zip(word_vectors.keys(), reduced_vectors):
                f.write(f'{word}, {vector.tolist()}\n')




def vectorize(model_type, vocabulary_file, dim_reduction_method=None):
    word_list = convert_file('Datasets/Vocabulary/' + vocabulary_file)

    output_directory = 'Training_Data/New/Analytics/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print("Vectorizing", vocabulary_file, "...")

    try:
        word_vectors, unknown_words = calculate_vectors(model_type, word_list, vocabulary_file)
    except ValueError as e:
        print(e)
        return

    if dim_reduction_method is not None:
        output_file_base = vocabulary_file.replace('vocabulary', 'vectors').split('vectors')[0]
        reduce_dimensions(word_vectors, dim_reduction_method, output_file_base)

    for word, vector in word_vectors.items():
        print()
        print(word, vector)

    print()
    print("Completed vectorization. \n")

    output_directory = 'Training_Data/New'
    shutil.copy2('Datasets/Vocabulary/' + vocabulary_file, output_directory)
