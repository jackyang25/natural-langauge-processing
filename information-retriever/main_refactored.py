import numpy as np
import re
import math
import stop_list

def read_file_lines(filename):
    with open(filename, 'r') as file:
        return file.readlines()

def preprocess_line(line, stop_words):
    x = line.split()
    x = [re.sub(r'[()\[\]/?,+W]', '', word) for word in x]
    new_x = []

    for word in x:
        if word not in stop_words and len(word) > 2 and not word.isdigit() and '.' not in word:
            new_x.append(word)

    return new_x

def get_term_frequencies(lines):
    tf = {}
    query_id = 0

    for line in lines:
        x = preprocess_line(line, stop_list.closed_class_stop_words)

        if x[0] == '.I':
            query_id += 1
            tf[query_id] = {}

        for word in x:
            if word not in tf[query_id]:
                tf[query_id][word] = 1
            else:
                tf[query_id][word] += 1

    return tf

def get_inverse_document_frequencies(tf, num_docs):
    idf = {}

    for key in tf:
        for word in tf[key]:
            if word not in idf:
                idf[word] = 0

    for word in idf:
        for query in tf:
            if word in tf[query]:
                idf[word] += 1

    for word in idf:
        idf[word] = math.log(num_docs / idf[word])

    return idf

def get_top_n_words(idf, n_percent):
    num_words = len(idf)
    n = int(num_words * n_percent)
    sorted_idf = {k: v for k, v in sorted(idf.items(), key=lambda item: item[1], reverse=True)}
    return list(sorted_idf.keys())[:n]

def vectorize_tf(tf, idf, top_n_words):
    vectors = {}
    for idq in tf:
        temp = {}
        for word in tf[idq]:
            var = 1
            if word in top_n_words:
                var = idf[word]
            temp[word] = (tf[idq][word] * var)
        vectors[idq] = temp
    return vectors

def vectorize_abstracts(abstract_tf, abstract_idf, top_n_words):
    raw_vectors = {}
    for ida in abstract_tf:
        temp = {}
        for word in abstract_tf[ida]:
            var = 1
            if word in top_n_words:
                var = abstract_idf[word] * 1.15
            temp[word] = (abstract_tf[ida][word] * var)
        raw_vectors[ida] = temp

    abstract_vectors = {}
    for key in query_vectors:
        for key2 in raw_vectors:
            temp = []
            for word in query_vectors[key]:
                if word in raw_vectors[key2]:
                    temp.append(raw_vectors[key2][word])
                else:
                    temp.append(0)

            if sum(temp) > 0:
                abstract_vectors[(key, key2)] = temp

    return abstract_vectors

def calculate_cosine_similarity(query_vectors, abstract_vectors):
    abstract_scores = {}
    for key in abstract_vectors:
        abstract_vector_list = np.array(abstract_vectors[key])
        query_vector_list = np.array(list(query_vectors[key[0]].values()))
        dot_product = np.dot(query_vector_list, abstract_vector_list)
        query_magnitude = np.linalg.norm(query_vector_list)
        abstract_magnitude = np.linalg.norm(abstract_vector_list)
        cosine_similarity = dot_product / (query_magnitude * abstract_magnitude)

        if key[0] not in abstract_scores:
            abstract_scores[key[0]] = {}
        if key[0] in abstract_scores:
            abstract_scores[key[0]][key[1]] = cosine_similarity

        print('Query Vector:', key[0], query_vector_list)
        print('Abstract Vector:', key[1], abstract_vector_list)
        print('Cosine Similarity:', cosine_similarity)
        print()

    return abstract_scores

if __name__ == '__main__':
    filename = 'sample_abstracts.txt'
    lines = read_file_lines(filename)
    tf = get_term_frequencies(lines)
    num_docs = len(tf)
    idf = get_inverse_document_frequencies(tf, num_docs)
    top_n_words = get_top_n_words(idf, 0.05)
    query_vectors = vectorize_tf(tf, idf, top_n_words)

    abstract_filename = 'sample_query.txt'
    abstract_lines = read_file_lines(abstract_filename)
    abstract_tf = get_term_frequencies(abstract_lines)
    abstract_vectors = vectorize_abstracts(abstract_tf, idf, top_n_words)

    scores = calculate_cosine_similarity(query_vectors, abstract_vectors)
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}

    for key in sorted_scores:
        print(f"Abstract {key[1]} has a score of {sorted_scores[key]} with Query {key[0]}.")


