# created by Jack Yang on 2/21/23
# ad-hoc information retriever

# messy code, will turn into functions later :)

import numpy as np
import re
import math
import stop_list

with open('sys_output.txt', 'w') as file:
    file.truncate(0)

# term frequency (queries)
with open('Cranfield_Collection/cran.qry', 'r') as file:
    lines = file.readlines()

top = 75
query_id = 0
queries_tf = {}

for line in lines:
    x = line.split()
    x = [re.sub(r'[()\[\]/?,+W]', '', word) for word in x]
    new_x = []

    if x[0] == '.I':
        query_id += 1
        queries_tf[query_id] = {}

    elif '.I' not in new_x:
        for word in x:
            if word not in stop_list.closed_class_stop_words and len(
                    word) > 2 and not word.isdigit() and '.' not in word:
                new_x.append(word)

    for word in new_x:
        if word not in queries_tf[query_id]:
            queries_tf[query_id][word] = 1
        else:
            queries_tf[query_id][word] += 1

# inverse document frequency (queries)
queries_idf = {}

for key in queries_tf:
    for word in queries_tf[key]:
        if word not in queries_idf:
            queries_idf[word] = 0

for word in queries_idf:
    for query in queries_tf:
        if word in queries_tf[query]:
            queries_idf[word] += 1

for word in queries_idf:
    queries_idf[word] = math.log(query_id / queries_idf[word])

# term frequency (abstracts)
with open('Cranfield_Collection/cran.all.1400', 'r') as file:
    lines = file.readlines()

abstract_id = 0
abstract_tf = {}
title = []

for line in lines:
    x = line.split()
    x = [re.sub(r'[\(\)\[\]/\?,\+]|\d', '', word) for word in x]

    new_x = []
    if x[0] == '.I':
        abstract_id += 1
        abstract_tf[abstract_id] = {}

    elif '.I' not in new_x:
        for word in x:
            if word not in stop_list.closed_class_stop_words \
                    and len(word) > 2 and '.' not in word:
                new_x.append(word)

    for word in new_x:
        if word not in title:
            if word not in abstract_tf[abstract_id]:
                abstract_tf[abstract_id][word] = 1
            else:
                abstract_tf[abstract_id][word] += 1

# inverse document frequency (abstracts)
abstract_idf = {}

for key in abstract_tf:
    for word in abstract_tf[key]:
        if word not in abstract_idf:
            abstract_idf[word] = 0

for word in abstract_idf:
    for abstract in abstract_tf:
        if word in abstract_tf[abstract]:
            abstract_idf[word] += 1

for word in abstract_idf:
    abstract_idf[word] = math.log(abstract_id / abstract_idf[word])

# top N words (queries)
top_n_percent = 1  # top n% of words - best 1
num_words = len(queries_idf)
top_n = int(num_words * top_n_percent)
sorted_idf_q = {k: v for k, v in sorted(queries_idf.items(), key=lambda item: item[1], reverse=True)}
top_n_words_q = list(sorted_idf_q.keys())[:top_n]

# top N words (abstracts)
top_n_percent = .10  # top n% of words - best .10
num_words = len(abstract_idf)
top_n = int(num_words * top_n_percent)
sorted_idf_a = {k: v for k, v in sorted(abstract_idf.items(), key=lambda item: item[1], reverse=True)}
top_n_words_a = list(sorted_idf_a.keys())[:top_n]

# vectorization
query_vectors = {}
for idq in queries_tf:
    temp = {}
    for word in queries_tf[idq]:
        var = 1
        if word in top_n_words_q:
            var = queries_idf[word]
        temp[word] = (queries_tf[idq][word] * var)
    query_vectors[idq] = temp

# print(query_vectors)

raw_abstract_vectors = {}
for ida in abstract_tf:
    temp = {}
    for word in abstract_tf[ida]:
        var = 1
        if word in top_n_words_a:
            var = abstract_idf[word] * 1.15
        temp[word] = (abstract_tf[ida][word] * var)
    raw_abstract_vectors[ida] = temp
# print()
# print(raw_abstract_vectors)

abstract_vectors = {}
for key in query_vectors:
    for key2 in raw_abstract_vectors:
        temp = []
        for word in query_vectors[key]:
            if word in raw_abstract_vectors[key2]:
                temp.append(raw_abstract_vectors[key2][word])
            else:
                temp.append(0)

        if sum(temp) > 0:
            abstract_vectors[(key, key2)] = temp

# cosine similarity
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

with open('sys_output.txt', 'w') as file:
    for query in abstract_scores:
        sorted_list = sorted(abstract_scores[query], key=abstract_scores[query].get, reverse=True)
        [file.write(f"{query} {abstract} {abstract_scores[query][abstract]}\n") for pi, abstract in
         enumerate(sorted_list) if pi < top]