from transformers import GPT2Tokenizer
from Modules import compute_means_module
from Modules import vectorizer_module
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import extract_words
import pickle
import time

output_file_title = []


def concatenate(raw_training_corpus, tokenize_length):
    # Returns optimized line length using byte-pair encoding
    print("Concatenating", raw_training_corpus, "...", "\n")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    with open('Datasets/' + raw_training_corpus, 'r') as f:
        text = f.read()
        words = text.split()

        lines = []
        current_line = []
        current_token_count = 0
        for word in tqdm(words):
            tokens = tokenizer.tokenize(word)
            if current_token_count + len(tokens) > tokenize_length and current_line:
                lines.append(" ".join(current_line))
                current_line = []
                current_token_count = 0

            current_line.append(word)
            current_token_count += len(tokens)

        if current_line:
            lines.append(" ".join(current_line))

        concatenated_text = "\n".join(lines)

    with open('Datasets/concatenated_corpus.txt', 'w') as f:
        f.write(concatenated_text)

    concatenated_corpus_dict = {i + 1: line for i, line in enumerate(lines)}

    with open('Datasets/Temp/concatenated_corpus_dict.pkl', 'wb') as f:
        pickle.dump(concatenated_corpus_dict, f)

    print("Finished concatenating training corpus.")
    print()

    return concatenated_corpus_dict


def filter(mean_domain_vector, kmeans_domain_vectors, model_type, tokenize_length, filter_method,
           threshold=0.7, max_weight=0.7, mean_weight=0.3, reward_weight=0.1):

    output_file_title.append(model_type)
    output_file_title.append('_' + str(tokenize_length))
    output_file_title.append('_' + str(filter_method))

    print("Computing scores for concatenated_corpus_dict ...", "\n")

    with open("Datasets/Temp/corpus_vectors.pkl", 'rb') as f:
        corpus_vectors = pickle.load(f)

    with open("Datasets/Temp/concatenated_corpus_dict.pkl", 'rb') as f:
        concatenated_corpus_dict = pickle.load(f)

    def method_1():
        hybrid_scores = {}
        word_similarity_scores = {}

        for key in tqdm(concatenated_corpus_dict.keys()):
            line = concatenated_corpus_dict[key]
            processed_line = extract_words.extract_from_text(line, process_lines=True)

            max_cos_similarity = 0
            total_cos_similarity = 0
            high_similarity_count = 0
            count = 0
            for word in processed_line.split():
                word = word.lower() if model_type.lower() != 'google' else word
                word_vector = corpus_vectors.get(word)

                if word_vector is None:
                    continue

                word_vector_normalized = word_vector / np.linalg.norm(word_vector)
                mean_domain_vector_normalized = mean_domain_vector / np.linalg.norm(mean_domain_vector)
                cos_similarity = np.dot(word_vector_normalized, mean_domain_vector_normalized)

                max_cos_similarity = max(max_cos_similarity, cos_similarity)
                total_cos_similarity += cos_similarity

                word_similarity_scores[word] = cos_similarity

                if cos_similarity > threshold:  # adjust the threshold as needed
                    high_similarity_count += 1

                count += 1

            mean_cos_similarity = total_cos_similarity / count if count > 0 else 0

            # Hybrid score as a weighted sum of max and mean cosine similarity
            # Bonus for having more high similarity words
            reward_factor = 1 + reward_weight * high_similarity_count  # adjust the weights as needed
            hybrid_score = reward_factor * (max_weight * max_cos_similarity + mean_weight * mean_cos_similarity)

            hybrid_scores[key] = hybrid_score

        sorted_hybrid_scores = dict(
            sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True))

        sorted_word_similarity_scores = dict(
            sorted(word_similarity_scores.items(), key=lambda item: item[1], reverse=True))

        with open('Training_Data/New/Analytics/line_weighted_scores.txt', 'w') as f:
            f.write(f'Mean vector: {mean_domain_vector}\n\n')
            for line, score in sorted_hybrid_scores.items():
                f.write(f'{line},{score}\n')

        with open('Training_Data/New/Analytics/word_similarity_scores.txt', 'w') as f:
            for word, score in sorted_word_similarity_scores.items():
                f.write(f'{word},{score}\n')

        return sorted_hybrid_scores, concatenated_corpus_dict

    def method_2():
        hybrid_scores = {}
        word_similarity_scores = defaultdict(dict)

        for key in tqdm(concatenated_corpus_dict.keys()):
            line = concatenated_corpus_dict[key]
            processed_line = extract_words.extract_from_text(line, process_lines=True)

            max_cluster_similarity = 0
            total_cluster_similarity = 0
            high_similarity_count = 0
            count = 0
            for word in processed_line.split():
                word = word.lower() if model_type.lower() != 'google' else word
                word_vector = corpus_vectors.get(word)

                if word_vector is None:
                    continue

                word_vector_normalized = word_vector / np.linalg.norm(word_vector)

                max_cos_similarity = 0
                for cluster_id, cluster_vector in kmeans_domain_vectors.items():
                    cluster_vector_normalized = cluster_vector / np.linalg.norm(cluster_vector)
                    cos_similarity = np.dot(word_vector_normalized, cluster_vector_normalized)

                    max_cos_similarity = max(max_cos_similarity, cos_similarity)

                    # update the word similarity score for the current cluster
                    word_similarity_scores[cluster_id][word] = cos_similarity

                max_cluster_similarity = max(max_cluster_similarity, max_cos_similarity)
                total_cluster_similarity += max_cos_similarity

                if max_cos_similarity > threshold:
                    high_similarity_count += 1

                count += 1

            mean_cluster_similarity = total_cluster_similarity / count if count > 0 else 0

            reward_factor = 1 + reward_weight * high_similarity_count
            hybrid_score = reward_factor * (max_weight * max_cluster_similarity + mean_weight * mean_cluster_similarity)

            hybrid_scores[key] = hybrid_score

        sorted_hybrid_scores = dict(
            sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True))

        with open('Training_Data/New/Analytics/line_weighted_scores.txt', 'w') as f:
            for line, score in sorted_hybrid_scores.items():
                f.write(f'{line},{score}\n')

        with open('Training_Data/New/Analytics/word_similarity_scores.txt', 'w') as f:
            for cluster_id, scores in word_similarity_scores.items():
                sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

                f.write(f'Cluster {cluster_id}\n')

                for word, score in sorted_scores.items():
                    f.write(f'{word},{score}\n')

                f.write('\n')

        return sorted_hybrid_scores, concatenated_corpus_dict

    if filter_method == 1:
        sorted_corpus_vector_scores, concatenated_corpus_dict = method_1()

        print('Line\t\tScore')
        for key, score in sorted_corpus_vector_scores.items():
            print(f'{key}\t\t{score}')

        print()
        print("Finished computing scores using method 1.", "\n")

    elif filter_method == 2:
        sorted_corpus_vector_scores, concatenated_corpus_dict = method_2()

        print('Line\t\tScore')
        for key, score in sorted_corpus_vector_scores.items():
            print(f'{key}\t\t{score}')

        print()
        print("Finished computing scores using method 2.", "\n")

    return sorted_corpus_vector_scores, concatenated_corpus_dict


def compress(concatenated_corpus_dict, sorted_cos_similarity_scores, compression_ratio):
    output_file_title.append('_' + str(compression_ratio))  # Ignore

    top_n_amount = int(((100 / compression_ratio) * .01) * len(concatenated_corpus_dict))

    file_name = "".join(output_file_title)
    with open('Training_Data/New/' + file_name + '.txt', 'w') as f:
        for i, key in enumerate(sorted_cos_similarity_scores.keys()):
            if i >= top_n_amount:
                break
            if key in concatenated_corpus_dict:
                f.write(concatenated_corpus_dict[key] + '\n')


def main():
    # Generate vectors for vocabulary
    model = 'glove'  # glove, google
    vectorizer_module.vectorize(model, 'corpus_vocabulary.txt', None)
    vectorizer_module.vectorize(model, 'domain_vocabulary_small.txt', None)

    # Compute domain means (check minimum clusters count and domain words count)
    mean_domain_vector, domain_vector_list = compute_means_module.create_mean_vector(
        'domain_vectors.txt')
    kmeans_domain_vectors = compute_means_module.kmeans_clustering(domain_vector_list)

    # Run concatenate once on desired hyperparameters
    tokenize_length = 256 # length (BPE) per line of data
    raw_training_corpus = 'medical_corpus.txt'
    concatenate(raw_training_corpus, tokenize_length)

    # Compute scores for lines in concatenated_corpus_dict using custom scoring algorithm
    # Method 1: better using smaller vocabulary (single mean)
    # Method 2: better using larger vocabulary (clusters)
    sorted_corpus_vector_scores, concatenated_corpus_dict = filter(
        mean_domain_vector, kmeans_domain_vectors, model, tokenize_length,
        filter_method=1)

    # Compress training corpus from top scored lines
    compress(concatenated_corpus_dict, sorted_corpus_vector_scores,
             compression_ratio=4)  # compression_ratio: 4:1 = 25% of total size


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f'Execution time: {end - start} seconds')
