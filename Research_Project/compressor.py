# Created by Jack Yang on April 26

import math
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(filepath):
    """Load in the text data from a file."""
    with open(filepath, 'r') as f:
        text = f.read()
    return text

def tokenize_documents(text):
    """Tokenize the text data into individual documents."""
    documents = text.split("ABSTRACTID")
    documents = [doc.strip() for doc in documents if doc.strip()]
    return documents


def compute_tfidf(documents):
    """Calculate the TF-IDF values for each word in each document."""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    tfidf_dict = defaultdict(dict)
    for doc_index, doc in enumerate(documents):
        feature_index = tfidf[doc_index, :].nonzero()[1]
        for i in feature_index:
            tfidf_dict[doc_index][feature_names[i]] = tfidf[doc_index, i]
    return tfidf_dict


def sort_tfidf_dict(tfidf_dict):
    """Sort the TF-IDF values for each word in each document in the tfidf_dict dictionary."""
    sorted_dict = {}
    for doc_id, tfidf_values in tfidf_dict.items():
        sorted_values = sorted(tfidf_values.items(), key=lambda x: x[1], reverse=True)
        sorted_dict[doc_id] = dict(sorted_values)
    return sorted_dict


def get_filter_words(percentage, sorted_dict):
    """Get the top n% words in each document in the sorted dictionary."""

    filter_words = {}
    total_words = 0
    for document in sorted_dict:
        total_words += len(sorted_dict[document])
        filter_count = int(percentage * len(sorted_dict[document]))
        curr_count = 0
        doc_filter_words = []
        for key in sorted_dict[document].keys():
            if curr_count >= filter_count:
                break
            if key not in doc_filter_words:
                doc_filter_words.append(key)
                curr_count += 1
        filter_words[document] = doc_filter_words

    print(" -- Metrics -- ")
    print(f"Number of unique words in the corpus: {total_words}")
    print(f"Number of unique words after filtering: {len(filter_words)}")
    # print(filter_words)
    return filter_words


def compress(input_filepath, filter_words, percentage):
    """Remove words from the text file that are not in the filter words list."""
    with open(input_filepath, 'r') as f:
        text = f.read()
    lines = text.splitlines()
    filtered_lines = []
    total_words_before = 0  # initialize word count before compression
    total_words_after = 0  # initialize word count after compression

    current_abstract_id = 0  # initialize current abstract id
    current_filter_words = filter_words[current_abstract_id]  # initialize current filter words


    for i, line in enumerate(lines):
        if i == 0:
            continue  # skip the first line
        words = line.split()
        total_words_before += len(words)  # update word count before compression
        filtered_words = []
        for word in words:
            if word == "ABSTRACTID":

                current_abstract_id += 1  # increment abstract id
                current_filter_words = filter_words[current_abstract_id]  # update current filter words

                filtered_words.append(word)
            elif word in current_filter_words:
                filtered_words.append(word)
        filtered_line = " ".join(filtered_words)
        filtered_lines.append(filtered_line)
        total_words_after += len(filtered_words)  # update word count after compression

    filtered_text = "\n".join(filtered_lines)

    with open(f'Training_Data/raw_compressed_dataset_{100 - (int(percentage * 100))}.txt', 'w') as f:
        f.write("ABSTRACTID\n")
        f.write(filtered_text)

    print()
    print(f'Total words before compression: {total_words_before}')
    print(f'Total words after compression: {total_words_after}')


def main():
    compress_corpus = 'Training_Data/cleaned_dataset.txt'

    text = load_data(compress_corpus)
    documents = tokenize_documents(text)
    tfidf_dict = compute_tfidf(documents)
    sorted_dict = sort_tfidf_dict(tfidf_dict)

    compression = .80  # set filter amount hyperparameter
    filter_words = get_filter_words(1 - compression, sorted_dict)

    compress(compress_corpus, filter_words, 1 - compression)


if __name__ == '__main__':
    main()
