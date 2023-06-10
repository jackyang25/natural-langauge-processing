import csv
import re

def extract_from_csv(file_path):
    unique_words = set()

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 1:
                tokens = row[1].split()
                for token in tokens:
                    token_without_commas = token.replace(',', '')
                    if len(token_without_commas) > 2 and not token_without_commas.isdigit():
                        unique_words.add(token_without_commas)

    write_to_file(unique_words, 'Datasets/Vocabulary/domain_vocabulary.txt')


def extract_from_text(text_file, process_lines=False):
    if process_lines:
        processed_line = ""
    else:
        unique_words = set()

    if isinstance(text_file, str) and process_lines:
        data = [text_file]
    else:
        with open(text_file, 'r') as f:
            data = f.readlines()

    for i, line in enumerate(data):
        tokens = line.split()
        processed_tokens = []
        for token in tokens:
            # Remove all non-alphanumeric characters except hyphens
            token = re.sub('[^0-9a-zA-Z-]+', '', token)
            # Remove trailing hyphens
            token = token.rstrip('-')
            # Only add the token if it's longer than 2 characters and is not solely composed of digits
            if len(token) > 2 and not token.isdigit():
                processed_tokens.append(token)
        if process_lines:
            processed_line = ' '.join(processed_tokens)
        else:
            unique_words.update(processed_tokens)

    if process_lines:
        return processed_line
    else:
        write_to_file(unique_words, 'Datasets/Vocabulary/vocabulary_output.txt')


def write_to_file(unique_words, file_path):
    with open(file_path, 'w') as f:
        for word in unique_words:
            f.write(word + '\n')


def main():
    extract_from_text('Datasets/medical_corpus.txt')


if __name__ == "__main__":
    main()
