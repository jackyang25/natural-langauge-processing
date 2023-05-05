# Created by Jack Yang on April 15

from nltk.corpus import stopwords
from nltk.tokenize import blankline_tokenize, word_tokenize
import string

#nltk.download('stopwords')

def remove_non_basic_ascii(text):
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, text))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    blank_lines = blankline_tokenize(text)
    cleaned_blank_lines = []
    for blank_line in blank_lines:
        sentences = blank_line.split('\n')
        cleaned_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            cleaned_words = [word if 'abstractid' in word.lower() else word.lower().replace('-', '') for word in words if (word.lower() not in stop_words) and (not any(char.isdigit() for char in word)) and (len(word) > 3) and ('.' not in word)]
            cleaned_words = [word.replace(',', '') for word in cleaned_words]
            cleaned_sentence = ' '.join(cleaned_words)
            cleaned_sentences.append(cleaned_sentence)
        cleaned_blank_line = '\n'.join(cleaned_sentences)
        cleaned_blank_lines.append(cleaned_blank_line)
    cleaned_text = '\n\n'.join(cleaned_blank_lines)
    return cleaned_text

def main():
    with open('Datasets/raw_merged_dataset.txt', 'r', encoding='utf-8') as infile:
        text = infile.read()

    text = remove_non_basic_ascii(text)
    cleaned_text = remove_stopwords(text)

    with open('Datasets/preprocessed_dataset.txt', 'w', encoding='utf-8') as outfile:
        outfile.write(cleaned_text)

if __name__ == '__main__':
    main()
