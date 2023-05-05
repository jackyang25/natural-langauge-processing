# Created by Jack Yang on 2/28/23
# noun group tagger

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import nltk


def preprocess(file):
    total_sent = []
    curr_sent = []

    with open(file, 'r') as f:
        lines = f.readlines()

    # collect data
    for line in lines:
        x = line.split()
        if not x:
            if curr_sent:
                total_sent.append(curr_sent)
                curr_sent = []
        else:
            curr_sent.append(line)

    features = {}
    sentence = 0

    # feature table
    for lst in total_sent:
        sentence += 1
        features[sentence] = []

        for i in range(len(lst)):
            curr_token = lst[i].split()
            if len(curr_token) > 2:

                if i > 1:
                    prev_2_token = lst[i - 2].split()
                    curr_token.insert(-1, prev_2_token[1])
                else:
                    curr_token.insert(-1, "START2")

                if i > 0:
                    prev_1_token = lst[i - 1].split()
                    curr_token.insert(-1, prev_1_token[1])
                else:
                    curr_token.insert(-1, "START1")

                if i < len(lst) - 2:
                    next_1_token = lst[i + 1].split()
                    curr_token.insert(-1, next_1_token[1])
                else:
                    curr_token.insert(-1, 'END1')

                if i < len(lst) - 3:
                    next_2_token = lst[i + 2].split()
                    curr_token.insert(-1, next_2_token[1])
                else:
                    curr_token.insert(-1, 'END2')

                if curr_token[0][0].isupper():
                    curr_token.insert(-1, 'True')
                else:
                    curr_token.insert(-1, 'False')

            else:
                if i > 1:
                    prev_2_token = lst[i - 2].split()
                    curr_token.append(prev_2_token[1])
                else:
                    curr_token.append("START2")

                if i > 0:
                    prev_1_token = lst[i - 1].split()
                    curr_token.append(prev_1_token[1])
                else:
                    curr_token.append("START1")

                if i < len(lst) - 2:
                    next_1_token = lst[i + 1].split()
                    curr_token.append(next_1_token[1])
                else:
                    curr_token.append('END1')

                if i < len(lst) - 3:
                    next_2_token = lst[i + 2].split()
                    curr_token.append(next_2_token[1])
                else:
                    curr_token.append('END2')

            features[sentence].append({i: v for i, v in enumerate(curr_token)})

    return features


# preprocessing
def extract_features(features_table, vectorizer=None, scaler=None, training=True):
    X = []
    Y = []

    # collect all features and labels first
    for sent in features_table:
        for word in features_table[sent]:
            features = {
                'curr_word': word[0],
                'curr_pos': word[1],
                'prev_pos1': word[3],
                'prev_pos2': word[2],
                'next_pos1': word[4],
                'next_pos2': word[5],
                'capitalized': word[6]

            }
            X.append(features)
            if training is True:
                Y.append(word[4])

    # fit the vectorizer and scaler only once on the training data
    if training is True:
        vectorizer = DictVectorizer()
        scaler = StandardScaler()
        X_numeric = vectorizer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_numeric.toarray())
    else:
        X_numeric = vectorizer.transform(X)
        X_scaled = scaler.transform(X_numeric.toarray())

    return X_scaled, Y, vectorizer, scaler


def main1():
    # load the training and test data
    training_features = preprocess('WSJ_CHUNK_FILES/WSJ_02-21.pos-chunk')
    testing_features = preprocess('WSJ_CHUNK_FILES/WSJ_24.pos')

    # extract features from the training set and scale the data
    x_train, y_train, vectorizer, scaler = extract_features(training_features)

    clf = LogisticRegression(max_iter=100)
    clf.fit(x_train, y_train)

    # extract features from the test set and scale the data
    x_test, _, _, _ = extract_features(testing_features, vectorizer, scaler, False)

    predicted_labels = clf.predict(x_test)

    lst = list(predicted_labels)


def main():
    training_features = preprocess('WSJ_CHUNK_FILES/WSJ_02-21.pos-chunk')
    testing_features = preprocess('WSJ_CHUNK_FILES/WSJ_23.pos')

    with open('MAX_ENT_FILES/training.feature', 'w') as f:
        for sent in training_features:
            for word in training_features[sent]:
                values = "\t".join(str(val) for val in word.values())
                f.write(values + '\n')
            f.write('\n')

    with open('MAX_ENT_FILES/test.feature', 'w') as f:
        for sent in testing_features:
            for word in testing_features[sent]:
                values = "\t".join(str(val) for val in word.values())
                f.write(values + '\n')
            f.write('\n')


if __name__ == "__main__":
    main1()
