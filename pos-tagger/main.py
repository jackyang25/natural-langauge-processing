# created by Jack Yang on 2/7/23
# part-of-speech tagger

import copy
import math
import nltk

# emission probability matrix
def emission(training_corpus):
    # emission table
    emission_table = {}
    pos_count = {"START": 0}

    # preprocessing
    for line in training_corpus:
        words = line.split()
        if len(words) == 2 and words[1] not in emission_table:
            emission_table[words[1]] = {}
            emission_table[words[1]][words[0]] = 1

        elif len(words) == 2 and words[1] in emission_table:
            if words[0] not in emission_table[words[1]]:
                emission_table[words[1]][words[0]] = 1
            else:
                emission_table[words[1]][words[0]] += 1

    word_count = copy.deepcopy(emission_table)

    # convert to probability
    for key in emission_table.keys():
        frequency = 0
        for key2 in emission_table[key]:
            frequency += emission_table[key][key2]

        for key2 in emission_table[key]:
            emission_table[key][key2] = math.log(emission_table[key][key2] / frequency)

        pos_count[key] = frequency
    return emission_table, pos_count, word_count


# transition probability matrix
def transition(training_corpus, emission_table, pos_count):
    # create transition table
    transition_table = {}
    sentence = ["START"]

    # preprocessing
    for line in training_corpus:
        x = line.split()
        if len(x) != 0:
            sentence.append(x[1])
        else:
            sentence.append("END")
            for i in range(len(sentence) - 1):
                bigram = (sentence[i], sentence[i + 1])
                if bigram[0] not in transition_table:
                    transition_table[bigram[0]] = {}
                if bigram[1] not in transition_table[bigram[0]]:
                    transition_table[bigram[0]][bigram[1]] = 1

                else:
                    transition_table[bigram[0]][bigram[1]] += 1

            sentence = ["START"]
            pos_count["START"] += 1

    # print(transition_table)

    # Laplace smoothing to handle OOV words
    for key in emission_table:
        for k in transition_table:
            if key not in transition_table[k]:
                transition_table[k][key] = 1

    # convert to probability
    for key in transition_table.keys():
        for key2 in transition_table[key]:
            transition_table[key][key2] = math.log(transition_table[key][key2] / pos_count[key])

    return transition_table


# modified Viterbi implementation
def viterbi(test_corpus, emission_table, transition_table, pos_count, word_count):
    corpus = []
    sentence = []
    for line in test_corpus:
        if line == '\n':
            corpus.append(sentence)
            sentence = []
        else:
            sentence.append(line.strip())

    # initialize per-sentence POS tagging
    for sentence in corpus:
        pi = []
        total = 0
        for key in emission_table:
            try:
                if sentence[0] in emission_table[key]:
                    total += word_count[key][sentence[0]]
                    pi.append([key, word_count[key][sentence[0]]])
            except KeyError:
                total += 1
                word_count[key][sentence[0]] = 1
                pi.append([key, word_count[key][sentence[0]]])

        # Laplace smoothing to handle OOV words
        if not pi:
            for key in emission_table:
                word_count[key][sentence[0]] = 1
                total += 1
                transition_table['START'][key] = 1
                pos_count[key] += 1

            # update emission probability matrix with new words
            for key in word_count:
                emission_table[key][sentence[0]] = math.log(word_count[key][sentence[0]] / total)
                pi.append([key, word_count[key][sentence[0]]])

        # starting possibilities (=1.0)
        for elem in pi:
            elem[1] = elem[1] / total
        matrix = []

        for i in range(1, len(sentence)):
            curr_word = sentence[i]
            matrix.append([])
            for pos in emission_table:
                if curr_word in emission_table[pos]:
                    matrix[-1].append((pos, emission_table[pos][curr_word], curr_word))

            # Laplace smoothing to handle OOV words
            if not matrix[-1]:
                OOV_freq = 0
                for key in emission_table:
                    emission_table[key][curr_word] = 1
                    OOV_freq += 1

                for key in emission_table:
                    # logarithmic normalization to prevent underflow
                    emission_table[key][curr_word] = math.log(emission_table[key][curr_word] / OOV_freq)

                for key in emission_table:
                    matrix[-1].append((key, emission_table[key][curr_word], curr_word))

        # initialize trellis
        trellis = []
        time_step = []
        state = {}

        # initialize starting states
        for elem in pi:
            state['tag'] = elem[0]
            state['prob'] = elem[1]
            state['prev'] = None
            time_step.append(state)
            state = {}

        trellis.append(time_step)
        time_step = []
        temp = {}

        # traverse k+1 tokens

        for lst in matrix:
            for curr in lst:
                for prev in trellis[-1]:

                    state['token'] = curr[2]
                    state['prev'] = prev['tag']
                    state['tag'] = curr[0]

                    try:
                        if state['tag'] in temp:
                            state['prob'] = max(temp[state['tag']], state['prob'])

                        elif state['tag'] not in temp:
                            state['prob'] = (prev['prob']) + (
                                transition_table[state['prev']][state['tag']]) + (
                                                emission_table[state['tag']][curr[-1]])

                        temp[state['tag']] = state['prob']
                        temp['token'] = state['token']
                        time_step.append(state)

                    except KeyError:
                        pass

                    state = {}
            temp = {}
            trellis.append(time_step)
            time_step = []
        print()
        print('<------------------------------------>')

        # outputs time steps for trellis
        for i in range(len(trellis)):
            print()
            print("Time Step:", i)
            for j in trellis[i]:
                print(j)

        # print(trellis)

        # update 'token' for state 0
        for fix in trellis[0]:
            fix['token'] = sentence[0]

        # generate POS tags
        trellis.reverse()
        POS_tags = []

        # trace path with max probability pointers
        for time_step in trellis:
            max_prob_tag = None
            max_prob = float('-inf')
            for state in time_step:
                if state['prob'] > max_prob:
                    max_prob = state['prob']
                    max_prob_tag = state['tag']
                    max_prob_token = state['token']
            POS_tags.append([max_prob_token, max_prob_tag])

        POS_tags.reverse()

        with open("sys_output.pos", "a") as output_file:
            for POS in POS_tags:
                output_file.write(POS[0] + "\t" + POS[1] + "\n")
            output_file.write("\n")


def main():
    with open('sys_output.pos', 'w') as output_file, \
            open('WSJ_POS_CORPUS/WSJ_02-21.pos', 'r') as training_file, \
            open('WSJ_POS_CORPUS/test.words', 'r') as test_file:
        output_file.truncate(0)
        training_corpus = training_file.readlines()
        test_corpus = test_file.readlines()

    # train HMM model
    emission_table, pos_count, word_count = emission(training_corpus)
    transition_table = transition(training_corpus, emission_table, pos_count)

    # decode POS sequence
    viterbi(test_corpus, emission_table, transition_table, pos_count, word_count)


if __name__ == "__main__":
    main()
