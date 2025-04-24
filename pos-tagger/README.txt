
The program was approached by first constructing a hidden Markov model. The model building process consisted of two main steps, namely, constructing the emission probability matrix and the transition probability matrix.

Step 1) Emission Probability Matrix:

    In constructing the emission probability matrix, a nested hash table was created to store the emission probabilities. Initially, all the part-of-speech (POS) tags from the training file were set as keys, each pointing to a dictionary of potential words and its total count for the corresponding key. The count of each word was converted to observable frequencies by dividing the count by the total count. The resulting emission probabilities were then logged to avoid underflow errors in the output by normalizing the values.

    emission[key][key2] = math.log(emission[key][key2] / frequency)

Step 2) Transition Probability Matrix:

    The transition probability matrix was also created using a nested hash table. The start state was first initialized, followed by iterating over the bigram, n, n+1 until a blank line was reached, at which an end state was added. Laplace smoothing was used for KeyError handling to account for unknown transitions. This was done by using a nested iteration over the keys in the emission and transition tables, initializing a key and value to 1 if the transition did not exist. The transition probabilities were then calculated by dividing the frequency of the transition probability by the count of the current state and taking the logarithm.


Step 3) Viterbi Initialization:

    A modified Viterbi algorithm was implemented to generate the most likely POS sequence for the given tokens. The Viterbi algorithm was initialized by calculating the starting probabilities of the initial state(s) in π, which summed up to 1.0 or 100%. In the event that an out-of-vocabulary (OOV) word was encountered in the first state (no states in array π), Laplace smoothing was applied as before, and the new word, along with a universal frequency, was added to the emissions matrix and to π (=1.0). The same procedure was applied for the remaining states, adding the POS, the token, and the emission probability as a tuple to an array, applying smoothing for OOV words when necessary.

Step 4) Viterbi Implementation:

    The Viterbi algorithm implementation involved initializing a trellis, which was used to hold the time step and its states as a three-dimensional hash table. The algorithm traversed the array stored before, the current time step, and previous states, updating each state with a new probability. This was done by taking the summation of the previous state probability, the transition probability from the previous state to the current one, and the emission probability for the current state. Note that log(abc) = loga + logb + logc. If there were duplicate POS tags in each time step, the state with the highest probability was kept, and the other paths were omitted as part of the dynamic implementation. When the final state was reached, following the previous pointers backwards starting at the highest probability, the most likely POS sequence for the given tokens was generated.
