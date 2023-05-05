
This is a Python program that performs Ad-hoc Information Retrieval using cosine similarity. The program takes two input files located, 'cran.qry' and 'cran.all.1400', and generates an output file 'output.txt'. The input files contain queries and abstracts respectively.

The program first performs term frequency and inverse document frequency calculations for queries and abstracts. It also incorporates several techniques to improve the accuracy of the results. Specifically, it uses a list of stop words imported from stop_list.py to remove common words that are unlikely to help with the retrieval, and removes punctuation and numbers using regular expressions. 

Then, it identifies the top n% of words from each set of documents based on their inverse document frequency values. These top words are then used for vectorization of queries and abstracts. Additionally, the program assigns weights to rare words in order to give them more importance in the calculation.

Using these vectors, the program calculates cosine similarity between query and abstract vectors and generates scores for each query-abstract pair, omitting 0-sum vectors and lowest scores in the process. The final scores are written to 'output.txt'.

To run the program, simply execute the code in a Python environment with the input files ('cran.qry' and 'cran.all.1400') in the same directory. The program will create the output file ('output.txt') in the same directory.