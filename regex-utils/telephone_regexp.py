import re
import os
import sys

# python3 telephone_regexp.py test_dollar_phone_corpus.txt

filename = sys.argv[1]
with open(filename, 'r') as file:
    corpus = file.read()

pattern1 = re.compile("(((\d{3})|(\(\d{3}\))|(\[\d{3}\]))((\.|-){1}|\s{1})\d{3}((\.|-){1}|\s{1})\d{4})")


matches = pattern1.finditer(corpus)

with open('telephone_output.txt', 'w') as output_file:
    for match in matches:

        text = match.group(1)
        new = text.replace("\n", " ")
        print(new)
        output_file.write(new + os.linesep)

