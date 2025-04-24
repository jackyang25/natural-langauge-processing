import re
import os
import sys

# python3 dollar_program.py test_dollar_phone_corpus.txt

filename = sys.argv[1]
with open(filename, 'r') as file:
    corpus = file.read()

pattern1 = re.compile("(?i)(((zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|(\w*teen)|(\w*ty))\s{1})+)(((hundred|thousand|million|billion)\s{1})+)?(cents|dollars)"
                      "|(?i)((\$((\d)|(,))+)((.)\d{2})?(\s{1}(dollars|dollar))?)"
                      "|(?i)((\$((\d)|(,))+)(cents|cent|dollars|dollar)\s{1}(and)\s{1}(\$((\d)|(,))+)(cents|cent|dollars|dollar))"
                      "|(?i)((\$?)((\d)|(,))+\s{1}(cents|cent|dollars|dollar|hundred|thousand|million|billion)((.)\d{2})?)"
                      "|(?i)((\d)+\s{1}(dollars|dollar)\s{1}(and)\s{1}(\d)+\s{1}(cents|cent))"
                      "|(?i)(a dollar)")

matches = pattern1.finditer(corpus)

with open('dollar_output.txt', 'w') as output_file:
    for match in matches:

        text = match.group()
        new = text.replace("\n", " ")
        print(new)
        output_file.write(new + os.linesep)

