
about the datasets:

1) merged_dataset.txt
- contains merged corpus from 7 medical textbooks 
- labelled sections (ABSTRACTID)
- metrics: 9,210,256 words, 73.1 MB

2) cleaned_dataset.txt
- preprocessed corpus to be used for TF-IDF calculations
- removed stop words, non-basic ASCII characters, digits, hyphens, capitalization, and words with less than 4 characters
- metrics: 5,303,557 (-3,906,699) words, 47.2 MB

3) compressed60_dataset.txt
- filtered version of cleaned_dataset.txt
- kept n% of highest TF-IDF (relevant) words in each abstract and removed the rest
- number in file name indicates compression amount, not filter amount (.80) -> 0.6000581497 actual value (% error due to duplicates)
- metrics: 2,121,114 (-3,182,442) words, 19.5 MB 
