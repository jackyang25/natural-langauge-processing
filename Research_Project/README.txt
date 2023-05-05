about the project:
- investigate novel technique to make training large language models (LLM) for fine-tuning tasks less computationally demanding and more accurate through TF-IDF filtering and compression
- test pretrained LLM (GPT-Neo) performance on disease classification tasks using experimental evaluation and comparative analysis with control and experimental models


about the programs:

1) merger.py
- combines all text files in Medical_Texts directory 
- output: merged_dataset.txt file

2) cleaner.py
- preprocesses the text in merged_dataset.txt
- output: merged_dataset.txt file

3) compressor.py *
- main program for the TF-IDF filtering and compression layer 
- output: compressed_dataset.txt file


about the datasets:

1) merged_dataset.txt
- contains merged corpus from 7 medical textbooks 
- labelled sections (ABSTRACTID)
- metrics: 9,210,256 words, 73.1 MB

2) cleaned_dataset.txt
- preprocessed corpus to be used for TF-IDF calculations
- removed stop words, non-basic ASCII characters, digits, hyphens, capitalization, and words with less than 4 characters
- metrics: 5,303,557 (-3,906,699) words, 47.2 MB

3) compressed_dataset_81.txt
- filtered version of cleaned_dataset.txt
- kept n% of highest TF-IDF (relevant) words in each abstract and removed the rest
- number at the end of file name indicates compression amount (.81) -> 0.6000581497 actual value (% error due to duplicates)
- metrics: 2,121,114 (-3,182,442) words, 19.5 MB 