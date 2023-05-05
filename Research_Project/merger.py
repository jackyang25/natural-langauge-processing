# Created by Jack Yang on April 15

import os

dir_path = "/Users/jacky/PyCharm/Natural-Langauge-Processing/FinalProject/Medical_Texts"

text_files = [f for f in os.listdir(dir_path) if f.endswith(".txt")]

combined_text = ""

for file_path in text_files:
    with open(os.path.join(dir_path, file_path), "r", encoding="iso-8859-1") as f:
        text = f.read()
        text = text.replace("\x0c", "ABSTRACTID\n")
        combined_text += text

sections = combined_text.split("ABSTRACTID\n")
numbered_sections = ["ABSTRACTID\n" + section for section in sections if section]
numbered_text = "".join(numbered_sections)


with open("Datasets/raw_merged_dataset.txt", "w") as f:
    f.write(numbered_text)
