import os

def remove_abstract_id_and_blank_lines(input_file, add_period=False):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    cleaned_lines = [line.replace('ABSTRACTID', '').rstrip() for line in lines if "ABSTRACTID" in line or line.strip() != '']
    if add_period:
        cleaned_lines = [line + '.' if line != '' else line for line in cleaned_lines]

    input_file_basename, ext = os.path.splitext(input_file)
    if "compressed" in input_file_basename:
        output_file = input_file_basename.replace("raw_compressed", "compressed") + ext
    elif "merged" in input_file_basename:
        output_file = input_file_basename.replace("raw_merged", "merged") + ext
    else:
        output_file = input_file_basename + "_new" + ext

    with open(output_file, 'w') as file:
        file.writelines(line + '\n' for line in cleaned_lines)

def main():
    input_file = "Datasets/raw_compressed_dataset81.txt"
    remove_abstract_id_and_blank_lines(input_file, add_period=True)

if __name__ == "__main__":
    main()
