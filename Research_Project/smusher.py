
def remove_abstract_id_and_blank_lines(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    cleaned_lines = [line.replace('ABSTRACTID', '') for line in lines if line.strip() != '']

    with open(output_file, 'w') as file:
        file.writelines(cleaned_lines)

def main():
    input_file = "Datasets/merged_dataset.txt"
    output_file = "Datasets/merged_new.txt"
    remove_abstract_id_and_blank_lines(input_file, output_file)

if __name__ == "__main__":
    main()