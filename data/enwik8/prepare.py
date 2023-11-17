from datasets import load_dataset

def download_and_dump_enwiki8(file_path):
    # Load the enwiki8 dataset
    dataset = load_dataset("enwik8", split='train')

    # Write the dataset to a text file
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in dataset:
            text = entry['text']
            file.write(text)

# Specify the path where you want to save the text file
output_file_path = 'data/enwik8.txt'

# Call the function
download_and_dump_enwiki8(output_file_path)
