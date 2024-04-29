import argparse
import random

FILE1 = "de-fem-v2.txt"
FILE2 = "de-masc-v2.txt"
OUTPUT_FILE = "../stereotype.txt"


def read_and_randomize_words(file_path, word_limit=500):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().split()
        random.shuffle(words)
        return words[:word_limit]

def write_words_to_file(words, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for word in words:
            file.write(f"{word}\n")


words_file1 = read_and_randomize_words(FILE1)
words_file2 = read_and_randomize_words(FILE2)

combined_words = words_file1 + words_file2
random.shuffle(combined_words)

write_words_to_file(combined_words, OUTPUT_FILE)

print(f"Randomized words from {FILE1} and {FILE2} have been written to {OUTPUT_FILE}")

