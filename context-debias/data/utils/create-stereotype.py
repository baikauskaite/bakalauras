import argparse
import random
import string
import os

LANGUAGE = "french"

LANGUAGE_SHORT = "fr" if LANGUAGE == "french" else "de"
BASE_DIR = os.path.join("/home/viktorija/bakalaurinis/context-debias/data", LANGUAGE)
FEM_FILE = os.path.join(BASE_DIR, "grammatical-gender", f"{LANGUAGE_SHORT}-fem-v3.txt")
MASC_FILE = os.path.join(BASE_DIR, "grammatical-gender", f"{LANGUAGE_SHORT}-masc-v3.txt")
STEREOTYPES_FILE = os.path.join(BASE_DIR, "stereotypes.txt")
FEM_ATTR_FILE = os.path.join(BASE_DIR, "female.txt")
MASC_ATTR_FILE = os.path.join(BASE_DIR, "male.txt")


def read_and_randomize_words(file_path, word_limit=1200):
    punctuation = set(string.punctuation)

    with open(FEM_ATTR_FILE, 'r', encoding='utf-8') as file:
        fem_attributes = [word.lower() for word in file.read().split()]

    with open(MASC_ATTR_FILE, 'r', encoding='utf-8') as file:
        masc_attributes = [word.lower() for word in file.read().split()]
    
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().split()
        words = [word for word in words
                 if not any(char in punctuation for char in word)
                 and not any(char.isdigit() for char in word) 
                 and len(word) > 2 and len(word) < 15
                 and word not in fem_attributes
                 and word not in masc_attributes]
        random.shuffle(words)

        return words[:word_limit]

def write_words_to_file(words, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for word in words:
            file.write(f"{word}\n")


words_file1 = read_and_randomize_words(FEM_FILE)
words_file2 = read_and_randomize_words(MASC_FILE)

combined_words = words_file1 + words_file2
random.shuffle(combined_words)

write_words_to_file(combined_words, STEREOTYPES_FILE)

print(f"Randomized words from {FEM_FILE} and {MASC_FILE} have been written to {STEREOTYPES_FILE}")

