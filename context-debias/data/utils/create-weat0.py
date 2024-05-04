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
FEM_OUTPUT_FILE = os.path.join(BASE_DIR, "grammatical-gender", "weat0-fem.txt")
MASC_OUTPUT_FILE = os.path.join(BASE_DIR, "grammatical-gender", "weat0-masc.txt")


def read_and_randomize_words(file_path, stereotypes_path, word_limit=300):
    punctuation = set(string.punctuation)

    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().split()
    
    with open(stereotypes_path, 'r', encoding='utf-8') as file:
        stereotypes = file.read().split()

    words = [word for word in words if word not in stereotypes]
    words = [word for word in words
                if not any(char in punctuation for char in word)
                and not any(char.isdigit() for char in word) 
                and len(word) > 2 and len(word) < 15]
    random.shuffle(words)
    
    return words[:word_limit]

def write_words_to_file(words, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("[\n")
        for word in words[:-1]:
            file.write(f"\t\"{word}\",\n")
        file.write(f"\t\"{words[-1]}\"\n]\n")


fem_words = read_and_randomize_words(FEM_FILE, STEREOTYPES_FILE)
masc_words = read_and_randomize_words(MASC_FILE, STEREOTYPES_FILE)

write_words_to_file(fem_words, FEM_OUTPUT_FILE)
write_words_to_file(masc_words, MASC_OUTPUT_FILE)

print(f"Randomized words from {FEM_FILE} and {MASC_FILE} have been written to {FEM_OUTPUT_FILE} and {MASC_OUTPUT_FILE}")

