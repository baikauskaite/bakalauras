import string
from conllu import parse_incr
import os
import spacy

LANGUAGE = "french"

LANGUAGE_SHORT = "fr" if LANGUAGE == "french" else "de"
BASE_DIR = os.path.join("/home/viktorija/bakalaurinis/context-debias/data", LANGUAGE)
IN_FILE = os.path.join(BASE_DIR, "grammatical-gender", f"{LANGUAGE_SHORT}_news_2023_1M-words.txt")
FEM_OUTPUT_FILE = os.path.join(BASE_DIR, "grammatical-gender", f"{LANGUAGE_SHORT}-fem-v3.txt")
MASC_OUTPUT_FILE = os.path.join(BASE_DIR, "grammatical-gender", f"{LANGUAGE_SHORT}-masc-v3.txt")
STEREOTYPES_FILE = os.path.join(BASE_DIR, "stereotypes.txt")

punctuation = set(string.punctuation)
nlp = spacy.load(f"{LANGUAGE_SHORT}_core_news_sm")

def read_words(file_path):
    words = set()

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split("\t")
            word = parts[1]
            frequency = int(parts[2])
            
            if frequency < 80:
                break
            elif is_valid(word):
                words.add(word)

    return words

def separate_by_gender(words):
    fem_words = set()
    masc_words = set()

    for word in words:
        tokens = nlp(word)
        word = tokens[0]
        if word.pos_ != "NOUN":
            continue
        gender = word.morph.get("Gender")
        if gender == ["Fem"]:
            fem_words.add(word.lemma_.lower())
        elif gender == ["Masc"]:
            masc_words.add(word.lemma_.lower())
            
    return fem_words, masc_words

def is_valid(word):
    if any(char in punctuation or char.isdigit() or char.isspace() for char in word) or len(word) < 3 or len(word) > 15 or word.isupper():
        return False
    return True

def write_words_to_file(words, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for word in words:
            file.write(f"{word}\n")


words = read_words(IN_FILE)
print(f"Read {len(words)} words from {IN_FILE}")
fem_words, masc_words = separate_by_gender(words)
print(f"Found {len(fem_words)} fem words and {len(masc_words)} masc words")

write_words_to_file(fem_words, FEM_OUTPUT_FILE)
write_words_to_file(masc_words, MASC_OUTPUT_FILE)

print(f"Randomized words from {IN_FILE} have been written to {FEM_OUTPUT_FILE} and {MASC_OUTPUT_FILE}")

