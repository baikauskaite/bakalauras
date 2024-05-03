import string
from conllu import parse_incr
import os

LANGUAGE = "german"

LANGUAGE_SHORT = "fr" if LANGUAGE == "french" else "de"
BASE_DIR = os.path.join("/home/viktorija/bakalaurinis/context-debias/data", LANGUAGE)
CONLLU_FILE = os.path.join(BASE_DIR, "grammatical-gender", f"{LANGUAGE_SHORT}_gsd-ud-train.conllu")
FEM_FILE = os.path.join(BASE_DIR, "grammatical-gender", f"{LANGUAGE_SHORT}-fem-v3.txt")
MASC_FILE = os.path.join(BASE_DIR, "grammatical-gender", f"{LANGUAGE_SHORT}-masc-v3.txt")
STEREOTYPES_FILE = os.path.join(BASE_DIR, "stereotypes.txt")
FEM_OUTPUT_FILE = os.path.join(BASE_DIR, "grammatical-gender", "weat0-fem.txt")
MASC_OUTPUT_FILE = os.path.join(BASE_DIR, "grammatical-gender", "weat0-masc.txt")

punctuation = set(string.punctuation)

def read_words(file_path):
    fem_words = set()
    masc_words = set()

    with open(file_path, 'r', encoding='utf-8') as file:
        for sentences in parse_incr(file):
            nouns = sentences.filter(upos='NOUN')
            fem_nouns = nouns.filter(feats__Gender='Fem')
            masc_nouns = nouns.filter(feats__Gender='Masc')

            for noun in fem_nouns:
                if is_valid(noun['lemma']):
                    fem_words.add(noun['lemma'].lower())
            for noun in masc_nouns:
                if is_valid(noun['lemma']):
                    masc_words.add(noun['lemma'].lower())

    return fem_words, masc_words

def is_valid(word):
    if any(char in punctuation or char.isdigit() or char.isspace() for char in word) or len(word) < 3 or len(word) > 15 or word.isupper():
        return False
    return True

def write_words_to_file(words, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for word in words:
            file.write(f"{word}\n")


fem_words, masc_words = read_words(CONLLU_FILE)

write_words_to_file(fem_words, FEM_OUTPUT_FILE)
write_words_to_file(masc_words, MASC_OUTPUT_FILE)

print(f"Randomized words from {CONLLU_FILE} have been written to {FEM_OUTPUT_FILE} and {MASC_OUTPUT_FILE}")

