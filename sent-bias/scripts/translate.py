import json
from deep_translator import GoogleTranslator
import os
import argparse

parser = argparse.ArgumentParser(
    description='Translate test file from one language to another',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument('--test_name', type=str, default='sent_weat1.jsonl',
                    help='Name of the file to translate')
parser.add_argument('--dir', type=str, default='sent-bias-master/tests',
                    help='Directory where the test files are located')
args = parser.parse_args()

SOURCE_LANGUAGE = "english"
TARGET_LANGUAGE = "french"
TEST_NAME = args.test_name
BASE_PATH = os.path.join("/home/viktorija/bakalaurinis", args.dir)


def translate_text(text_list: list) -> list:
    translator = GoogleTranslator(source=SOURCE_LANGUAGE, target=TARGET_LANGUAGE)
    translation = translator.translate_batch(text_list)
    return translation


def translate_examples_in_json_file(from_path, to_path):
    with open(from_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for test_type in data.values():
        if "examples" in test_type:
            test_type["examples"] = translate_text(test_type["examples"])

    with open(to_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def translate_txt_file(from_path, to_path):
    with open(from_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        translated_lines = translate_text([line.strip() for line in lines])

    with open(to_path, 'w', encoding='utf-8') as file:
        for line in translated_lines:
            file.write(f"{line}\n")


def translate_file():
    from_path = os.path.join(BASE_PATH, SOURCE_LANGUAGE, TEST_NAME)
    to_path = os.path.join(BASE_PATH, TARGET_LANGUAGE, TEST_NAME)

    supported_languages = GoogleTranslator.get_supported_languages(as_dict=True)
    if TARGET_LANGUAGE not in supported_languages:
        raise ValueError(f"Language {TARGET_LANGUAGE} is not supported by the translator")
    
    if TEST_NAME.endswith('.json'):
        translate_examples_in_json_file(from_path, to_path)
    elif TEST_NAME.endswith('.txt'):
        translate_txt_file(from_path, to_path)
    else:
        raise ValueError("Unsupported file format. Only .json and .txt are supported.")

    return to_path


translated_file_path = translate_file()
print(f"File {TEST_NAME} has been translated from {SOURCE_LANGUAGE} to {TARGET_LANGUAGE} and saved to {translated_file_path}")