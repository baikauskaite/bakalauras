import json
from deep_translator import GoogleTranslator
import os
import argparse

parser = argparse.ArgumentParser(
    description='Run specified SEAT tests on specified models.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument('--test_name', type=str, default='sent_weat1.jsonl',
                    help='Name of the test file to translate')
args = parser.parse_args()

BASE_PATH = "/home/viktorija/bakalaurinis/sent-bias-master/tests"
SOURCE_LANGUAGE = "english"
TARGET_LANGUAGE = "french"
TEST_NAME = args.test_name

def translate_text(text_list: list) -> list:
    translator = GoogleTranslator(source=SOURCE_LANGUAGE, target=TARGET_LANGUAGE)
    translation = translator.translate_batch(text_list)
    return translation

def translate_examples_in_file():
    from_path = os.path.join(BASE_PATH, SOURCE_LANGUAGE, TEST_NAME)

    with open(from_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Check if the target language is supported
    supported_languages = GoogleTranslator.get_supported_languages(as_dict=True)
    if TARGET_LANGUAGE not in supported_languages:
        raise ValueError(f"Language {TARGET_LANGUAGE} is not supported by the translator")
    
    # Iterate over the JSON structure and translate "examples"
    for test_type in data.values():
        if "examples" in test_type:
            test_type["examples"] = translate_text(test_type["examples"])
    
    # Save the modified JSON back to a file
    to_path = os.path.join(BASE_PATH, TARGET_LANGUAGE, TEST_NAME)

    with open(to_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"Test {TEST_NAME} has been translated from {SOURCE_LANGUAGE} to {TARGET_LANGUAGE}")


translated_file_path = translate_examples_in_file()