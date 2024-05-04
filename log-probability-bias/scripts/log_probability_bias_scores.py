import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import copy
import os
import logging as log
import re
from tqdm import tqdm
import json

from transformers import AutoModelForMaskedLM, AutoTokenizer
from utils import split_comma_and_check, test_sort_key, initialize_model_tokenizer

TEST_EXT = '.jsonl'

####### CONFIG #######
log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--model_version', type=str)
parser.add_argument('--data_dir', '-d', type=str,
                    help="Directory containing examples for each test",
                    default='tests')
parser.add_argument('--tests', '-t', type=str,
                    help="WEAT tests to run (a comma-separated list; test files should be in `data_dir` and "
                            "have corresponding names, with extension {}). Default: all tests.".format(TEST_EXT))
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

MODEL = args.model
MODEL_VERSION = args.model_version
DATA_DIR = args.data_dir

all_tests = sorted(
    [
        entry[:-len(TEST_EXT)]
        for entry in os.listdir(args.data_dir)
        if not entry.startswith('.') and entry.endswith(TEST_EXT)
    ],
    key=test_sort_key
)
log.debug('Tests found:')
for test in all_tests:
    log.debug('\t{}'.format(test))

TESTS = split_comma_and_check(args.tests, all_tests, "test") if args.tests is not None else all_tests
log.info('Tests selected:')
for test in TESTS:
    log.info('\t{}'.format(test))

####################################

# Load pre-trained model with masked language model head
model, tokenizer = initialize_model_tokenizer(MODEL, MODEL_VERSION)

# create a list of loaded TESTS
WEAT_TESTS = []

for test in TESTS:
    WEAT_TESTS.append({
        'name': test, 
        'data': json.load(open(os.path.join(DATA_DIR, test + TEST_EXT)))
    })

# Generate target and prior sentences
for weat in WEAT_TESTS:
    templates = weat['data']['templates']
    templates = [x.replace("[TARG]", tokenizer.mask_token) for x in templates]
    weat['tgt_texts'] = []
    weat['prior_texts'] = []
    weat['attributes'] = []

    for ATTRIBUTE in ['attr1', 'attr2']:
        for template in templates:

            if 'ATTR' in template:
                examples = weat['data'][ATTRIBUTE]['examples']

                for attr_word in examples:
                    tmp = copy.deepcopy(template)
                    tgt_text = tmp.replace("[ATTR]", attr_word)
                    prior_text = tmp.replace("[ATTR]", f'{tokenizer.mask_token} ' * len(attr_word.split(" "))) # Not sure why this is needed

                    weat['tgt_texts'].append(tgt_text)
                    weat['prior_texts'].append(prior_text)
                    weat['attributes'].append(weat['data'][ATTRIBUTE]['category'])

# Function for finding the target position (helper function for later)
def find_tgt_pos(text, tgt):
    txt = text.split(" ")
    for i in range(len(txt)):
        if tgt in txt[i]: # careful with things like "_," or "_."
            return i
    # if we've looped all positions but didn't find _
    print('Target position not found!')
    raise


# Return probability for the target word, and fill in the sentence (just for debugging)
# Assumes that the first masked position is the target word
def predict_word(text: str, model: AutoModelForMaskedLM, tokenizer: AutoTokenizer, tgt_word: str):
    tokenized_text = tokenizer(text, return_tensors='pt')

    model.eval()
    with torch.no_grad():
        outputs = model(**tokenized_text)
        predictions = outputs.logits

    # normalize by softmax
    predictions = F.softmax(predictions, dim=-1)
    mask_positions = []

    for i in range(len(tokenized_text["input_ids"][0])):
        if tokenized_text["input_ids"][0][i] == tokenizer.mask_token_id:
            mask_positions.append(i)

    # For the target word position, get probabilities for each word of interest
    normalized = predictions[0, mask_positions[0], :]
    tgt_word_token = tokenizer.tokenize(tgt_word)
    tgt_word_id = tokenizer.convert_tokens_to_ids(tgt_word_token)[0]
    out_prob = normalized[tgt_word_id].item()

    # Use the capitalized version of the target word if it has a higher probability
    capitalized = tgt_word.capitalize()
    if capitalized in tokenizer.get_vocab():
        tgt_word_token_cap = tokenizer.tokenize(capitalized)
        tgt_word_id_cap = tokenizer.convert_tokens_to_ids(tgt_word_token_cap)[0]
        out_prob_cap = normalized[tgt_word_id_cap].item()

        if out_prob_cap > out_prob:
            tgt_word_id = tgt_word_id_cap
            tgt_word = capitalized
            out_prob = out_prob_cap

    # No masks are filled in
    # tokenized_text = tokenizer.decode(tokenized_text['input_ids'][0])
    
    # Fill the first mask with the target word
    # tokenized_text2 = copy.deepcopy(tokenized_text)
    # tokenized_text2["input_ids"][0][mask_positions[0]] = tgt_word_id
    # tokenized_text2 = tokenizer.decode(tokenized_text2['input_ids'][0])

    # Fill in all masks by max probability
    tokenized_text3 = copy.deepcopy(tokenized_text)
    for mask_pos in mask_positions:
        predicted_index = torch.argmax(predictions[0, mask_pos, :]).item()
        tokenized_text3['input_ids'][0][mask_pos] = predicted_index
    
    tokenized_text3 = tokenizer.decode(tokenized_text3['input_ids'][0])

    # Get top 5 predictions
    # topk = torch.topk(predictions[0, mask_positions[0], :], 5)
    # topk_probs = topk.values.tolist()
    # topk_tokens = tokenizer.convert_ids_to_tokens(topk.indices.tolist())
    # print('Top 5 predictions: ', topk_tokens)
    # print('Top 5 probabilities: ', topk_probs)

    return out_prob, tokenized_text3


# run through all generated templates and calculate results dataframe
for weat in WEAT_TESTS:
    results = {}
    results['attribute'] = []
    results['target'] = []
    results['tgt_word'] = []
    results['tgt_sent'] = []
    results['log_probs'] = []
    results['pred_sent'] = []

    # Run through all generated permutations
    for i in tqdm(range(len(weat['tgt_texts']))):
        tgt_text = weat['tgt_texts'][i]
        prior_text = weat['prior_texts'][i]

        for TARGET in ['targ1', 'targ2']:
            examples = weat['data'][TARGET]['examples']

            for tgt_word in examples:
                tgt_probs, pred_sent = predict_word(tgt_text, model, tokenizer, tgt_word)
                prior_probs, _ = predict_word(prior_text, model, tokenizer, tgt_word)

                # calculate log and store in results dictionary
                tgt_probs, pred_sent, prior_probs = np.array(tgt_probs), np.array(pred_sent), np.array(prior_probs)
                log_probs = np.log(tgt_probs / prior_probs)

                results['attribute'].append(weat['attributes'][i])
                results['target'].append(weat['data'][TARGET]['category'])
                results['tgt_word'].append(tgt_word)
                results['tgt_sent'].append(tgt_text)
                results['log_probs'].append(log_probs)
                results['pred_sent'].append(pred_sent)

    # Write results to tsv
    results = pd.DataFrame(results)
    OUT_FILE = os.path.join(args.output_dir, f"{weat['name']}-scores.tsv")
    results.to_csv(OUT_FILE, sep='\t', index=False)
    log.info('Results written to {}'.format(OUT_FILE))

