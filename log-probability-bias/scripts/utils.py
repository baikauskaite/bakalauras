import os
import re
from transformers import AutoModelForMaskedLM, AutoTokenizer

def split_comma_and_check(arg_str, allowed_set, item_type) -> list:
    ''' Given a comma-separated string of items,
    split on commas and check if all items are in allowed_set.
    item_type is just for the assert message. '''
    items = arg_str.split(',')
    for item in items:
        if item not in allowed_set:
            raise ValueError("Unknown %s: %s!" % (item_type, item))
    return items

def test_sort_key(test):
    '''
    Return tuple to be used as a sort key for the specified test name.
   Break test name into pieces consisting of the integers in the name
    and the strings in between them.
    '''
    key = ()
    prev_end = 0
    for match in re.finditer(r'\d+', test):
        key = key + (test[prev_end:match.start()], int(match.group(0)))
        prev_end = match.end()
    key = key + (test[prev_end:],)

    return key

def initialize_model_tokenizer(model_type, version):
    if 'camembert' in model_type or 'gottbert' in model_type:
        model, tokenizer = load_model_tokenizer(version)
    else:
        raise ValueError('Model not supported')
    
    return model, tokenizer

def load_model_tokenizer(version='camembert-base'):
    model = AutoModelForMaskedLM.from_pretrained(version)
    tokenizer = AutoTokenizer.from_pretrained(version)
    model.eval()
    
    return model, tokenizer