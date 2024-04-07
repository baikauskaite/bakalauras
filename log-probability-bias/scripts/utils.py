import os
import re
from transformers import CamembertForMaskedLM, CamembertTokenizer, CamembertConfig

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

def initialize_model_tokenizer(model_type, model_dir):
    if model_type == 'camembert':
        model, tokenizer = load_camembert(model_dir)
    else:
        raise ValueError('Model not supported')
    
    return model, tokenizer

def load_camembert(model_dir):
    if model_dir is None:
        model = CamembertForMaskedLM.from_pretrained('camembert-base')
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    else:
        path = os.path.join(model_dir, 'camembert')
        config = CamembertConfig.from_pretrained(path)
        model = CamembertForMaskedLM.from_pretrained(path, config=config)
        tokenizer = CamembertTokenizer.from_pretrained(path)
    
    return model, tokenizer