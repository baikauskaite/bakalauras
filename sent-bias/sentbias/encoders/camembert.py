''' Convenience functions for handling CamemBERT model'''
import torch
import os

from transformers import CamembertModel, CamembertTokenizer, CamembertConfig

def load_model(version='camembert-base', local_path=None):
    ''' 
    Load CamemBERT model and corresponding tokenizer.
    
    Parameters:
    - version: The version of the model to load from Hugging Face model hub if local_path is None. Default is 'camembert-base'.
    - local_path: The path to a local directory containing model and tokenizer files. If None, load from Hugging Face model hub.
    
    Returns:
    - model: The loaded CamemBERT model.
    - tokenizer: The corresponding tokenizer.
    '''
    if local_path is None:
        model = CamembertModel.from_pretrained(version)
        tokenizer = CamembertTokenizer.from_pretrained(version)
    else:
        path = os.path.join(local_path, 'camembert')
        config = CamembertConfig.from_pretrained(path)
        model = CamembertModel.from_pretrained(path, config=config)
        tokenizer = CamembertTokenizer.from_pretrained(path)
    
    return model, tokenizer


def encode(model: CamembertModel, tokenizer: CamembertTokenizer, texts):
    ''' Use tokenizer and model to encode texts '''
    encs = {}
    for text in texts:
        tokenized = tokenizer.tokenize(text)
        indexed = tokenizer.convert_tokens_to_ids(tokenized)
        segments_idxs = [0] * len(tokenized)
        tokens_tensor = torch.tensor([indexed])
        segments_tensor = torch.tensor([segments_idxs])
        enc = model(tokens_tensor, token_type_ids=segments_tensor)
        enc = enc.last_hidden_state
        enc = enc[:, 0, :] # extract the last rep of the first input

        # Print duplicates
        if text in encs:
            print(text)

        encs[text] = enc.detach().view(-1).numpy()

    return encs