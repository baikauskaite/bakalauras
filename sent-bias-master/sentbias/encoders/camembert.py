''' Convenience functions for handling CamemBERT model'''
import torch

from transformers import CamembertModel, CamembertTokenizer, CamembertConfig

def load_model(version='camembert-base'):
    ''' Load camemBERT model and corresponding tokenizer '''
    tokenizer = CamembertTokenizer.from_pretrained(version)
    # config = CamembertConfig.from_pretrained(version, output_hidden_states=True)
    model = CamembertModel.from_pretrained(version)
    model.eval()

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