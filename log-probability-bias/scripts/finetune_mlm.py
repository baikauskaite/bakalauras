from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, default_data_collator
from datasets import load_dataset, load_from_disk
from transformers import TrainerCallback
import os
import collections
import numpy as np
import subprocess
import torch
import evaluate
import torch.nn.functional as F

# ##############################################################################

# Select model_checkpoint, source_model_dir, and language
BASE_DIR = "/home/viktorija/bakalaurinis/log-probability-bias"
MODEL_CHECKPOINT = "camembert-base"
# MODEL_CHECKPOINT = "uklfr/gottbert-base"
SOURCE_MODEL_DIR = os.path.join(BASE_DIR, "../models/camembert-debiased")
# SOURCE_MODEL_DIR = os.path.join(BASE_DIR, "../models/gottbert-debiased")
LANGUAGE = "fr"
# LANGUAGE = "de"

SAVE_MODEL_DIR = os.path.join(BASE_DIR, "../models")

# Select to use the original model or the debiased model
MODEL_SELECTION = MODEL_CHECKPOINT
# MODEL_SELECTION = SOURCE_MODEL_DIR

dataset_path = os.path.join(BASE_DIR, "tokenized", LANGUAGE)
model_name = MODEL_SELECTION.split("/")[-1]

num_epochs = 4
learning_rate = 2e-5
wwm_probability = 0.2
batch_size = 8
train_size = 1_000
chunk_size = 128

# ##############################################################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_SELECTION)
model = AutoModelForMaskedLM.from_pretrained(MODEL_SELECTION, output_hidden_states=False, output_attentions=False)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
accuracy_metric = evaluate.load("accuracy")

def tokenize_function(examples):
    result = tokenizer(examples["text"], truncation=False)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    filtered_labels = labels[mask]
    filtered_predictions = predictions[mask]
    filtered_logits = logits[mask]
    cross_entropy_loss = F.cross_entropy(torch.from_numpy(filtered_logits), torch.from_numpy(filtered_labels))

    return {
        "accuracy": accuracy_metric.compute(predictions=filtered_predictions, references=filtered_labels),
        "perplexity": torch.exp(cross_entropy_loss).item()
    }

class LogProbabilityCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:  # Ensures this only runs on the main process in distributed training
            model_path = os.path.join(args.output_dir, f"checkpoint-{int(state.global_step)}")
            # Save the model explicitly
            model.save_pretrained(model_path)
            if tokenizer is not None:
                tokenizer.save_pretrained(model_path)
            
            print(f"Epoch {state.epoch} has ended, model saved at {model_path}, executing script...")

            command = ['../bash_scripts/log_probability.sh', model_path]
            result = subprocess.run(command, capture_output=True, text=True)
            print("Script output:", result.stdout)
            if result.stderr:
                print("Script error:", result.stderr)

# ##############################################################################

def main():
    # Load the dataset from tokenized data if it exists
    train_path = os.path.join(dataset_path, "train")

    if os.path.exists(train_path) and "dataset_info.json" in os.listdir(train_path):
        lm_datasets = load_from_disk(dataset_path)
    else:
        dataset = load_dataset('wikipedia', '20220301.%s' % LANGUAGE)
        total_examples = len(dataset['train'])
        half_point = total_examples // 30
        part_train = dataset['train'].select(range(half_point))
        dataset['train'] = part_train
        tokenized_datasets = dataset.map(
            tokenize_function, batched=True, remove_columns=['id', 'url', 'title', 'text']
        )
        lm_datasets = tokenized_datasets.map(group_texts, batched=True)
        decoded_example = tokenizer.decode(lm_datasets["train"][1]["input_ids"])
        lm_datasets.save_to_disk(dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    samples = [lm_datasets["train"][i] for i in range(2)]
    for sample in samples:
        _ = sample.pop("word_ids")

    samples = [lm_datasets["train"][i] for i in range(2)]
    batch = whole_word_masking_data_collator(samples)

    test_size = int(0.1 * train_size)

    downsampled_dataset = lm_datasets["train"].train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(SAVE_MODEL_DIR, f"{model_name}-finetuned-mlm"),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.06,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # fp16=True,
        num_train_epochs=num_epochs,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=downsampled_dataset["train"],
        eval_dataset=downsampled_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        # callbacks=[LogProbabilityCallback()],
        compute_metrics=compute_metrics
    )

    trainer.train()
    

if __name__ == "__main__":
    main()