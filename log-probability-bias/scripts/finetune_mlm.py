from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, default_data_collator
from datasets import load_dataset, load_from_disk
from transformers import TrainerCallback
import os
import collections
import numpy as np
import math
import subprocess
import torch
import evaluate

# ##############################################################################

BASE_DIR = "/home/viktorija/bakalaurinis/log-probability-bias"
MODEL_CHECKPOINT = "camembert-base"
SOURCE_MODEL_DIR = os.path.join(BASE_DIR, "../models/camembert")
SAVE_MODEL_DIR = os.path.join(BASE_DIR, "../models")

# MODEL_SELECTION = MODEL_CHECKPOINT
MODEL_SELECTION = SOURCE_MODEL_DIR

dataset_path = os.path.join(BASE_DIR, "tokenized")
model_name = MODEL_SELECTION.split("/")[-1]
chunk_size = 128
wwm_probability = 0.2
batch_size = 8

# ##############################################################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_SELECTION)
model = AutoModelForMaskedLM.from_pretrained(MODEL_SELECTION)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

def tokenize_function(examples):
    # examples["text"] = map(lambda x: x.replace("\n", " "), examples["text"])
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
        dataset = load_dataset('wikipedia', '20220301.fr')
        total_examples = len(dataset['train'])
        half_point = total_examples // 30
        part_train = dataset['train'].select(range(half_point))
        dataset['train'] = part_train

        print(dataset)
        print(dataset['train'][0])

        tokenized_datasets = dataset.map(
            tokenize_function, batched=True, remove_columns=['id', 'url', 'title', 'text']
        )
        print("Tokenization completed")
        print(tokenized_datasets)

        lm_datasets = tokenized_datasets.map(group_texts, batched=True)
        decoded_example = tokenizer.decode(lm_datasets["train"][1]["input_ids"])
        print("Grouping completed")
        print(decoded_example)
        print(lm_datasets)

        lm_datasets.save_to_disk(dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    samples = [lm_datasets["train"][i] for i in range(2)]
    for sample in samples:
        _ = sample.pop("word_ids")

    for chunk in data_collator(samples)["input_ids"]:
        print(f"\n'>>> {tokenizer.decode(chunk)}'")

    samples = [lm_datasets["train"][i] for i in range(2)]
    batch = whole_word_masking_data_collator(samples)

    for chunk in batch["input_ids"]:
        print(f"\n'>>> {tokenizer.decode(chunk)}'")


    train_size = 10_000
    test_size = int(0.1 * train_size)

    downsampled_dataset = lm_datasets["train"].train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )
    print(downsampled_dataset)

    metric = evaluate.load("accuracy")

    training_args = TrainingArguments(
        output_dir=os.path.join(SAVE_MODEL_DIR, f"{model_name}-finetuned-mlm"),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # fp16=True,
        num_train_epochs=3,
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
        compute_metrics=metric.compute
    )

    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.train()

    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    

if __name__ == "__main__":
    main()