from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
import os
import evaluate
import numpy as np

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

model_name = MODEL_SELECTION.split("/")[-1]

num_epochs = 4
batch_size = 8
train_size = 1_000

# ##############################################################################

def load_model_and_tokenizer(model_name, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, output_hidden_states=False,output_attentions=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def prepare_data(tokenizer, dataset: Dataset | DatasetDict):
    def tokenize(examples):
        return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', return_tensors='pt', max_length=64)
    
    dataset = dataset.map(tokenize, batched=True)

    if 'token_type_ids' in dataset.column_names:
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
    else:
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
    return dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels),
        "precision": precision_metric.compute(predictions=predictions, references=labels, average='macro'),
        "recall": recall_metric.compute(predictions=predictions, references=labels, average='macro')
    }

class DebugTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        outputs = super().prediction_step(model, inputs, prediction_loss_only=False, ignore_keys=None)
        _, logits, labels = outputs
        predictions = np.argmax(logits, axis=-1)
        return outputs

# ##############################################################################

# Load XNLI dataset
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

dataset = load_dataset("xnli", LANGUAGE)

test_size = int(0.1 * train_size)

downsampled_dataset = dataset["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

num_labels = len(set(downsampled_dataset['train']['label']))
model, tokenizer = load_model_and_tokenizer(MODEL_SELECTION, num_labels)

# Prepare data for training and evaluation
downsampled_dataset = downsampled_dataset.map(lambda example: {
    'labels': example['label'], 
    **{k: v for k, v in example.items() if k != 'label'}
    }, remove_columns=['label'])

train_dataset = prepare_data(tokenizer, downsampled_dataset['train'])
eval_dataset = prepare_data(tokenizer, downsampled_dataset['test'])

# Define training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(SAVE_MODEL_DIR, f"{model_name}-finetuned-xnli"),
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
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

# Initialize the Trainer
trainer = DebugTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()