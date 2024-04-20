from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os
import evaluate
import numpy as np

# ##############################################################################

BASE_DIR = "/home/viktorija/bakalaurinis/log-probability-bias"
MODEL_CHECKPOINT = "camembert-base"
SOURCE_MODEL_DIR = os.path.join(BASE_DIR, "../models/camembert")
SAVE_MODEL_DIR = os.path.join(BASE_DIR, "../models")

# MODEL_SELECTION = MODEL_CHECKPOINT
MODEL_SELECTION = SOURCE_MODEL_DIR

model_name = MODEL_SELECTION.split("/")[-1]
batch_size = 8

# ##############################################################################

def load_model_and_tokenizer(model_name, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def prepare_data(tokenizer, dataset):
    def tokenize(examples):
        return tokenizer(examples['premise'], examples['hypothesis'], padding=True, truncation=True, return_tensors='pt', max_length=256)
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# ##############################################################################

# Load XNLI dataset
metric = evaluate.load("accuracy")
dataset = load_dataset("xnli", "fr")

train_size = 1_000
test_size = int(0.1 * train_size)

downsampled_dataset = dataset["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
print(downsampled_dataset['train'][:4])

num_labels = len(set(downsampled_dataset['train']['label']))
print(f"Number of labels: {num_labels}")
model, tokenizer = load_model_and_tokenizer(MODEL_SELECTION, num_labels)

# Prepare data for training and evaluation
train_dataset = prepare_data(tokenizer, downsampled_dataset['train'])
eval_dataset = prepare_data(tokenizer, downsampled_dataset['test'])

# Define training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(SAVE_MODEL_DIR, f"{model_name}-finetuned-xnli"),
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

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
# print(f"Accuracy on XNLI task: {results['eval_accuracy']}") nera tokio eval_accuracy
