# importing libraries

from transformers import AutoTokenizer
from datasets import load_dataset
import evaluate
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments
import transformers
from datasets import Features, Value, Dataset, DatasetDict
import comet_ml
import comet_llm
import os
import numpy as np
import pickle
import json
import pandas as pd
import torch
from comet_ml import API


transformers.set_seed(35)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# setting comet api and workspace

api = API(api_key=os.environ["COMET_API_KEY"])
COMET_WORKSPACE = os.environ["COMET_WORKSPACE"]

# loads the data from the jsonl files
emotion_dataset_train = pd.read_json(path_or_buf="https://raw.githubusercontent.com/comet-ml/comet-llmops/main/data/merged_training_sample_prepared_train.jsonl", lines=True)
emotion_dataset_val_temp = pd.read_json(path_or_buf="https://raw.githubusercontent.com/comet-ml/comet-llmops/main/data/merged_training_sample_prepared_valid.jsonl", lines=True)

# takes first half of samples from emotion_dataset_val_temp and make emotion_dataset_val
emotion_dataset_val = emotion_dataset_val_temp.iloc[:int(len(emotion_dataset_val_temp)/2)]

# takes second half of samples from emotion_dataset_val_temp and make emotion_dataset_test
emotion_dataset_test = emotion_dataset_val_temp.iloc[int(len(emotion_dataset_val_temp)/2):]

sample = True

if sample == True:
    final_ds = DatasetDict({
        "train": Dataset.from_pandas(emotion_dataset_train.sample(50)),
        "validation": Dataset.from_pandas(emotion_dataset_val.sample(50)),
        "test": Dataset.from_pandas(emotion_dataset_test.sample(50))
    })
else:
    final_ds = DatasetDict({
        "train": Dataset.from_pandas(emotion_dataset_train),
        "validation": Dataset.from_pandas(emotion_dataset_val),
        "test": Dataset.from_pandas(emotion_dataset_test)
    })
    

# model checkpoint
model_checkpoint = "google/flan-t5-base"

# We'll create a tokenizer from model checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)

# We'll need padding to have same length sequences in a batch
tokenizer.pad_token = tokenizer.eos_token

# prefix
prefix_instruction = "Classify the provided piece of text into one of the following emotion labels.\n\nEmotion \
    labels: ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']"

# Define a tokenization function that first concatenates text and target
def tokenize_function(example):
    merged = prefix_instruction + "\n\n" + "Text: " + example["prompt"].strip("\n\n###\n\n") + "\n\n" + "\
        Emotion output:" + example["completion"].strip(" ").strip("\n")
    batch = tokenizer(merged, padding='max_length', truncation=True)
    batch["labels"] = batch["input_ids"].copy()
    return batch

# Apply it on our dataset, and remove the text columns
tokenized_datasets = final_ds.map(tokenize_function, remove_columns=["prompt", "completion"])

# initialize comet_ml
comet_ml.init(project_name="emotion-classification")

# training an autoregressive language model from a pretrained checkpoint
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

# set this to log HF results and assets to Comet
os.environ["COMET_LOG_ASSETS"] = "True"

# HF Trainer
model_name = model_checkpoint.split("/")[-1]
training_args = Seq2SeqTrainingArguments(
    num_train_epochs=1,
    output_dir="./results",
    overwrite_output_dir=True,
    logging_steps=1,
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    weight_decay=0.01,
    save_total_limit=5,
    save_steps=7,
    auto_find_batch_size=True
)

# instantiate HF Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# run trainer
trainer.train()

# save the model
trainer.save_model("./results")