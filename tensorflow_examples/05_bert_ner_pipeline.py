# named entity recognition using BERT from Hugging Face

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, DataCollatorForTokenClassification, create_optimizer
#from datasets import load_dataset, load_metric
from datasets import load_dataset
import evaluate
from transformers import TrainingArguments
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

metric = evaluate.load("seqeval")

# Load dataset
dataset = load_dataset("conll2003")
label_list = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_list)

# Load tokenizer
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenize and align labels
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    #tokenized_inputs["labels"] = list(map(int, labels))
    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["ner_tags"][word_idx])
        else:
            labels.append(-100)
        previous_word_idx = word_idx
    #tokenized_inputs["labels"] = labels
    tokenized_inputs["labels"] = list(map(int, labels))
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
tokenized_datasets.set_format("tensorflow")

# Convert to tf.data.Dataset
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")

train_features = {x: tokenized_datasets["train"][x] for x in tokenizer.model_input_names}
train_tfdataset = tf.data.Dataset.from_tensor_slices((train_features, tokenized_datasets["train"]["labels"]))
train_tfdataset = train_tfdataset.shuffle(1000).batch(16).prefetch(tf.data.AUTOTUNE)

val_features = {x: tokenized_datasets["validation"][x] for x in tokenizer.model_input_names}
val_tfdataset = tf.data.Dataset.from_tensor_slices((val_features, tokenized_datasets["validation"]["labels"]))
val_tfdataset = val_tfdataset.batch(16).prefetch(tf.data.AUTOTUNE)

# Load model
model = TFAutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# Optimizer
num_train_steps = len(train_tfdataset) * 3
optimizer, schedule = create_optimizer(init_lr=2e-5, num_train_steps=num_train_steps, num_warmup_steps=0)

# Compile model
model.compile(optimizer=optimizer, loss=model.compute_loss)

# Train
model.fit(train_tfdataset, validation_data=val_tfdataset, epochs=3)

# Save model
model.save_pretrained("./ner-bert-tf")
tokenizer.save_pretrained("./ner-bert-tf")

