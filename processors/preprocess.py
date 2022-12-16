import os
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, AutoTokenizer, BertTokenizer
from datasets import load_dataset, load_metric, list_datasets
from BIONER_ER.config.model_config import model_checkpoint, FILE_NAME, MODEL_BIOBERT
from BIONER_ER.processors.utils_ner import BioNERProcessor, Split, convert_examples_to_features, InputFeatures, \
    get_labels
import pandas as pd
import numpy as np
from BIONER_ER.processors.utils_ner import DataProcessor, InputExample

task = "ner"  # 需要是"ner", "pos" 或者 "chunk"
batch_size = 16
processors = BioNERProcessor()

"""加载数据集和分词器"""
# datasets = load_dataset("conll2003")
datasets = processors.get_train_examples(FILE_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_BIOBERT)
lables = ["O", "B-Chemical", "I-Chemical"]
print(lables)
# print(datasets["train"].features["ner_tags"])

# label_list = datasets["train"].features["ner_tags"].feature.names
# print(label_list)

# labels_to_ids = {k: v for v, k in enumerate(sorted(label_list))}
# ids_to_labels = {v: k for v, k in enumerate(sorted(label_list))}
# print(labels_to_ids)

example = datasets[:7]
print(example)
# 实现 tokenization
# tokenized_input = tokenizer.tokenize(example["tokens"])
# tokenized_input = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
# print(tokenized_input)
# word_ids = tokenized_input.word_ids()
# print(word_ids)


label_all_tokens = True

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["words"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_data = convert_examples_to_features(example, lables, 20, tokenizer)
print(tokenized_data[6])
tokens = tokenizer.convert_ids_to_tokens(tokenized_data[6].input_ids)
print(tokens)
#
# train_data = DataProcessor.read_tsv(FILE_NAME+"\\train.tsv")
# train_data2 = processors.get_train_examples(FILE_NAME)
# print(train_data[:5])
# print(train_data2[:5])


# 构建自己的数据集类
class BioDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            tokenizer: BertTokenizerFast,
            labels: List[str],
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        examples = processors.get_examples(data_dir, mode)
        self.features = convert_examples_to_features(
            examples,
            labels,
            max_seq_length,
            tokenizer
        )
        # logger.info(f"Saving features into cached file {cached_features_file}")
        # torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


# features = convert_examples_to_features(
#     examples,
#     labels,
#     max_seq_length,
#     tokenizer
#     )
# dataset = BioDataset(FILE_NAME)
# print(len(dataset))
# print(dataset[50])
# print(dataset[1:100])