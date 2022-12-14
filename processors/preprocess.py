import os
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, AutoTokenizer, BertTokenizer
from datasets import load_dataset, load_metric, list_datasets
from BIONER_ER.config.model_config import model_checkpoint, FILE_NAME
from BIONER_ER.processors.utils_ner import BioNERProcessor
import pandas as pd
import numpy as np

from BIONER_ER.processors.utils_ner import DataProcessor, InputExample

task = "ner"  # 需要是"ner", "pos" 或者 "chunk"
batch_size = 16
processors = BioNERProcessor()

"""加载数据集和分词器"""
datasets = load_dataset("conll2003")
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)

# print(datasets["train"].features["ner_tags"])

label_list = datasets["train"].features["ner_tags"].feature.names
print(label_list)

labels_to_ids = {k: v for v, k in enumerate(sorted(label_list))}
ids_to_labels = {v: k for v, k in enumerate(sorted(label_list))}
print(labels_to_ids)

example = datasets["train"][3]
print(example)
# 实现 tokenization
tokenized_input = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
print(tokenized_input)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
print(tokens)

word_ids = tokenized_input.word_ids()
print(word_ids)

train_data = DataProcessor.read_tsv(FILE_NAME)
train_data2 = processors.get_train_examples("D:\pythonProject\BioBERT\datas\BC4CHEMD")
print(train_data[5])
print(train_data2[:5])


"""我们通常将特殊字符的label设置为-100，在模型中-100通常会被忽略掉不计算loss。
我们有两种对齐label的方式：
- 多个subtokens对齐一个word，对齐一个label
- 多个subtokens的第一个subtoken对齐word，对齐一个label，其他subtokens直接赋予-100.
我们提供这两种方式，通过`label_all_tokens = True`切换。
"""
label_all_tokens = True

""" 预处理数据
在将数据喂入模型之前，我们需要对数据进行预处理。预处理的工具叫`Tokenizer`。`Tokenizer`首先对输入进行tokenize，然后将tokens转化为预模型中需要对应的token ID，再转化为模型需要的输入格式。

为了达到数据预处理的目的，我们使用`AutoTokenizer.from_pretrained`方法实例化我们的tokenizer，这样可以确保：

- 我们得到一个与预训练模型一一对应的tokenizer。
- 使用指定的模型checkpoint对应的tokenizer的时候，我们也下载了模型需要的词表库vocabulary，准确来说是tokens vocabulary。

这个被下载的tokens vocabulary会被缓存起来，从而再次使用的时候不会重新下载。
"""

def dataset_feature_HuggingFace(examples):
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



#
# train_dataset = tokenize_and_align_labels(train_data)
# print(train_dataset[:5])

def align_label(texts, labels):
    # 首先tokenizer输入文本
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
  # 获取word_ids
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []
    # 采用上述的第一中方法来调整标签，使得标签与输入数据对其。
    for word_idx in word_ids:
        # 如果token不在word_ids内，则用 “-100” 填充
        if word_idx is None:
            label_ids.append(-100)
        # 如果token在word_ids内，且word_idx不为None，则从labels_to_ids获取label id
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        # 如果token在word_ids内，且word_idx为None
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids
# 构建自己的数据集类
class DataSequence(Dataset):
    def __init__(self, df):
        # 根据空格拆分labels
        lb = [i.split() for i in df['labels'].values.tolist()]
        # tokenizer 向量化文本
        txt = df['words'].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 512,
                                truncation=True, return_tensors="pt") for i in txt]
        # 对齐标签
        self.labels = [align_label(i, j) for i, j in zip(txt, lb)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels


class BioDataset(Dataset):
    def __init__(self, df):
        self.train_data = DataProcessor.read_tsv(FILE_NAME)

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels


dataset = BioDataset()
print(len(dataset))
print(dataset[50])
print(dataset[1:100])