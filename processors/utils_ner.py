import csv
import json
import os
import copy
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Union

import pandas as pd
import torch


class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, words, labels):
        self.guid = guid
        # list of words of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
        self.words = words
        # list of label sequence of the sentence,like: [B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=4, sort_keys=True) + "\n"
    # guid: str
    # words: List[str]
    # labels: Optional[List[str]]

class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_dict())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class Split(Enum):
    train = "train_dev"
    dev = "devel"
    test = "test"


# 读取本地数据集
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        lines = []
        with open(input_file, "r", encoding="utf-8-sig") as f:
            words = []
            labels = []
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            for line in reader:
                if len(line) == 0 or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    words.append(line[0])
                    if len(line) > 1:
                        labels.append(line[1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
            return lines

    @classmethod
    def read_text(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    @classmethod
    def read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label', None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                lines.append({"words": words, "labels": labels})
        return lines


class BioNERProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(self.read_tsv(os.path.join(data_dir, "train.tsv")))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(self.read_tsv(os.path.join(data_dir, "devel.tsv")))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(self.read_tsv(os.path.join(data_dir, "test.tsv")))

    def get_labels(self):
        """See base class."""
        return ["O", "CONT", "ORG", "LOC", 'EDU', 'NAME', 'PRO', 'RACE', 'TITLE']

    def create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            words = line["words"]
            labels = line["labels"]
            examples.append(InputExample(guid=guid, words=words, labels=labels))
        return examples


def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for index, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = index
            chunk[2] = index
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = index
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = index
            if index == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for index, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = index
            chunk[0] = tag.split('-')[1]
            chunk[2] = index
            if index == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = index

            if index == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, id2label, markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)


def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return S
