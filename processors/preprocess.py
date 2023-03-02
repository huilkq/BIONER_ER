import copy
import json
import os
from typing import List, Optional, Dict

import torch
import logging

from filelock import FileLock
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer

from BIONER_ER.processors.data_loader import DataProcessor

logger = logging.getLogger()


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, input_len, token_type_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


# 把数据集转换成bert需要的格式
def convert_examples_to_features_plus(examples, label_list, max_seq_length, tokenizer,
                                      cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=0,
                                      sep_token="[SEP]", sep_token_extra=False, pad_on_left=False, pad_token=0,
                                      pad_token_segment_id=0, pad_token_label_id=-100,
                                      sequence_a_segment_id=0, mask_padding_with_zero=True, ) -> List[InputFeatures]:
    """
        Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # 对单词的第一个标记使用实际标签id，对其余的使用标记填充id
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        # 打印前五个example
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None
        # 以InputFeatures对象存储example
        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, input_len=input_len,
                token_type_ids=segment_ids, label_ids=label_ids))
    return features


# 把数据集转换成bert需要的格式
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                 sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                 sequence_a_segment_id=0, mask_padding_with_zero=True, ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        # if isinstance(example.words, list):
        #     example.words = " ".join(example.words)
        # tokens = tokenizer.tokenize(example.words)
        # label_ids = [label_map[x] for x in example.labels]
        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # 对单词的第一个标记使用实际标签id，对其余的使用标记填充id
                label_ids.extend([label_map[label]] + [label_map[label]] * (len(word_tokens) - 1))
        # Account for [CLS] and [SEP] with "- 2".
        # special_tokens_count = tokenizer.num_special_tokens_to_add()
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0

        tokens += [sep_token]
        label_ids += [label_map['O']]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label_map['O']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map['O']] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, input_len=input_len,
                token_type_ids=segment_ids, label_ids=label_ids))
    return features


# 构建自己的数据集

def load_and_cache_examples(
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        data_type='train'
):
    """
        :param data_dir: 数据存放路径
        :param tokenizer: Tokenizer的实例化对象
        :param labels: 标签列表
        :param max_seq_length: 输入序列的最大的长度
        :param overwrite_cache:
        :param data_type: train/dev/test 模式
        """
    # 从缓存或数据集文件加载数据集文件
    cached_features_file = os.path.join(
        data_dir, "cached_{}_{}_{}".format(data_type, tokenizer.__class__.__name__, str(max_seq_length)),
    )
    # 确保分布式训练中只有第一个进程处理数据集，
    # 其他人将使用缓存.
    lock_path = cached_features_file + ".lock"
    with FileLock(lock_path):

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            # 从缓存中加载数据集
            features = torch.load(cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {data_dir}")
            if data_type == 'train':
                examples = processors.get_train_examples(data_dir)
            elif data_type == 'dev':
                examples = processors.get_dev_examples(data_dir)
            else:
                examples = processors.get_test_examples(data_dir)
            # TODO clean up all this to leverage built-in features of tokenizers
            features = convert_examples_to_features(
                examples,
                labels,
                max_seq_length,
                tokenizer
            )
            logger.info(f"Saving features into cached file {cached_features_file}")
            # 保存处理好的数据集到缓存中
            torch.save(features, cached_features_file)

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, all_lens, label_ids)
    return dataset


# 转换成tensor
def data_collator(features: List[InputFeatures]) -> Dict[str, torch.Tensor]:
    first = features[0]
    batch = {}
    for k, v in first.__dict__.items():
        if k == 'metadata':
            batch[k] = [f.__dict__[k] for f in features]
        else:
            batch[k] = torch.tensor([f.__dict__[k] for f in features], dtype=torch.long)
    return batch


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    input_ids, attention_mask, token_type_ids, all_lens, label_ids = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]
    token_type_ids = token_type_ids[:, :max_len]
    label_ids = label_ids[:, :max_len]
    return input_ids, attention_mask, token_type_ids, label_ids, all_lens


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


processors = BioNERProcessor()
