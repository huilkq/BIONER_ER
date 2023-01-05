import os
from typing import List, Optional

import numpy as np
import torch
import logging

from filelock import FileLock
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoTokenizer, BertTokenizer, PreTrainedTokenizer
from BIONER_ER.config.model_config import model_checkpoint, FILE_NAME, MODEL_BIOBERT
from BIONER_ER.processors.data_loader import BioNERProcessor, Split,  InputFeatures, get_labels, InputExample

logger = logging.getLogger()
task = "ner"  # 需要是"ner", "pos" 或者 "chunk"
batch_size = 16
processors = BioNERProcessor()

"""加载数据集和分词器"""
# datasets = load_dataset("conll2003")
tokenizer = AutoTokenizer.from_pretrained(MODEL_BIOBERT)
lables = ["O", "B-Chemical", "I-Chemical"]

# labels_to_ids = {k: v for v, k in enumerate(sorted(label_list))}
# ids_to_labels = {v: k for v, k in enumerate(sorted(label_list))}
# print(labels_to_ids)

# example = datasets[:7]
# print(example)


# 把数据集转换成bert需要的格式
"""
examples: InputExample的实例对像
label_list: label的列表
max_seq_length: 输入序列的最大的长度
tokenizer: Tokenizer的实例化对象
后面的参数一般固定不用动
"""
def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
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
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
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
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids
            )
        )
    return features


# 构建自己的数据集类
class BioDataset(Dataset):

    features: List[InputFeatures]
    """
    data_dir: 数据存放路径
    tokenizer: Tokenizer的实例化对象
    labels: label的列表
    max_seq_length: 输入序列的最大的长度
    overwrite_cache: 
    mode: train/dev/test 模式
    """
    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        # 从缓存或数据集文件加载数据集文件
        cached_features_file = os.path.join(
            data_dir, "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
        )
        # 确保分布式训练中只有第一个进程处理数据集，
        # 其他人将使用缓存.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                # 从缓存中加载数据集
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                examples = processors.get_examples(data_dir, mode)
                # TODO clean up all this to leverage built-in features of tokenizers
                self.features = convert_examples_to_features(
                    examples,
                    labels,
                    max_seq_length,
                    tokenizer
                )
                logger.info(f"Saving features into cached file {cached_features_file}")
                # 保存处理好的数据集到缓存中
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        item = self.features[i]
        return item

# def coffate_fn():


# 開始訓練
if __name__ == "__main__":
    device = torch.device("cuda")
    train_datasets = BioDataset(
        data_dir=FILE_NAME,
        tokenizer=tokenizer,
        labels=lables,
        max_seq_length=50,
        mode=Split.train
    )
    print(train_datasets[1])
    # train_dataloader = DataLoader(dataset=train_datasets, num_workers=4,  shuffle=True)
    # features, targets = next(iter(train_dataloader))  # 从dataloader中取出一个batch
    # print(features.shape)
    # print(targets.shape)
    # print(targets)
    # for i, train_data in enumerate(train_dataloader):
    #     train_label = train_data['label_ids']
    #     mask = train_data['attention_mask']
    #     input_id = train_data['input_ids']
    #     print(input_id.shape, train_label)