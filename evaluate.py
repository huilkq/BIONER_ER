import os

import numpy as np
import torch

def split_entity(label_sequence):
    entity_mark = dict()
    entity_pointer = None
    for index, label in enumerate(label_sequence):
        if label.startswith('B'):
            category = label.split('-')[1]
            entity_pointer = (index, category)
            entity_mark.setdefault(entity_pointer, [label])
        elif label.startswith('I'):
            if entity_pointer is None:
                continue
            if entity_pointer[1] != label.split('-')[1]:
                continue
            entity_mark[entity_pointer].append(label)
        else:
            entity_pointer = None
    return entity_mark


def evaluate(real_label, predict_label):
    # 序列标注的准确率和召回率计算，详情查看：https://zhuanlan.zhihu.com/p/56582082
    real_entity_mark = split_entity(real_label)
    predict_entity_mark = split_entity(predict_label)

    true_entity_mark = dict()
    key_set = real_entity_mark.keys() & predict_entity_mark.keys()
    for key in key_set:
        real_entity = real_entity_mark.get(key)
        predict_entity = predict_entity_mark.get(key)
        if tuple(real_entity) == tuple(predict_entity):
            true_entity_mark.setdefault(key, real_entity)

    real_entity_num = len(real_entity_mark)
    predict_entity_num = len(predict_entity_mark)
    true_entity_num = len(true_entity_mark)

    precision = true_entity_num / predict_entity_num
    recall = true_entity_num / real_entity_num
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


