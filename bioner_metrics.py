from collections import Counter
from processors.utils_ner import get_entities


class SeqEntityScore(object):
    """
    准确率: accuracy = 预测对的元素个数/总的元素个数
    查准率：precision = 预测正确的实体个数 / 预测的实体总个数
    召回率：recall = 预测正确的实体个数 / 标注的实体总个数
    F1值：F1 = 2 *准确率 * 召回率 / (准确率 + 召回率)
    """
    def __init__(self, id2label, markup='bio'):
        self.id2label = id2label
        self.markup = markup
        self.reset()

    def reset(self):
        self.golds = []
        self.predicts = []
        self.corrects = []

    def compute(self, predict_num, correct_num, gold_num):
        """

        :param predict_num: 预测的数量
        :param correct_num: 预测准确的数量
        :param gold_num: 金标数量
        :return:
        """
        precision = 0 if predict_num == 0 else (correct_num / predict_num)
        recall = 0 if gold_num == 0 else (correct_num / gold_num)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return precision, recall,  f1

    def result(self):
        class_info = {}
        gold_counter = Counter([x[0] for x in self.golds])
        predict_counter = Counter([x[0] for x in self.predicts])
        correct_counter = Counter([x[0] for x in self.corrects])
        for type_, count in gold_counter.items():
            gold = count
            predict = predict_counter.get(type_, 0)
            correct = correct_counter.get(type_, 0)
            precision, recall, f1 = self.compute(predict, correct, gold)
            class_info[type_] = {"precision": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        gold = len(self.golds)
        predict = len(self.predicts)
        correct = len(self.corrects)
        precision, recall, f1 = self.compute(predict, correct, gold)
        return {'precision': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_paths, pred_paths):
        """
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]
        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        """
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path, self.id2label)
            pre_entities = get_entities(pre_path, self.id2label)
            self.golds.extend(label_entities)
            self.predicts.extend(pre_entities)
            self.corrects.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])


def split_entity(label_sequence):
    entity_mark = dict()
    entity_pointer = None
    for index, label in enumerate(label_sequence):
        if label.startswith('B'):
            category = label.split('-')[1]
            entity_pointer = (index, category)
            entity_mark.setdefault(entity_pointer, [label])
        elif label.startswith('I'):
            if entity_pointer is None: continue
            if entity_pointer[1] != label.split('-')[1]: continue
            entity_mark[entity_pointer].append(label)
        else:
            entity_pointer = None
    return entity_mark


def evaluate(real_label, predict_label):
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