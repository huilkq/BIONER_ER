import os
import torch

import config
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score, accuracy_score, precision_score, recall_score
from typing import Dict
from transformers import AutoTokenizer, BertForTokenClassification, get_linear_schedule_with_warmup, Trainer, \
    AutoConfig, BertTokenizerFast

from BIONER_ER.config import model_config, config
from BIONER_ER.models.bert_models import BertCrfNer, Bert_NER
from BIONER_ER.processors.preprocess import BioDataset, data_collator


# 训练函数
def bioner_train(model, train_loader, optimizer, scheduler, device):
    """
    :param model: 网络模型
    :param train_loader: 训练数据集
    :param optimizer: 优化器
    :param scheduler: 对优化器的学习率进行调整
    :param device: 训练设备
    :return:
    """
    # 训练模型
    model.train()
    # total_acc_train = 0
    total_loss_train = 0
    total_pre_train = 0
    total_accuracy_train = 0
    total_recall_train = 0
    total_f1_train = 0
    total_iter_train = len(train_loader)
    # 按批量循环训练模型
    for idx, batch in enumerate(train_loader):
        # 从train_data中获取mask和input_id
        input_ids = batch['input_ids'].to(device)  # shape: [batch_size, max_seq_length]
        attention_mask = batch['attention_mask'].to(device)  # shape: [batch_size, max_seq_length]
        labels = batch['label_ids'].to(device)  # shape: [batch_size, max_seq_length]
        # 梯度清零！！
        optimizer.zero_grad()
        # 输入模型训练结果：损失及分类概率
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]  # shape: [batch_size, max_seq_length, num_labels]
        # 清楚无效token对应的结果
        logits_clean = logits[labels != -100]
        label_clean = labels[labels != -100]
        # 获取最大概率值
        preds = logits.argmax(dim=2)  # shape[batch_size, max_seq_length]
        predictions = logits_clean.argmax(dim=1)  # shape:[batch_size*max_seq_length]
        result = metrics(predictions, label_clean)
        # 计算准确率
        # acc = (predictions == label_clean).float().mean()
        # if idx % 50 == 0:  # 看模型的准确率
        #     with torch.no_grad():
        #         # 假如输入的是64个字符，64 * 7
        #         print((predictions == label_clean).float().mean().item(), loss.item())
        total_pre_train += result["precision"]
        total_recall_train += result["recall"]
        total_accuracy_train += result["accuracy"]
        total_f1_train += result["f1"]
        # total_acc_train += acc
        total_loss_train += loss.item()
        # 反向传播，累加梯度
        loss.backward()
        # 解决梯度爆炸问题
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 参数更新
        optimizer.step()
        scheduler.step()
        # if ((idx+1) % 200 == 0):
        #     print("iter_num: %d, loss: %.4f, Accuracy：%.4f%% 进度：%.4f%%" % (
        #         idx, loss.item(), (total_acc_train / idx) * 100, idx / total_iter_train * 100))

    # 计算一个epoch在训练集上的损失和精度
    # train_acc = total_acc_train / total_iter_train
    train_accuracy = total_accuracy_train / total_iter_train
    train_pre = total_pre_train / total_iter_train
    train_recall = total_recall_train / total_iter_train
    train_f1 = total_f1_train / total_iter_train
    train_loss = total_loss_train / total_iter_train
    print(
        f'''Loss:{train_loss: .4f} | Accuracy: {train_accuracy: .4f} | Precision:{train_pre: .4f} | Recall:{train_recall: .4f} | F1:{train_f1: .4f} ''')


"""
准确率: accuracy = 预测对的元素个数/总的元素个数
查准率：precision = 预测正确的实体个数 / 预测的实体总个数
召回率：recall = 预测正确的实体个数 / 标注的实体总个数
F1值：F1 = 2 *准确率 * 召回率 / (准确率 + 召回率)
"""


# 评估函数
def bioner_evaluate(model, eval_loader, device):
    """
    :param model: 网络模型
    :param eval_loader: 评估数据集
    :param device: 设备：cpu&&gpu
    :return:
    """
    model.eval()
    # total_acc_eval = 0
    total_loss_eval = 0
    total_pre_eval = 0
    total_accuracy_eval = 0
    total_recall_eval = 0
    total_f1_eval = 0
    total_iter_eval = len(eval_loader)
    for idx, batch in enumerate(eval_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label_ids'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs[1]
        # 清楚无效token对应的结果
        logits_clean = logits[labels != -100]
        label_clean = labels[labels != -100]
        # 获取概率值最大的预测
        predictions = logits_clean.argmax(dim=1)
        result = metrics(predictions, label_clean)
        # 计算精度
        # acc = (predictions == label_clean).float().mean()
        total_pre_eval += result["precision"]
        total_recall_eval += result["recall"]
        total_accuracy_eval += result["accuracy"]
        total_f1_eval += result["f1"]
        total_loss_eval += loss.item()
        # total_acc_eval += acc
    # 计算一个epoch在验证集上的损失和精度
    # val_acc = total_acc_eval / total_iter_eval
    val_accuracy = total_accuracy_eval / total_iter_eval
    val_pre = total_pre_eval / total_iter_eval
    val_recall = total_recall_eval / total_iter_eval
    val_f1 = total_f1_eval / total_iter_eval

    val_loss = total_loss_eval / total_iter_eval
    # print("Accuracy testing: %.4f" % val_acc)
    print(f'''Loss:{val_loss: .4f} | Accuracy:{val_accuracy: .4f} | Precision:{val_pre: .4f} | Recall:{val_recall: .4f} | F1:{val_f1: .4f} ''')


# 评估指标
def metrics(predictions, labels):
    pre_convert = []
    label_convert = []
    pre_list = []
    label_list = []
    for prediction, label in zip(predictions, labels):
        if label >= 0:
            pre_convert.extend([config.ids_to_labels[prediction.item()]])
            label_convert.extend([config.ids_to_labels[label.item()]])
    pre_list.append(pre_convert)
    label_list.append(label_convert)
    return {
        "accuracy": accuracy_score(label_list, pre_list),
        "precision": precision_score(label_list, pre_list),
        "recall": recall_score(label_list, pre_list),
        "f1": f1_score(label_list, pre_list),
    }


def save_pretrained(model, path):
    # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))


def main():
    # 标签+
    label_map: Dict[int, str] = {i: label for i, label in enumerate(config.labels)}
    # 参数
    bert_config = AutoConfig.from_pretrained(
        model_config.MODE_BIOBERT,
        num_labels=len(config.labels),
        id2label=label_map,
        label2id={label: i for i, label in enumerate(config.labels)},
    )
    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(model_config.MODE_BIOBERT)
    # 模型
    model_bert = BertForTokenClassification.from_pretrained(model_config.MODE_BIOBERT,
                                                            num_labels=len(config.labels_JNLPBA)).to(config.device)
    # model_bert_crf = BertCrfNer.from_pretrained(model_config.MODE_BIOBERT, config=bert_config).to(config.device)

    # 定义训练和验证集数据
    train_dataset = BioDataset(
        data_dir=model_config.FILE_NAME,
        tokenizer=tokenizer,
        labels=config.labels,
        max_seq_length=config.max_seq_length,
        data_type='train'
    )
    dev_dataset = BioDataset(
        data_dir=model_config.FILE_NAME,
        tokenizer=tokenizer,
        labels=config.labels,
        max_seq_length=config.max_seq_length,
        data_type='dev'
    )
    # 批量获取训练和验证集数据
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, collate_fn=data_collator)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=32, shuffle=True, collate_fn=data_collator)

    # 定义优化器
    optimizer_SGD = SGD(model_bert.parameters(), lr=config.lr)  # SGD
    optimizer_AdamW = AdamW(model_bert.parameters(), lr=config.lr)  # AdamW
    # 交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    #  Warmup学习率预热
    len_dataset = len(train_dataset)
    total_steps = (len_dataset // config.batch_size) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer_AdamW, num_warmup_steps=config.warm_up_ratio * total_steps,
                                                num_training_steps=total_steps)
    # total_steps = len(train_dataset) * 1
    # scheduler = get_linear_schedule_with_warmup(optimizer_AdamW, num_warmup_steps=0,
    #                                             num_training_steps=total_steps)  # Default value in run_glue.py

    print('Start Train...,')
    for epoch in range(1, config.epochs + 1):
        print(f"=========train at epoch={epoch}=========")
        bioner_train(model_bert, train_dataloader, optimizer_AdamW, scheduler, config.device)
        print(f"=========eval at epoch={epoch}=========")
        bioner_evaluate(model_bert, dev_dataloader, config.device)
        # print("accuracy  precision  recall  f1  loss")


if __name__ == "__main__":
    main()
