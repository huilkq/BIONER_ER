import argparse
import os
import torch
from transformers.modeling_bert import BertForTokenClassification

import config
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, Trainer, AutoConfig

from BIONER_ER.config.config import Arguments
from BIONER_ER.config.model_config import MODEL_BIOBERT, FILE_NAME
from BIONER_ER.models.bert_models import BertCrfNer
from BIONER_ER.processors.data_loader import Split
from BIONER_ER.processors.preprocess import BioDataset


# 训练函数
def train(model, train_loader, optimizer, scheduler, device):
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
    total_acc_train = 0
    total_loss_train = 0
    total_iter_train = len(train_loader)
    # 按批量循环训练模型
    for idx, batch in tqdm(train_loader):
        # 从train_data中获取mask和input_id
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label_ids'].to(device)  # shape: [32, 64]
        # 梯度清零！！
        optimizer.zero_grad()
        # 输入模型训练结果：损失及分类概率
        loss, logits = model(input_ids, attention_mask=attention_mask, labels_ids=labels)
        """
            过滤掉特殊token及padding的token
            logits_clean = logits[0][train_label != -100]
            label_clean = train_label[train_label != -100]
            """
        # 获取最大概率值
        predictions = logits.argmax(dim=1)
        # 计算准确率
        acc = (predictions == labels).float().mean()
        total_acc_train += acc
        total_loss_train += loss.item()
        # 反向传递
        loss.backward()
        # 参数更新
        optimizer.step()
        scheduler.step()
    # 计算一个epoch在训练集上的损失和精度
    train_accuracy = total_acc_train / total_iter_train
    train_loss = total_loss_train / total_iter_train
    print("Accuracy: %.4f" % train_accuracy)
    print("Average testing loss: %.4f" % train_loss)


"""
准确率: accuracy = 预测对的元素个数/总的元素个数
查准率：precision = 预测正确的实体个数 / 预测的实体总个数
召回率：recall = 预测正确的实体个数 / 标注的实体总个数
F1值：F1 = 2 *准确率 * 召回率 / (准确率 + 召回率)
"""


def evaluate(model, eval_loader, device):
    """

    :param model: 网络模型
    :param eval_loader: 评估数据集
    :param device: 设备：cpu&&gpu
    :return:
    """
    model.eval()
    total_acc_eval = 0
    total_loss_eval = 0
    total_iter_eval = len(eval_loader)
    for batch in eval_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels_ids'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs[1]

        total_loss_eval += loss.item()
        total_acc_eval += (logits.argmax(dim=1).data == labels.data).float().mean().item()
    # 计算一个epoch在训练集上的损失和精度
    val_accuracy = total_acc_eval / total_iter_eval
    val_loss = total_loss_eval / total_iter_eval
    print("Accuracy: %.4f" % val_accuracy)
    print("Average testing loss: %.4f" % val_loss)
    print("-------------------------------")


def save_pretrained(model, path):
    # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))


def main():
    # 训练参数
    arg = Arguments.get_parser()
    # 标签
    labels = ["O", "B-Chemical", "I-Chemical"]
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    bert_config = AutoConfig.from_pretrained(
        MODEL_BIOBERT
        # num_labels=len(labels),
        # id2label=label_map,
        # label2id={label: i for i, label in enumerate(labels)},
        # cache_dir=arg.cache_dir,
    )
    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BIOBERT)
    # 调用gpu
    device = torch.device("cuda")
    # 模型
    model_bert = BertForTokenClassification.from_pretrained(MODEL_BIOBERT)
    model_bert_crf = BertCrfNer(bert_config).to(device)

    # 定义训练和验证集数据
    train_dataset = BioDataset(
        data_dir=FILE_NAME,
        tokenizer=tokenizer,
        labels=labels,
        max_seq_length=256,
        mode=Split.train
    )
    dev_dataset = BioDataset(
        data_dir=FILE_NAME,
        tokenizer=tokenizer,
        labels=labels,
        max_seq_length=256,
        mode=Split.dev
    )
    # 批量获取训练和验证集数据
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=4, shuffle=True)
    dev_dataloader = DataLoader(dataset=dev_dataset, num_workers=4, )

    # 定义优化器
    optimizer_SGD = SGD(model_bert_crf.parameters(), lr=arg.lr)  # SGD
    optimizer_AdamW = AdamW(model_bert_crf.parameters(), lr=arg.lr, eps=1e-6)  # AdamW
    # 交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss(ignore_index=100)

    # Warmup学习率预热
    len_dataset = len(train_dataset)
    total_steps = (len_dataset // arg.batch_size) * arg.epochs if len_dataset % arg.batch_size == 0 else (len_dataset // arg.batch_size + 1) * arg.epochs

    scheduler = get_linear_schedule_with_warmup(optimizer_AdamW, num_warmup_steps=arg.warm_up_ratio * total_steps,
                                                num_training_steps=total_steps)

    print('Start Train...,')
    for epoch in range(1, arg.epochs + 1):
        train(model_bert_crf, train_dataloader, optimizer_AdamW, scheduler, device)
        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists(arg.logdir): os.makedirs(arg.logdir)
        fname = os.path.join(arg.logdir, str(epoch))
        eval(model_bert_crf, dev_dataloader, device)

        torch.save(model_bert_crf.state_dict(), f"{fname}.pt")
        print(f"weights were saved to {fname}.pt")


if __name__ == "__main__":
    main()
