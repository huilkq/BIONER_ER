import logging
import os

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.optim import AdamW
from tqdm import tqdm

import config
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from typing import Dict
from transformers import AutoTokenizer, BertForTokenClassification, get_linear_schedule_with_warmup, AutoConfig

from BIONER_ER.bioner_metrics import SeqEntityScore
from BIONER_ER.config import model_config, config
from BIONER_ER.models.models_ner import BertCRF, BertBiLSTMCRF
from BIONER_ER.processors.preprocess import load_and_cache_examples, collate_fn, data_collator


# 训练函数
def bioner_train(model, train_loader, optimizer, scheduler, device):
    """
    :param model: 网络模型
    :param train_loader: 训练数据集
    :param optimizer: 优化器
    :param scheduler: 对优化器的学习率进行调整
    :param device: 训练设备
    :return
    """
    # 训练模型
    metric = SeqEntityScore(config.ids_to_labels, markup='bio')
    model.train()
    total_loss_train, total_pre_train, total_recall_train, total_f1_train = 0, 0, 0, 0
    # 按批量循环训练模型
    for idx, batch in enumerate(tqdm(train_loader)):
        batch = tuple(t.to(device) for t in batch)
        # 从train_data中获取mask和input_id
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        # 梯度清零！！
        optimizer.zero_grad()
        # 输入模型训练结果：损失及分类概率
        outputs = model(**inputs)
        loss, logits = outputs[:2]  # logits.shape: [batch_size, max_seq_length, num_labels]
        if config.model_type == "bert":
            tags = logits.argmax(dim=2).cpu().numpy().tolist()
        else:
            tags = model.crf.decode(logits, inputs['attention_mask'])
            tags = tags.squeeze(0).cpu().numpy().tolist()
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        input_lens = batch[4].cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(config.ids_to_labels[out_label_ids[i][j]])
                    temp_2.append(config.ids_to_labels[tags[i][j]])

        total_loss_train += loss.item()
        # 反向传播，累加梯度
        loss.backward()
        # 解决梯度爆炸问题
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 参数更新
        optimizer.step()
        scheduler.step()
        # model.zero_grad()
    eval_info, entity_info = metric.result()
    val_loss = total_loss_train / len(train_loader)
    print("train loss: {:.4f}, train Precision:{:.4f}, train Recall:{:.4f},train f1:{:.4f}"
          .format(val_loss, eval_info["precision"], eval_info["recall"], eval_info["f1"]))
    return {
        "loss": val_loss,
        "f1": eval_info["f1"]
    }


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
    metric = SeqEntityScore(config.ids_to_labels, markup='bio')
    total_loss_eval, total_pre_eval, total_recall_eval, total_f1_eval = 0, 0, 0, 0,
    model.eval()
    for idx, batch in enumerate(eval_loader):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            outputs = model(**inputs)
            loss, logits = outputs[:2]
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        input_lens = batch[4].cpu().numpy().tolist()
        if config.model_type == "bert":
            tags = logits.argmax(dim=2).cpu().numpy().tolist()
        else:
            tags = model.crf.decode(logits, inputs['attention_mask'])
            tags = tags.squeeze(0).cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(config.ids_to_labels[out_label_ids[i][j]])
                    temp_2.append(config.ids_to_labels[tags[i][j]])
        total_loss_eval += loss.item()
    eval_info, entity_info = metric.result()
    val_loss = total_loss_eval / len(eval_loader)
    print("Val loss: {:.4f}, Val Precision:{:.4f}, Val Recall:{:.4f},Val f1:{:.4f}"
          .format(val_loss, eval_info["precision"], eval_info["recall"], eval_info["f1"]))
    return {
        "loss": val_loss,
        "f1": eval_info["f1"]
    }


# 保存模型
def save_pretrained(model, path):
    # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))


def train(train_loader, dev_loader, model, optimizer, scheduler):
    train_loss_all = []
    train_f1_all = []
    val_loss_all = []
    val_f1_all = []
    epochs = []
    for epoch in range(1, config.epochs + 1):
        print('=========train at epoch={}========='.format(epoch))
        train_process = bioner_train(model, train_loader, optimizer, scheduler, config.device)
        print('=========val at epoch={}========='.format(epoch))
        evaluate_process = bioner_evaluate(model, dev_loader, config.device)
        train_loss_all.append(train_process["loss"])
        train_f1_all.append(train_process["f1"])
        val_loss_all.append(evaluate_process["loss"])
        val_f1_all.append(evaluate_process["f1"])
        epochs.append(epoch)
    data = {'epochs': epochs,
            'train_loss_all': train_loss_all,
            'val_loss_all': val_loss_all,
            }
    data2 = {'epochs': epochs,
             'train_f1_all': train_f1_all,
             'val_f1_all': val_f1_all
             }
    pd.DataFrame(data).to_csv("train_loss_data.csv")
    pd.DataFrame(data2).to_csv("train_f1_data.csv")


def main():
    # 标签+
    label_map: Dict[int, str] = {i: label for i, label in enumerate(config.labels_JNLPBA)}
    # 参数
    bert_config = AutoConfig.from_pretrained(
        model_config.MODE_BIOBERT,
        num_labels=len(config.labels_JNLPBA),
        id2label=label_map,
        label2id={label: i for i, label in enumerate(config.labels_JNLPBA)},
    )
    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(model_config.MODE_BIOBERT)
    # 模型
    model_bert = BertForTokenClassification.from_pretrained(model_config.MODE_BIOBERT,
                                                            num_labels=len(config.labels_JNLPBA)).to(config.device)
    model_bert_crf = BertCRF.from_pretrained(model_config.MODE_BIOBERT, config=bert_config).to(config.device)
    model_bert_bilstm_crf = BertBiLSTMCRF.from_pretrained(model_config.MODE_BIOBERT,
                                                          config=bert_config).to(config.device)

    # 定义训练和验证集数据
    train_dataset = load_and_cache_examples(
        data_dir=model_config.FILE_NAME,
        tokenizer=tokenizer,
        labels=config.labels_JNLPBA,
        max_seq_length=config.max_seq_length,
        data_type='train'
    )
    dev_dataset = load_and_cache_examples(
        data_dir=model_config.FILE_NAME,
        tokenizer=tokenizer,
        labels=config.labels_JNLPBA,
        max_seq_length=config.max_seq_length,
        data_type='dev'
    )
    # 批量获取训练和验证集数据
    train_sampler = RandomSampler(train_dataset)  # 随机采样
    dev_sampler = SequentialSampler(dev_dataset)  # 顺序采样
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, sampler=train_sampler,
                                  collate_fn=collate_fn)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=config.batch_size, sampler=dev_sampler,
                                collate_fn=collate_fn)

    # 交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    no_decay = ['bias', 'LayerNorm.weight']  # 控制系数不衰减的项
    # 定义优化器
    if config.model_type == "bert":
        param_optimizer = list(model_bert.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    elif config.model_type == "bert_crf":
        # BERT+CRF
        bert_param_optimizer = list(model_bert_crf.bert.named_parameters())
        crf_param_optimizer = list(model_bert_crf.crf.named_parameters())
        linear_param_optimizer = list(model_bert_crf.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay, 'lr': config.lr},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': config.lr},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay, 'lr': config.crf_lr},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': config.crf_lr},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay, 'lr': config.crf_lr},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': config.crf_lr}
        ]
    else:
        # BERT_bisltm_CRF
        bert_param_optimizer = list(model_bert_bilstm_crf.bert.named_parameters())
        crf_param_optimizer = list(model_bert_bilstm_crf.crf.named_parameters())
        lstm_param_optimizer = list(model_bert_bilstm_crf.bilstm.named_parameters())
        linear_param_optimizer = list(model_bert_bilstm_crf.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay, 'lr': config.lr},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': config.lr},

            {'params': [p for n, p in lstm_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay, 'lr': config.crf_lr},
            {'params': [p for n, p in lstm_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': config.crf_lr},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay, 'lr': config.crf_lr},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': config.crf_lr},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay, 'lr': config.crf_lr},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': config.crf_lr}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=config.adam_epsilon)  # AdamW
    #  Warmup学习率预热
    total_steps = len(train_dataset) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warm_up_ratio * total_steps,
                                                num_training_steps=total_steps)

    # Train the model
    logging.info("--------Start Training!--------")
    train(train_dataloader, dev_dataloader, model_bert, optimizer, scheduler)


if __name__ == "__main__":
    main()
