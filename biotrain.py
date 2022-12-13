import os

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

import modle
from BioBERT.processors.preprocess import DataSequence


def train_loop(model, df_train, df_val):
    # 定义训练和验证集数据
    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)
    # 批量获取训练和验证集数据
    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=1)
    # 判断是否使用GPU，如果有，尽量使用，可以加快训练速度
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义优化器
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()
    # 开始训练循环
    best_acc = 0
    best_loss = 1000
    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0
        # 训练模型
        model.train()
        # 按批量循环训练模型
        for train_data, train_label in tqdm(train_dataloader):
      # 从train_data中获取mask和input_id
            train_label = train_label[0].to(device)
            mask = train_data['attention_mask'][0].to(device)
            input_id = train_data['input_ids'][0].to(device)
            # 梯度清零！！
            optimizer.zero_grad()
            # 输入模型训练结果：损失及分类概率
            loss, logits = model(input_id, mask, train_label)
            # 过滤掉特殊token及padding的token
            logits_clean = logits[0][train_label != -100]
            label_clean = train_label[train_label != -100]
            # 获取最大概率值
            predictions = logits_clean.argmax(dim=1)
      # 计算准确率
            acc = (predictions == label_clean).float().mean()
            total_acc_train += acc
            total_loss_train += loss.item()
      # 反向传递
            loss.backward()
            # 参数更新
            optimizer.step()
        # 模型评估
        model.eval()

        total_acc_val = 0
        total_loss_val = 0
        for val_data, val_label in val_dataloader:
      # 批量获取验证数据
            val_label = val_label[0].to(device)
            mask = val_data['attention_mask'][0].to(device)
            input_id = val_data['input_ids'][0].to(device)
      # 输出模型预测结果
            loss, logits = model(input_id, mask, val_label)
      # 清楚无效token对应的结果
            logits_clean = logits[0][val_label != -100]
            label_clean = val_label[val_label != -100]
            # 获取概率值最大的预测
            predictions = logits_clean.argmax(dim=1)
            # 计算精度
            acc = (predictions == label_clean).float().mean()
            total_acc_val += acc
            total_loss_val += loss.item()

        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)

        print(
            f'''Epochs: {epoch_num + 1} | 
                Loss: {total_loss_train / len(df_train): .3f} | 
                Accuracy: {total_acc_train / len(df_train): .3f} |
                Val_Loss: {total_loss_val / len(df_val): .3f} | 
                Accuracy: {total_acc_val / len(df_val): .3f}''')

def save_pretrained(model, path):
    # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))


LEARNING_RATE = 1e-2
EPOCHS = 5
model = modle.BertModel()
train_loop(model, df_train, df_val)