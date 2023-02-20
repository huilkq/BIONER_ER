import torch

model_type = "bert"
epochs = 50  # 训练轮次
batch_size = 32  # 每次取的大小
lr = 5e-5  # bert学习率
crf_lr = 3e-2  # 条件随机场学习率
weight_decay = 0.01  # 权重衰减
warm_up_ratio = 0.1  # 预热学习率
max_seq_length = 128  # tokens最大长度

labels = ["O", "B-Chemical", "I-Chemical"]
labels_JNLPBA = ["O", "B-protein", "I-protein", "B-DNA", "I-DNA", "B-cell_type", "I-cell_type", "B-cell_line",
                 "I-cell_line", "B-RNA", "I-RNA"]
labels_to_ids = {k: v for v, k in enumerate(labels)}
ids_to_labels = {v: k for v, k in enumerate(labels)}

# 调用gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")