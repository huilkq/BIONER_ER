import torch

model_type = "bert_crf"
loss_type = "focal"
epochs = 10  # 训练轮次
batch_size = 16  # 每次取的大小
lr = 3e-5  # bert学习率
crf_lr = 1e-4  # 条件随机场学习率
adam_epsilon = 1e-8  # Epsilon for AdamW optimizer
weight_decay = 0.01  # 权重衰减
warm_up_ratio = 0.1  # 预热学习率
max_seq_length = 256  # tokens最大长度

labels = ["O", "B-Chemical", "I-Chemical"]
labels_JNLPBA = ["O", "B-protein", "I-protein", "B-DNA", "I-DNA", "B-cell_type", "I-cell_type", "B-cell_line",
                 "I-cell_line", "B-RNA", "I-RNA"]
labels_to_ids = {k: v for v, k in enumerate(labels_JNLPBA)}
ids_to_labels = {v: k for v, k in enumerate(labels_JNLPBA)}

# 调用gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
