import os
import torch
from bert.modeling import BertModel


class BertNERModel(torch.nn.Module):

    # 初始化类
    def __init__(self, ner_labels, pretrained_name='bert-base-cased'):
        """
        Args:
            class_size  :指定分类模型的最终类别数目，以确定线性分类器的映射维度
            pretrained_name :用以指定bert的预训练模型
        """
        super(BertNERModel, self).__init__()
        # 加载HuggingFace的BertModel
        # BertModel的最终输出维度默认为768
        # return_dict=True 可以使BertModel的输出具有dict属性，即以 bert_output['last_hidden_state'] 方式调用
        self.bert = BertModel.from_pretrained(pretrained_name,
                                              return_dict=True)
        # 通过一个线性层将标签对应的维度：768->class_size
        self.classifier = torch.nn.Linear(768, ner_labels)

    def forward(self, inputs):
        # 获取DataLoader中已经处理好的输入数据：
        # input_ids :tensor类型，shape=batch_size*max_len   max_len为当前batch中的最大句长
        # input_tyi :tensor类型，
        # input_attn_mask :tensor类型，因为input_ids中存在大量[Pad]填充，attention mask将pad部分值置为0，让模型只关注非pad部分
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs[
            'token_type_ids'], inputs['attention_mask']
        output = self.bert(input_ids, input_tyi, input_attn_mask)
        # bert_output 分为两个部分：
        #   last_hidden_state:最后一个隐层的值
        #   pooler output:对应的是[CLS]的输出,用于分类任务
        # categories_numberic：tensor类型，shape=batch_size*class_size，用于后续的CrossEntropy计算
        categories_numberic = self.classifier(output.last_hidden_state)
        batch_size, seq_len, ner_class_num = categories_numberic.shape
        categories_numberic = categories_numberic.view(
            (batch_size * seq_len, ner_class_num))
        return categories_numberic


def save_pretrained(model, path):
    # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))