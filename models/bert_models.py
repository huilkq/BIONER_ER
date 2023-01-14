import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig, BertForTokenClassification
from torchcrf import CRF
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss


class Bert_BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def _get_features(self, sentence):
        with torch.no_grad():
            embeds, _ = self.bert(sentence)
        enc, _ = self.lstm(embeds)
        enc = self.dropout(enc)
        feats = self.linear(enc)
        return feats

    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence)
        if not is_test:  # 训练阶段，返回loss
            loss = -self.crf.forward(emissions, tags, mask, reduction='mean')
            return loss
        else:  # 测试阶段，返回decoding结果
            decode = self.crf.decode(emissions, mask)
            return decode


class BertCrfForNer(BertPreTrainedModel):

    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        # 加载预训练BERT模型-编码器模块
        self.bert = BertModel(config)
        # Dropout的是为了防止过拟合而设置,只能用在训练部分而不能用在测试部分
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 命名实体标签分类层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # crf层
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # 得到判别值
        logits = self.classifier(sequence_output)
        print(logits.shape)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1*loss,)+outputs
        return outputs  # (loss), scores


class BertCrfNer(BertPreTrainedModel):

    def __init__(self, config):
        super(BertCrfNer, self).__init__(config)
        self.config = config
        # 加载预训练BERT模型-编码器模块
        self.bert = BertModel(config)
        # Dropout的是为了防止过拟合而设置,只能用在训练部分而不能用在测试部分
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 命名实体标签分类层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # crf层
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids
                            )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # 得到判别值
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1*loss,)+outputs
        return outputs  # (loss), scores


class Bert_NER(torch.nn.Module):
    def __init__(self):
        super(Bert_NER, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('D:\BioNLP\BIONER_ER\\biobert-base-cased-v1.1', num_labels=3)

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           labels=labels, return_dict=False)
        return output

