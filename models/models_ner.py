import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertForTokenClassification
from .layers.crf import CRF
from .layers.linears import PoolerStartLogits, PoolerEndLogits
from ..losses.focal_loss import FocalLoss
from ..losses.label_smoothing import LabelSmoothingCrossEntropy


class BertBiLSTMCRF(BertPreTrainedModel):

    def __init__(self, config):
        super(BertBiLSTMCRF, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)  # 第一层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 非线性层
        self.bilstm = nn.LSTM(  # LSTM层
            input_size=768,  # 1024
            hidden_size=config.hidden_size // 2,  # 1024 因为是双向LSTM，隐藏层大小为原来的一半
            batch_first=True,
            num_layers=2,
            dropout=0.5,  # 0.5 非线性
            bidirectional=True
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 得到每个词对于所有tag的分数
        self.crf = CRF(config.num_labels, batch_first=True)  # CEF层

        self.init_weights()  # 初始化权重，先全部随机初始化，然后调用bert的预训练模型中的权重覆盖

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, ):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids, )
        sequence_output = outputs[0]

        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(sequence_output)
        lstm_output, _ = self.bilstm(padded_sequence_output)
        # 得到判别值
        logits = self.classifier(lstm_output)
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs


class BertSoftmax(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmax, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # dropout pred_label的一部分feature
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if loss_mask is not None:
                # 只留下label存在的位置计算loss
                active_loss = loss_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


class BertCRF(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCRF, self).__init__(config)
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
        sequence_output = self.dropout(sequence_output)  # (batch_size, max_seq_length, num_labels)
        # 得到判别值
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask) * (-1)
            outputs = (loss,) + outputs
        return outputs  # (loss), scores


class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, config, ):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = config.soft_label
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs


class Bert_NER(torch.nn.Module):
    def __init__(self):
        super(Bert_NER, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('D:\BioNLP\BIONER_ER\\biobert-base-cased-v1.1',
                                                               num_labels=3)

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           labels=labels, return_dict=False)
        return output
