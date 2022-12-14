import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from .module import SlotClassifier


class NerBERT(BertPreTrainedModel):
    """
    构建NerBERT模块
    """
    def __init__(self, config, args, slot_label_lst):
        super(NerBERT, self).__init__(config)
        # 参数
        self.args = args
        # 命名实体标签长度
        self.num_slot_labels = len(slot_label_lst)
        # 加载预训练BERT模型-编码器模块
        self.bert = BertModel(config=config)
        # 命名实体标签分类层
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)
        # 通过use_crf参数来判断是否使用crf层，如果用的话就加载
        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    # 定义NERBERT模型的前向传播
    def forward(self, input_ids, attention_mask, token_type_ids, slot_labels_ids):
        # input_ids: (B, L)
        # attention_mask: (B, L)
        # token_type_ids: (B, L)
        # slot_labels_ids: (B, L)

        # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        # 通过BERT模型得到向量表征
        sequence_output = outputs[0]  # [B, L, D]
        # [CLS]  # [B, D]
        pooled_output = outputs[1]
        # 将sequence_output在每一个tokens上做一个分类
        # (B, L, num_slot_labels)
        slot_logits = self.slot_classifier(sequence_output)
        # add hidden states and attention if they are here
        outputs = (slot_logits, ) + outputs[2:]

        # Slot Softmax
        if slot_labels_ids is not None:
            # 如果使用了crf，计算slot_loss
            if self.args.use_crf:
                slot_loss = self.crf(
                    slot_logits,
                    slot_labels_ids,
                    mask=attention_mask.byte(),
                    reduction='mean'
                )
                slot_loss = -1 * slot_loss  # negative log-likelihood
            # 如果没有使用crf，计算slot_loss
            else:
                # 交叉熵损失函数，设置loss function中忽略的label编号
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index) # default -100
                # Only keep active parts of the loss
                if attention_mask is not None:
                    # attention_mask中有1的部分就是有实际意义的部分
                    active_loss = attention_mask.view(-1) == 1  # (B * L, )
                    # slot_logits: (B, L, num_slot_labels) --> (B * L, num_slot_labels)
                    # (有效长度， num_slot_labels)
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    # ner标签有效长度部分
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    # 计算有效长度的交叉熵损失
                    slot_loss = slot_loss_fct(active_logits, active_labels)

                else:
                    # slot_logits.view(-1, self.num_slot_labels)输出为每一token对应每一个命名实体标签的概率
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))

            outputs = (slot_loss,) + outputs
        # 返回(loss), logits, (hidden_states), (attentions)
        return outputs
