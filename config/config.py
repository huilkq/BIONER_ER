import argparse


class Arguments:
    parser = argparse.ArgumentParser()
    # args for path
    parser.add_argument('--output_dir', default='./checkpoints/',
                        help='the output dir for model checkpoints')

    parser.add_argument('--bert_dir', default='../model_hub/bert-base-chinese/',
                        help='bert dir for uer')

    parser.add_argument('--PubMedBERT_dir', default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                        help='PubMedBERT dir for uer')

    parser.add_argument('--scibert_dir', default='allenai/scibert_scivocab_cased',
                        help='scibert dir for uer')

    parser.add_argument('--biobert_dir', default='dmis-lab/biobert-base-cased-v1.1',
                        help='biobert dir for uer')

    parser.add_argument('--data_dir', default='D:\BioNLP\BIONER_ER\datas\BC4CHEMD',
                        help='data dir for uer')

    parser.add_argument('--log_dir', default='./logs/',
                        help='log dir for uer')

    # other args
    parser.add_argument('--num_tags', default=53, type=int,
                        help='number of tags')

    parser.add_argument('--seed', type=int, default=123, help='random seed')

    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')

    parser.add_argument('--max_seq_len', default=256, type=int)

    parser.add_argument('--eval_batch_size', default=12, type=int)

    parser.add_argument('--swa_start', default=3, type=int,
                        help='the epoch when swa start')

    # train args
    parser.add_argument("--batch_size", type=int, default=1,
                        help='')

    parser.add_argument('--epochs', default=30, type=int,
                        help='训练轮次')

    parser.add_argument('--dropout_prob', default=0.1, type=float,
                        help='drop out probability')

    parser.add_argument('--lr', default=0.001, type=float,
                        help='bert学习率')

    parser.add_argument('--other_lr', default=3e-4, type=float,
                        help='bilstm和多层感知机学习率')

    parser.add_argument('--crf_lr', default=3e-2, type=float,
                        help='条件随机场学习率')

    parser.add_argument('--warm_up_ratio', default=0.1, type=float,
                        help='预热学习率')

    # 0.5
    parser.add_argument('--max_grad_norm', default=1, type=float,
                        help='max grad clip')

    parser.add_argument('--lstm_hidden', default=128, type=int,
                        help='lstm隐藏层大小')

    parser.add_argument('--num_layers', default=1, type=int,
                        help='lstm层数大小')

    parser.add_argument('--data_name', type=str, default='c',
                        help='数据集名字')

    parser.add_argument('--model_name', type=str, default='bert',
                        help='模型名字')

#     return parser
#
# def get_parser(self):
#     parser = self.parse()
#     parser = self.initialize(parser)
#     return parser.parse_args()
