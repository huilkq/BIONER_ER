import nltk


def sampleNE():
    sent = nltk.corpus.treebank.tagged_sents()[0]  # 语料库第一句
    print(nltk.ne_chunk(sent))  # nltk.ne_chunk()函数分析识别一个句子的命名实体


def sampleNE2():
    sent = nltk.corpus.treebank.tagged_sents()[0]
    print(nltk.ne_chunk(sent, binary=True))  # 包含识别无类别的命名实体


if __name__ == "__main__":
    sampleNE()
    sampleNE2()