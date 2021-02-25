#!/usr/bin/env python
# encoding: utf-8
"""
@Author: dingmengru
@Contact: dingmengru1993@gmail.com
@File: train_bert_clf.py
@Software: PyCharm
@Time: 2021/2/23 7:41 下午
@Desc: 与bert_classification.jsonnet对应，写成脚本方便调试

"""
from typing import Iterable

from allennlp.models import Model
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer

from tal_allennlp.data.dataset_readers.dataset_utils.writing_correction_data_loader import ActionCLSTrainDataReader

from config import *


def build_dataset_reader() -> DatasetReader:
    tokenizer = PretrainedTransformerTokenizer(model_name=PRETRAIN_MODEL)
    token_indexers = PretrainedTransformerIndexer(model_name=PRETRAIN_MODEL)
    return ActionCLSTrainDataReader(tokenizer=tokenizer,
                                    token_indexers={'bert': token_indexers},
                                    max_tokens=150,)


def read_data(reader: DatasetReader, train_file_name, valid_file_name):
    train_data_instances = reader.read(os.path.join(DATA_PATH, train_file_name))
    valid_data_instances = reader.read(os.path.join(DATA_PATH, valid_file_name))
    return train_data_instances, valid_data_instances


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print('Building the vocabulary')
    return Vocabulary.from_instances(instances)


def run_training_loop():
    dataset_reader = build_dataset_reader()
    print('Reading data')
    train_instances, valid_instances = read_data(dataset_reader, 'train_ding.jsonl', 'valid_ding.jsonl')
    vocab = build_vocab(train_instances + valid_instances)
    print(vocab.get_vocab_size('bert'))


def build_model(vocab: Vocabulary) -> Model:
    print('Building the model')


if __name__ == '__main__':
    run_training_loop()
