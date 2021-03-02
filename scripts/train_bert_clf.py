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
import torch
from typing import Iterable, Tuple

import allennlp
from allennlp.models import Model
from allennlp.data import PyTorchDataLoader
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer

from tal_allennlp.data.dataset_readers.dataset_utils.writing_correction_data_loader import ActionCLSTrainDataReader

from config import *


def build_dataset_reader() -> DatasetReader:
    tokenizer = PretrainedTransformerTokenizer(model_name=PRETRAIN_MODEL)
    token_indexers = PretrainedTransformerIndexer(model_name=PRETRAIN_MODEL)
    return ActionCLSTrainDataReader(tokenizer=tokenizer,
                                    token_indexers={'tokens': token_indexers},
                                    max_tokens=150,)


def read_data(reader: DatasetReader, train_file_name, valid_file_name):
    train_data_instances = reader.read(os.path.join(DATA_PATH, train_file_name))
    valid_data_instances = reader.read(os.path.join(DATA_PATH, valid_file_name))
    return train_data_instances, valid_data_instances


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print('Building the vocabulary')
    vocab_path = os.path.join(PRETRAIN_MODEL, 'vocab')
    if os.path.exists(vocab_path):
        vocab = Vocabulary.from_files(vocab_path)
    else:
        vocab = Vocabulary.from_instances(instances)
        vocab.save_to_files(vocab_path)
    return vocab


def build_data_loaders(
        train_data: torch.utils.data.Dataset,
        dev_data: torch.utils.data.Dataset
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    batch_size = 8
    train_loader = PyTorchDataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = PyTorchDataLoader(dev_data, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader


def build_model(vocab: Vocabulary) -> Model:
    print('Building the model')


def run_training_loop():
    dataset_reader = build_dataset_reader()
    print('Reading data')
    train_instances, valid_instances = read_data(dataset_reader, 'train_ding.jsonl', 'valid_ding.jsonl')
    for instance in list(train_instances)[:5]:
        print(instance)
    vocab = build_vocab(train_instances + valid_instances)
    # print(vocab.get_vocab_size('tokens'))


if __name__ == '__main__':
    run_training_loop()
