#!/usr/bin/env python
# encoding: utf-8
"""
@Author: dingmengru
@Contact: dingmengru1993@gmail.com
@File: writing_correction_data_loader.py
@Software: PyCharm
@Time: 2021/2/7 7:30 下午
@Desc: 载入动作描写分类数据

"""
import os
import json
import logging
from typing import Dict, Iterable
from overrides import overrides

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.tokenizers import Tokenizer, CharacterTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from config import *
from tal_allennlp.data.tokenizers.jieba_tokenizer import JiebaTokenizer
from tal_allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("action_cls_traindata_reader")
class ActionCLSTrainDataReader(DatasetReader):

    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None,
            tokenizer: Tokenizer = None,
            max_tokens: int = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or CharacterTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as fin:
            for line in fin:
                data = json.loads(line)
                text, label = data['text'], str(data['label'])
                tokens = self.tokenizer.tokenize(text)
                if self.max_tokens:
                    tokens = tokens[:self.max_tokens]
                text_field = TextField(tokens, self.token_indexers)
                label_field = LabelField(label)
                fields: Dict[str, Field] = {'tokens': text_field, 'label': label_field}
                yield Instance(fields)


def test_data_read(file_name: str):
    """
    测试ActionCLSTrainDataReader类是否正确
    :param file_path:
    :return:
    """
    dataset_reader = ActionCLSTrainDataReader(tokenizer=JiebaTokenizer(), max_tokens=150)
    instances = list(dataset_reader.read(os.path.join(DATA_PATH, file_name)))
    for instance in instances[:5]:
        print(instance)


def test_data_read_bert(file_name: str):
    token_indexers = PretrainedBertIndexer(pretrained_model=PRETRAIN_MODEL,
                                           do_lowercase=False,
                                           use_starting_offsets=False)
    dataset_reader = ActionCLSTrainDataReader(max_tokens=150,
                                              token_indexers={'tokens': token_indexers})
    instances = list(dataset_reader.read(os.path.join(DATA_PATH, file_name)))
    for instance in instances[:5]:
        print(instance)


if __name__ == '__main__':
    test_data_read_bert('train_ding.jsonl')
