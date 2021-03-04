#!/usr/bin/env python
# encoding: utf-8
"""
@Author: dingmengru
@Contact: dingmengru1993@gmail.com
@File: people_daily2004_data_loader.py
@Software: PyCharm
@Time: 2021/3/4 4:22 下午
@Desc: 以人民日报NER数据为例

"""
import os
import json
import logging
from typing import Dict, Iterable, List
from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer

from config import *

logger = logging.getLogger(__name__)


@DatasetReader.register("people_daily_tagging")
class PeopleDailyTaggingDatasetReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 limit: int = -1) -> None:
        super(PeopleDailyTaggingDatasetReader, self).__init__()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._max_tokens = max_tokens
        self._limit = limit

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        limit = self._limit
        with open(os.path.join(DATA_PATH, file_path), 'r') as fin:
            for line in fin:
                data = json.loads(line)
                tokens = [Token(token) for token in data['text'].split()]
                tags = [tag for tag in data['tags'].split()]
                if self._max_tokens:
                    tokens = tokens[:self._max_tokens]
                    tags = tags[:self._max_tokens]
                yield self.text_to_instance(tokens, tags)
                limit -= 1
                if limit == 0:
                    break

    def text_to_instance(self,
                         tokens: List[Token],
                         tags: List[str]) -> Instance:
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["passage"] = sequence
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        if tags is not None:
            fields["labels"] = SequenceLabelField(tags, sequence)
        return Instance(fields)


def test_data_read_bert(file_name: str):
    token_indexers = PretrainedTransformerIndexer(model_name=PRETRAIN_MODEL,
                                                  max_length=500)
    dataset_reader = PeopleDailyTaggingDatasetReader(token_indexers={'tokens': token_indexers})
    instances = list(dataset_reader.read(os.path.join(DATA_PATH, file_name)))
    for instance in instances[:5]:
        print(instance)


if __name__ == '__main__':
    test_data_read_bert('valid.jsonl')
