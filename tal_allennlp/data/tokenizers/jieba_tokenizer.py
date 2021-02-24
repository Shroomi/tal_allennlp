#!/usr/bin/env python
# encoding: utf-8
"""
@Author: dingmengru
@Contact: dingmengru1993@gmail.com
@File: jieba_tokenizer.py
@Software: PyCharm
@Time: 2021/2/19 8:49 下午
@Desc: jieba分词

"""
import logging
from typing import List

import jieba
from overrides import overrides

from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("jieba_tokenizer")
class JiebaTokenizer(Tokenizer):
    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(t) for t in jieba.lcut(text)]
