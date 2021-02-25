#!/usr/bin/env python
# encoding: utf-8
"""
@Author: dingmengru
@Contact: dingmengru1993@gmail.com
@File: train_debug.py
@Software: PyCharm
@Time: 2021/2/25 2:15 下午
@Desc: 根据config文件进行模型训练，并且支持debug

"""
import os
from allennlp.commands.train import train_model_from_file

from config import *


def train_debug(param_path, serialization_dir):
    train_model_from_file(param_path, serialization_dir, force=True)


if __name__ == '__main__':
    print(PROJECT_ROOT)
    CONFIG_DIR = os.path.join(PROJECT_ROOT, 'training_configs')
    CONFIG_FILE = 'bert_clf_test.jsonnet'
    SERIALIZATION_DIR = 'debug_bert_clf'
    train_debug(os.path.join(CONFIG_DIR, CONFIG_FILE), SERIALIZATION_DIR)
