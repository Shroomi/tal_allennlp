#!/usr/bin/env python
# encoding: utf-8
"""
@Author: dingmengru
@Contact: dingmengru1993@gmail.com
@File: config.py
@Software: PyCharm
@Time: 2021/2/7 7:46 下午
@Desc: configuration of the project

"""
import os
import yaml

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
CONF_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')
config_dict = {}

with open(CONF_PATH, 'r') as fin:
    config_dict = yaml.load(fin)

DATA_PATH = config_dict['ACTION_PATH']['DATA_PATH']
EMB_PATH = config_dict['ACTION_PATH']['EMB_PATH']
MODEL_SAVE_PATH = config_dict['ACTION_PATH']['MODEL_SAVE_PATH']
TAG_PATH = config_dict['ACTION_PATH']['TAG_PATH']
PRETRAIN_MODEL = config_dict['ACTION_PATH']['PRETRAIN_MODEL']
