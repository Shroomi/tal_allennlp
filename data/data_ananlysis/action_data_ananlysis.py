#!/usr/bin/env python
# encoding: utf-8
"""
@Author: dingmengru
@Contact: dingmengru1993@gmail.com
@File: action_data_ananlysis.py
@Software: PyCharm
@Time: 2021/2/25 7:49 下午
@Desc: 动作描写训练数据分析

"""
import json
from typing import Set

from config import *


class DataAnalysis(object):

    def __init__(self):
        self.file1_set = set()
        self.file2_set = set()

    def get_file_set(self, file_name: str, file_set: set):
        with open(os.path.join(DATA_PATH, file_name), 'r') as fin:
            for line in fin:
                data = json.loads(line)
                text_label = (data['text'], data['label'])
                file_set.add(text_label)

    def diff_from_votes(self, file_name1, file_name2):
        self.get_file_set(file_name1, self.file1_set)
        self.get_file_set(file_name2, self.file2_set)
        diff_set_1to2 = self.file1_set - self.file2_set
        diff_set_2to1 = self.file2_set - self.file1_set
        return diff_set_1to2, diff_set_2to1

    def write_diff(self, input_set: Set[tuple], output_file):
        with open(os.path.join(DATA_PATH, output_file), 'w', encoding='utf-8') as fout:
            for text_label in input_set:
                text = text_label[0]
                label = text_label[1]
                fout.write(json.dumps({
                    'text': text,
                    'label': label,
                }, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    data_analysis = DataAnalysis()
    diff_1to2, diff_2to1 = data_analysis.diff_from_votes('test_greater3.jsonl', 'test_greater4.jsonl')
    print(diff_1to2)  # greater3 - greater4 应该有内容
    print(diff_2to1)  # greater4 - greater3 应该为空集
    data_analysis.write_diff(diff_1to2, 'diff_set_3to4.jsonl')
