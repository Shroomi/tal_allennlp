#!/usr/bin/env python
# encoding: utf-8
"""
@Author: dingmengru
@Contact: dingmengru1993@gmail.com
@File: action_classification_tag.py
@Software: PyCharm
@Time: 2021/2/26 1:47 下午
@Desc: 生成动作描写提标数据

"""
import os
import json
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy

from config import *


def apply_for_tag(data_path, input_file='train.jsonl', output_file='total_label.json'):

    output_dict = defaultdict(list)
    each_passage = defaultdict()
    passage_num = 1
    each_passage['resourceName'] = '文本' + str(passage_num)
    each_passage['textarea'] = []
    textId = 1
    textNum = 1

    with open(os.path.join(data_path, input_file), 'r') as fin, \
            open(os.path.join(data_path, output_file), 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            data = json.loads(line)
            each_passage['textarea'].append({
                'textId': textId,
                'textNum': textNum,
                'textContent': data['source']+'###'+data['text'],
            })
            textId += 1
            textNum += 1
            if len(each_passage['textarea']) == 50:
                tmp_passage = deepcopy(each_passage)
                output_dict['taskDates'].append(tmp_passage)
                textId = 1
                textNum = 1
                passage_num += 1
                each_passage['resourceName'] = '文本' + str(passage_num)
                each_passage['textarea'] = []
        output_dict['taskDates'].append(each_passage)
        fout.write(json.dumps(output_dict, ensure_ascii=False))


if __name__ == '__main__':
    tag_files = os.listdir(DATA_PATH)
    i = 1
    for tag_file in tag_files:
        if tag_file.startswith('shuf'):
            apply_for_tag(DATA_PATH, input_file=tag_file, output_file='tag_action_p{}.json'.format(i))
            i += 1
