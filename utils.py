#!/usr/bin/env python
# encoding: utf-8
"""
@Author: dingmengru
@Contact: dingmengru1993@gmail.com
@File: utils.py
@Software: PyCharm
@Time: 2021/2/24 11:29 上午
@Desc: 存储一些临时处理脚本

"""
import json
from tqdm import tqdm
from collections import defaultdict, Counter

from config import *


LABEL_MAP = {
    '是': 1,
    '否': 0
}


class TagProcess(object):
    """
    处理标注相关的数据
    """

    def __init__(self):
        self.text_tag_dict = defaultdict(list)

    def all_people_tags(self, tag_path=TAG_PATH):
        """
        获取每条text，对应的5个人的标注结果
        存储结构为：
        :param tag_path:
        :return:
        """
        tag_files = os.listdir(tag_path)
        for tag_file in tqdm(tag_files):
            if not tag_file.endswith('json'):
                continue
            with open(os.path.join(tag_path, tag_file), 'r') as fin:
                file_data = json.load(fin)
                tag_data = file_data['true_message']['datas']
                for tag_info in tag_data:
                    for info in tag_info['mark_datas']:
                        text = info['section_text']
                        label = info['label'][0]['children'][0]
                        self.text_tag_dict[text].append(label)
        print(self.text_tag_dict['我跑起来像火车一样快。'])
        print('Finish getting all tags from five people!')
        return self.text_tag_dict

    def get_final_tag(self, input_file_name, output_path):
        with open(os.path.join(DATA_PATH, input_file_name), 'r') as fin, \
                open(os.path.join(output_path), 'w', encoding='utf-8') as fout:
            train_file = json.load(fin)
            data_list = train_file['data']
            print(len(data_list))
            for text, label_list in tqdm(self.text_tag_dict.items()):
                most_label = Counter(label_list).most_common(1)[0]
                if most_label[1] >= 4 and most_label[0] in LABEL_MAP:
                    data_list.append({
                        'text': text,
                        'label': LABEL_MAP[most_label[0]],
                    })
            fout.write(json.dumps({
                'data': data_list
            }, ensure_ascii=False))
            print(len(data_list))
        print('Finish writing new tag data in the train file!')


if __name__ == '__main__':
    tag_process = TagProcess()
    text_tag = tag_process.all_people_tags()
    tag_process.get_final_tag('train.json', '/root/dingmengru/data/action/v05/train_zh.json')
