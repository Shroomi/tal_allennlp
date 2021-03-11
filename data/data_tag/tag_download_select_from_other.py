#!/usr/bin/env python
# encoding: utf-8
"""
@Author: dingmengru
@Contact: dingmengru1993@gmail.com
@File: tag_download_select_from_other.py
@Software: PyCharm
@Time: 2021/3/10 7:43 下午
@Desc:

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
                    essay_id = tag_info['text_name']
                    for info in tag_info['mark_datas']:
                        text = essay_id + '###' + info['section_text']
                        label = info['label'][0]['children'][0]
                        # 去重
                        if text in self.text_tag_dict and len(self.text_tag_dict[text]) >= 5:
                            continue
                        self.text_tag_dict[text].append(label)
        print('tag file path:', tag_path)
        print('Finish getting all tags from five people!')
        return self.text_tag_dict

    def get_final_tag(self, input_file_name, output_path):
        """
        从5个标注人员的标注结果中，选取投票大于等于4的投票结果，其余数据抛弃
        在原有的训练数据的基础上增量添加
        此处原有的训练数据是json文件而非jsonl文件
        :param input_file_name: 原有的训练数据文件
        :param output_path: 加入新增后的训练数据文件
        :return:
        """
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

    def write_final_tag(self, output_file):
        """
        根据标注人员的标注结果投票获得最终结果
        :param output_file:
        :return:
        """
        with open(os.path.join(TAG_PATH, output_file), 'w', encoding='utf-8') as fout:
            for id_text, label_list in tqdm(self.text_tag_dict.items()):
                most_label = Counter(label_list).most_common(1)[0]
                if '游春悠悠漫步在小溪边，目光被一簇粉红吸引静默了一秋冬的桃枝上竟冒出串串花苞' in id_text:
                    print(label_list)
                if most_label[1] >= 4:
                    text_tmp = id_text.split('###')
                    essay_id = text_tmp[0]
                    text = text_tmp[1]
                    fout.write(json.dumps({
                        'text': text,
                        'label': most_label[0],
                        'essay_id': essay_id
                    }, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    tag_process = TagProcess()
    text_tag = tag_process.all_people_tags()
    # tag_process.get_final_tag('train.json', '/root/dingmengru/data/action/v05/train_zh.json')
    tag_process.write_final_tag('final_tag_greater4.jsonl')
