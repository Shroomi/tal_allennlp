#!/usr/bin/env python
# encoding: utf-8
"""
@Author: dingmengru
@Contact: dingmengru1993@gmail.com
@File: check_tag_results.py
@Software: PyCharm
@Time: 2021/3/8 11:05 上午
@Desc: 验收标注人员的标注结果

"""
import json
import argparse
from tqdm import tqdm

from config import *


class CheckTags(object):

    def write_tag_file(self, tag_file_name):
        """
        将 从vegas下载的标注结果文件 转为 jsonl格式文件
        :param tag_file_name: 标注人员的标注结果文件
        :return:
        """
        write_file_name = tag_file_name.split('-')[0] + '_' + tag_file_name.split('-')[-1] + 'l'
        with open(os.path.join(TAG_PATH, tag_file_name), 'r') as fin, \
                open(os.path.join(TAG_PATH, write_file_name), 'w', encoding='utf-8') as fout:
            file_data = json.load(fin)
            tag_data = file_data['true_message']['datas']
            for tag_info in tag_data:
                for info in tag_info['mark_datas']:
                    text_tmp = info['section_text'].split('###')
                    dataset = text_tmp[0]
                    text = text_tmp[1]
                    label = info['label'][0]['children'][0]
                    fout.write(json.dumps({
                        'text': text,
                        'label': label,
                        'source': dataset
                    }, ensure_ascii=False) + '\n')

    def check_tag_file(self, standard_tag_file, file_start: str, file_end: str):
        # 将自己埋雷的标注文件记录到字典中，key为文本内容，value为自己的标注结果
        text_tag_standard = dict()
        with open(os.path.join(TAG_PATH, standard_tag_file), 'r') as fin:
            for line in fin:
                data = json.loads(line)
                print(data)
                text_tag_standard[data['text']] = data['label']
        standard_num = len(text_tag_standard)

        tag_files = os.listdir(TAG_PATH)
        for tag_file in tag_files:
            right_cnt = 0
            denom_cnt = 0
            if tag_file.startswith(file_start) and tag_file.endswith(file_end):
                with open(os.path.join(TAG_PATH, tag_file), 'r') as fin, \
                        open(os.path.join(TAG_PATH, 'wrong_' + tag_file), 'w', encoding='utf-8') as fout:
                    for line in fin:
                        data = json.loads(line)
                        text = data['text']
                        tag_from_others = data['label']
                        if text in text_tag_standard:
                            denom_cnt += 1
                            if int(tag_from_others) == int(text_tag_standard[text]):
                                right_cnt += 1
                            else:
                                fout.write(json.dumps({
                                    'text': text,
                                    'right_tag': text_tag_standard[text],
                                    'wrong_tag': tag_from_others
                                }, ensure_ascii=False) + '\n')
                    print(denom_cnt)
                    print('file: {}, accuracy: {}'.format(os.path.join(TAG_PATH, tag_file), right_cnt / denom_cnt))


def run():
    # 获取用户输入
    parser = argparse.ArgumentParser()
    parser.description = 'pleaser enter your choice(c): write tag file or check tag file...'
    parser.add_argument('-c', '--inputC', help='value choice: write or check', type=str, default='check')
    args = parser.parse_args()
    print(args.inputC)
    user_para = args.inputC

    check_tags = CheckTags()
    if user_para == 'check':
        # 检查标注人员标注结果是否符合要求
        check_tags.check_tag_file('tag_standard_0.jsonl', '8222', '.jsonl')
    elif user_para == 'write':
        # 将标注人员的标注结果转换成jsonl格式文件
        tag_files_list = os.listdir(TAG_PATH)
        for tag_file in tqdm(tag_files_list):
            if tag_file.startswith('8222'):
                check_tags.write_tag_file(tag_file)
    else:
        print('user parameter is wrong(write or check) !!!')


if __name__ == '__main__':
    run()
