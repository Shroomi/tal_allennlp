#!/usr/bin/env python
# encoding: utf-8
"""
@Author: dingmengru
@Contact: dingmengru1993@gmail.com
@File: tag_download_select.py
@Software: PyCharm
@Time: 2021/2/19 10:04 下午
@Desc: 数据相关的utils

"""
import os
import json
import pickle
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd

from config import *


def load_emb_info(embedding_path: str):
    """
    载入word embeddings相关文件信息
    :return:
    """
    embd = np.load(os.path.join(EMB_PATH, embedding_path, 'matrix.npy'))
    print(embd.shape[0])
    word_to_idx = pickle.load(open(os.path.join(EMB_PATH, embedding_path, 'w2i.pkl'), 'rb'))
    print(word_to_idx['unk'])
    idx_to_word = pickle.load(open(os.path.join(EMB_PATH, embedding_path, 'i2w.pkl'), 'rb'))
    print(idx_to_word[0])


def load_dense_drop_repeat(embedding_path, file_name, emb_size):
    """
    根据word embeddings文件，
    去重转换为模型可用的w2i，i2w以及matrix

    用处：可以根据自己的训练数据分词结果，生成此训练集对应的word embeddings子集，
    从而提高分词命中率，减少unk

    :param embedding_path:
    :param file_name:
    :return:
    """
    w2i, i2w = {}, {}
    count = 2
    w2i['pad'], w2i['unk'] = 0, 1
    i2w[0], i2w[1] = 'pad', 'unk'

    file_path = os.path.join(EMB_PATH, embedding_path, file_name)
    vocab_size = int(os.popen('wc -l {}'.format(file_path)).read().split(' ')[0])
    matrix = np.zeros(shape=(vocab_size, emb_size), dtype=np.float32)

    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            data = json.loads(line)
            word = data['word']
            embedding = data['embedding']
            if word == 'pad':
                matrix[0, :] = np.array([float(x) for x in embedding.split(' ')])
                continue
            if word == 'unk':
                matrix[1, :] = np.array([float(x) for x in embedding.split(' ')])
                continue
            if word not in w2i:
                w2i[word] = count
                i2w[count] = word
                matrix[count, :] = np.array([float(x) for x in embedding.split(' ')])
                count += 1
        assert len(w2i) == len(i2w) == matrix.shape[0]
        write_pickle(os.path.join(EMB_PATH, embedding_path, 'w2i.pkl'), w2i)
        write_pickle(os.path.join(EMB_PATH, embedding_path, 'i2w.pkl'), i2w)
        np.save(os.path.join(EMB_PATH, embedding_path, 'matrix.npy'), matrix)


def write_pickle(file_path, data_dict):
    with open(file_path, 'wb') as fout:
        pickle.dump(data_dict, fout)


def process_other_data(data_path, file_name, source='train'):
    """
    将数据转换成jsonl格式
    :param data_path:
    :param file_name:
    :return:
    """
    data = pd.read_json(os.path.join(data_path, file_name), orient='records')
    with open(os.path.join(data_path, file_name.split('.')[0]+'.jsonl'), 'w', encoding='utf-8') as fout:
        for index, row in tqdm(data.iterrows()):
            # print(row['text'], row['label'])
            fout.write(json.dumps(
                {
                    'text': row['text'],
                    'label': row['label'],
                    'source': source
                }, ensure_ascii=False
            ) + '\n')


def cal_pos_neg(data_path, file_name):
    data_frame = pd.read_json(os.path.join(data_path, file_name), lines=True)
    print(data_frame['label'].value_counts()[0] / data_frame['label'].value_counts()[1])


if __name__ == '__main__':
    """
    各个函数用例
    """
    # load_emb_info('tencent_big')
    # load_dense_drop_repeat('tencent_big', 'v04_embeddings.json', 200)
    process_other_data(DATA_PATH, 'test_old_ding.json', source='test_old')
    # cal_pos_neg(DATA_PATH, 'tmp_tr.jsonl')
