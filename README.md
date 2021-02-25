### 代码架构(更新中)

```sh
tal_allennlp
├── tal_allennlp
│   ├── data
│   │     ├── dataset_readers
│   │     ├── token_indexers
│   │     └── tokenizers
│   ├── models
│   │     └── classification
│   └── modules
│         └── loss
├── training_configs
│   ├── bert_classification.jsonnet
├── scripts
├── config.py
├── config.yaml
├── README.md

```

### 训练数据样例
#### 1，分类
train.jsonl:
```text
{"text": "现在在学校住宿，我已经能够住下来了。", "label": 0}
{"text": "因为感恩我们的校园溢满幸福。", "label": 0}
{"text": "紫萱闭住呼吸，手抱住了那匹马的脖子，乘机骑了上去，样子看起来像“白马公主”。", "label": 1}
{"text": "说完他永远的倒下了。", "label": 0}
{"text": "她倚靠在门框上，手不住地在围裙上来回地搓动。", "label": 1}
```

### 参考文献
- Allennlp官方教程：<https://guide.allennlp.org/overview>
- Allennlp注册机制：<https://guide.allennlp.org/using-config-files>
- HuggingFace手册：<https://huggingface.co/transformers/>