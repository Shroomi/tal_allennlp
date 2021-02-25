### 代码架构(ing)

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


### 参考文献
- allennlp官方教程：<https://guide.allennlp.org/overview>
- allennlp注册机制：<https://guide.allennlp.org/using-config-files>
- hugging face手册：<https://huggingface.co/transformers/>