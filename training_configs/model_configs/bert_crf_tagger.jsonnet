{
    "dataset_reader": {
        "type": "people_daily_tagging",
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": "/root/dingmengru/model/pretrained_bert/chinese_wwm_ext_pytorch",
                "max_length": 500
            }
        },
        "max_tokens": 500
        // "limit" : 10
    },
    "train_data_path": "/root/dingmengru/data/action/peopel_daily_2014_ner/train.jsonl",
    "validation_data_path": "/root/dingmengru/data/action/peopel_daily_2014_ner/valid.jsonl",
    "model": {
        "type": "bert_crf_tagger",
        "text_field_embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": "/root/dingmengru/model/pretrained_bert/chinese_wwm_ext_pytorch",
                    "last_layer_only": true,
                    "train_parameters": true
                }
            }
        },
        "dropout": 0
},

    "data_loader": {
        "batch_size": 16,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 10,
        "validation_metric": "+f1-measure-overall",
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 0
    }
}