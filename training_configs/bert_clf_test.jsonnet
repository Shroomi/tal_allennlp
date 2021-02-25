{
    "dataset_reader": {
        "lazy": false,
        "type": "action_cls_traindata_reader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": "/root/dingmengru/model/pretrained_bert/chinese_wwm_ext_pytorch",
            "do_lowercase": true
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "pretrained_model": "/root/dingmengru/model/pretrained_bert/chinese_wwm_ext_pytorch",
            }
        }
    },
    "train_data_path": "/root/dingmengru/data/action/v04-1/data/train_ding.jsonl",
    "validation_data_path": "/root/dingmengru/data/action/v04-1/data/valid_ding.jsonl",
    "model": {
        "type": "text_classifier_fscore_focal_loss",
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets"],
            },
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "pretrained_model": "/root/dingmengru/model/pretrained_bert/chinese_wwm_ext_pytorch",
                    "last_layer_only": false,
                    "train_parameters": false
                }
            }
        },
        "seq2vec_encoder": {
           "type": "bert_pooler",
           "pretrained_model": "/root/dingmengru/model/pretrained_bert/chinese_wwm_ext_pytorch",
           "requires_grad": false
        },
        "dropout": 0.25,
        "loss":"cross_entropy_loss"
},
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 5
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 15,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 0
    }
}