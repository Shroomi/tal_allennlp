{
    "dataset_reader": {
        "lazy": false,
        "type": "action_cls_traindata_reader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": "/root/dingmengru/model/pretrained_bert/chinese_wwm_ext_pytorch",
            "max_length": 150
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": "/root/dingmengru/model/pretrained_bert/chinese_wwm_ext_pytorch",
                "max_length": 150
            }
        }
    },
    "train_data_path": "/root/dingmengru/data/action/v04-1/data/train_ding.jsonl",
    "validation_data_path": "/root/dingmengru/data/action/v04-1/data/valid_ding.jsonl",
    "model": {
        "type": "text_classifier_fscore_focal_loss",
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
        "seq2vec_encoder": {
           "type": "bert_pooler",
           "pretrained_model": "/root/dingmengru/model/pretrained_bert/chinese_wwm_ext_pytorch",
           "requires_grad": false
        },
        "dropout": 0.25,
        "loss":"focal_loss"
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
        "num_epochs": 5,
        "validation_metric": "+accuracy",
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 0
    }
}