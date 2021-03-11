local batch_size = std.parseInt(std.extVar('batch_size'));
local dropout = std.parseJson(std.extVar('dropout'));
local lr = std.parseJson(std.extVar('lr'));

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
    "train_data_path": "/root/dingmengru/data/action/include_animal_v1/train_greater4.jsonl",
    "validation_data_path": "/root/dingmengru/data/action/include_animal_v1/valid_greater4.jsonl",
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
        "loss":"focal_loss"
},

    "data_loader": {
        "batch_size": batch_size,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": lr
        },
        "num_epochs": 5,
        "validation_metric": "+accuracy",
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 0
    }
}