local batch_size = std.parseInt(std.extVar('batch_size'));
local lr = std.parseJson(std.extVar('lr'));
local dropout = std.parseJson(std.extVar('dropout'));
local dropout_prob = std.parseJson(std.extVar('dropout_prob'));


{
    "dataset_reader": {
        "type": "action_cls_traindata_reader",
        "max_tokens": 512
    },
    "train_data_path": "/root/dingmengru/data/action/include_animal_v1/train_greater4.jsonl",
    "validation_data_path": "/root/dingmengru/data/action/include_animal_v1/valid_greater4.jsonl",
    "model":{
        "type": "text_classifier_fscore_focal_loss",
        "text_field_embedder":{
            "type": "basic",
            "token_embedders":{
                "tokens":{
                    "type": "embedding",
                    "embedding_dim": 200,
                    "pretrained_file" : "/root/dingmengru/model/word_emb/tencent_all/Tencent_AILab_ChineseEmbedding.txt"
                }
            }
        },
        "seq2seq_encoder":{
            "type": "pytorch_transformer",
            "input_dim": 200,
            "num_layers": 6,
            "feedforward_hidden_dim": 1024,
            "num_attention_heads": 8,
            "positional_encoding": "sinusoidal",
            "positional_embedding_size": 200,
            "dropout_prob": dropout_prob,
            "activation": "relu"
        },
        "seq2vec_encoder":{
            "type": "cls_pooler",
            "embedding_dim": 200,
            "cls_is_last_token": false,
        },
        "loss": "focal_loss",
        "dropout": dropout
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
        "num_epochs": 50,
        "validation_metric": "+accuracy",
        "grad_norm": 10.0,
        "patience": 6,
        "cuda_device": 0
    }
}