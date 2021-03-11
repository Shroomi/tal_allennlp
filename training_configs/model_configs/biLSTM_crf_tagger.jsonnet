{
    "dataset_reader": {
        "type": "people_daily_tagging",
        "max_tokens": 512
        // "limit" : 10
    },
    "train_data_path": "/root/dingmengru/data/action/peopel_daily_2014_ner/train.jsonl",
    "validation_data_path": "/root/dingmengru/data/action/peopel_daily_2014_ner/valid.jsonl",
    "model":{
        "type": "crf_tagger",
        "text_field_embedder":{
            "type":"basic",
            "token_embedders":{
                "tokens":{
                    "type":"embedding",
                    "embedding_dim": 200,
                    "pretrained_file" : "/root/dingmengru/model/word_emb/tencent_all/Tencent_AILab_ChineseEmbedding.txt"
                }
            }
        },
        "encoder":{
            "type":"lstm" ,
            "hidden_size" : 200 ,
            "input_size": 200 ,
            "num_layers": 1 ,
            "bidirectional": true
        }
    },

    "data_loader":{
        "batch_size": 16,
        "shuffle": true
    },

    "trainer":{
        "optimizer" :"adam",
        "num_epochs" : 10
    }
}