# bert classifier
allennlp train training_configs/bert_clf_test.jsonnet -f -s debug_bert_clf --include-package tal_allennlp
allennlp evaluate debug_bert_clf/model.tar.gz /root/dingmengru/data/action/v05/test_ding.jsonl --include-package tal_allennlp

# bert+crf tagger
allennlp train training_configs/bert_crf_tagger.jsonnet -f -s bert_crf_tagger --include-package tal_allennlp
allennlp evaluate bert_crf_tagger/model.tar.gz /root/dingmengru/data/action/peopel_daily_2014_ner/test.jsonl --include-package tal_allennlp