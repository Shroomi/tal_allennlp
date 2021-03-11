# bert classifier
allennlp train training_configs/model_configs/bert_clf_test.jsonnet -f -s debug_bert_clf --include-package tal_allennlp
allennlp evaluate debug_bert_clf/model.tar.gz /root/dingmengru/data/action/v05/test_ding.jsonl --include-package tal_allennlp

# bert classifier with hyperparameters
allennlp tune \
  training_configs/model_configs/bert_clf_test.jsonnet \
  training_configs/hyperpara_configs/bert_clf_para.json \
  --serialization-dir /root/dingmengru/model/writing_correction_action/bert_clf_optuna \
  --study-name /root/dingmengru/model/writing_correction_action/db/bert_clf_optuna \
  --timeout 36000 \
  --direction maximize \
  --include-package tal_allennlp
allennlp evaluate /root/dingmengru/model/writing_correction_action/bert_clf_optuna/trial_0/model.tar.gz \
  /root/dingmengru/data/action/include_animal_v1/test_greater4.jsonl \
  --include-package tal_allennlp

# bert+crf tagger
allennlp train training_configs/model_configs/bert_crf_tagger.jsonnet -f -s bert_crf_tagger --include-package tal_allennlp
allennlp evaluate bert_crf_tagger/model.tar.gz /root/dingmengru/data/action/peopel_daily_2014_ner/test.jsonl --include-package tal_allennlp

# biLSTM+crf tagger
allennlp train training_configs/model_configs/biLSTM_crf_tagger.jsonnet -f -s biLSTM_crf_tagger --include-package tal_allennlp
allennlp evaluate biLSTM_crf_tagger/model.tar.gz /root/dingmengru/data/action/peopel_daily_2014_ner/test.jsonl --include-package tal_allennlp