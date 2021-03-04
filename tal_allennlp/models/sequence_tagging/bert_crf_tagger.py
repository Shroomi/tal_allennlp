#!/usr/bin/env python
# encoding: utf-8
"""
@Author: dingmengru
@Contact: dingmengru1993@gmail.com
@File: bert_crf_tagger.py
@Software: PyCharm
@Time: 2021/3/4 8:48 下午
@Desc:

"""
import logging
from typing import Dict, List, Optional, Any

import torch
from torch.nn.modules.linear import Linear
from overrides import overrides
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from allennlp.nn.util import get_text_field_mask
from overrides import overrides

logger = logging.getLogger(__name__)


@Model.register("bert_crf_tagger")
class BertCrfTaggerModel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: Optional[float] = 0,
                 label_encoding: Optional[str] = 'BIO',
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        """
        :param vocab: ``Vocabulary``
        :param text_field_embedder: ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
        :param dropout:
        :param label_encoding: BIO
        :param initializer:``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
        :param regularizer:``RegularizerApplicator``, optional (default=``None``)
        :param print_bad_case: 是否将出错的case打印出来
        If provided, will be used to calculate the regularization penalty during training.
        """
        super(BertCrfTaggerModel, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size('labels')

        self._labels_predictor = Linear(self._text_field_embedder.get_output_dim(), self.num_tags)
        self.dropout = torch.nn.Dropout(dropout)
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self._f1_metric = SpanBasedF1Measure(vocab, tag_namespace='labels', label_encoding=label_encoding)
        labels = self.vocab.get_index_to_token_vocabulary('labels')
        constraints = allowed_transitions(label_encoding, labels)
        self.label_to_index = self.vocab.get_token_to_index_vocabulary('labels')
        self.crf = ConditionalRandomField(self.num_tags, constraints, include_start_end_transitions=True)
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
                passage: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]] = None,
                labels: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        question_and_passage : Dict[str, torch.LongTensor]
            From a ``TextField``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
        labels: A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : Tensor, a linear transformation to the incoming data: :math:`y = xA^T + b`
        mask : mask
        tags : int
            probabilities of the span end position (inclusive).
        probabilities : torch.FloatTensor
            The result of ``softmax(tag_logits, dim=-1)``.
        loss : torch.IntTensor
        """
        embedded_question_and_passage = self._text_field_embedder(passage)
        passage_mask = get_text_field_mask(passage)

        tag_logits = self._labels_predictor(embedded_question_and_passage)
        predicted_probability = torch.nn.functional.softmax(tag_logits, dim=-1)
        best_paths = self.crf.viterbi_tags(tag_logits, passage_mask)

        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]

        confidence = [[confidence_decode[tag_decode] for tag_decode, confidence_decode in zip(exp_tag, exp_prob)] for
                      exp_tag, exp_prob in zip(predicted_tags, predicted_probability)]

        output_dict = {"logits": tag_logits, "mask": passage_mask, "tags": predicted_tags,
                       "probabilities": predicted_probability, "confidence": confidence}

        # Compute the loss for training.
        if labels is not None:
            log_likelihood = self.crf(tag_logits, labels, passage_mask)
            output_dict["loss"] = -log_likelihood
            class_probabilities = tag_logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    if tag_id >= len(self.label_to_index):
                        tag_id = self.label_to_index['O']
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, labels, passage_mask)
            self._f1_metric(class_probabilities, labels, passage_mask)

        if metadata is not None:
            passage_tokens = []
            for i, _ in enumerate(metadata):
                passage_tokens.append(metadata[i].get('words', []))
            output_dict['passage_tokens'] = passage_tokens
        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
            [self.vocab.get_token_from_index(tag, namespace="labels")
             for tag in instance_tags]
            for instance_tags in output_dict["tags"]
        ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}

        f1_dict = self._f1_metric.get_metric(reset=reset)

        metrics_to_return.update({x: y for x, y in f1_dict.items() if "overall" in x})
        return metrics_to_return
