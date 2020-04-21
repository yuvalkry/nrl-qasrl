from typing import List, Tuple
from overrides import overrides
import logging
import gzip
import torch
import numpy

from torch.nn.parameter import Parameter

from allennlp.models import Model
from allennlp.common.util import JsonDict
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ListField, SpanField
from allennlp.service.predictors import Predictor

from nrl.service.predictors.qasrl_parser import QaSrlParserPredictor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Predictor.register("unlabeled_qasrl_parser")
class UnlabeledQasrlParserPredictor(QaSrlParserPredictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader, prediction_threshold: float = None):
        super().__init__(model, dataset_reader, prediction_threshold)

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = 0) -> JsonDict:

        instances, results, words, verb_indexes = self._sentence_to_qasrl_instances(inputs)

        verbs_for_instances = results["verbs"]
        results["verbs"] = []

        instances_with_spans = []
        instance_spans = []
        if instances:
            span_outputs = self._model.span_detector.forward_on_instances(instances)

            for instance, span_output in zip(instances, span_outputs):
                field_dict = instance.fields
                text_field = field_dict['text']

                spans = [s[0] for s in span_output['spans'] if s[1] >= self._prediction_threshold]
                if len(spans) > 0:
                    instance_spans.append(spans)

                    labeled_span_field = ListField([SpanField(span.start(), span.end(), text_field) for span in spans])
                    field_dict['labeled_spans'] = labeled_span_field
                    instances_with_spans.append(Instance(field_dict))

        # instead of predicting questions - predict only arguments
        if instances_with_spans:
            for spans, verb, index in zip(instance_spans, verbs_for_instances, verb_indexes):
                answers = []
                for span in spans:
                    span_text = " ".join([words[i] for i in range(span.start(), span.end()+1)])
                    span_rep = {"start": span.start(), "end": span.end(), "text":span_text}
                    answers.append(span_rep)

                results["verbs"].append({"verb": verb, "arguments": answers, "index": index})

        return results


