from typing import List, Tuple
from overrides import overrides
import logging
import gzip
import torch
import numpy

from nltk.parse import CoreNLPParser

from torch.nn.parameter import Parameter

from allennlp.models import Model
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ListField, SpanField
from allennlp.predictors import Predictor

from nrl.service.predictors.qasrl_parser import QaSrlParserPredictor
from nrl.service.predictors.unlabeled_qanom_parser import UnlabeledQaNomParserPredictor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
@Predictor.register("qanom_parser")
class QaNomParserPredictor(UnlabeledQaNomParserPredictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader, prediction_threshold: float = None):
        super().__init__(model, dataset_reader, prediction_threshold=prediction_threshold or 0.1)

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = 0) -> JsonDict:
        return QaSrlParserPredictor.predict_json(self, inputs, cuda_device)
