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
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter, SimpleWordSplitter
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ListField, SpanField
from allennlp.service.predictors import Predictor

from nrl.service.predictors.qasrl_parser import QaSrlParserPredictor
from nrl.service.predictors.unlabeled_qasrl_parser import UnlabeledQasrlParserPredictor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
@Predictor.register("unlabeled_qanom_parser")
class UnlabeledQaNomParserPredictor(UnlabeledQasrlParserPredictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader, prediction_threshold: float = None):
        super().__init__(model, dataset_reader, prediction_threshold=prediction_threshold or 0.2)
        # self._tokenizer = SimpleWordSplitter()
        self._pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')

    @overrides
    def _sentence_to_qasrl_instances(self, json_dict: JsonDict) -> Tuple[List[Instance], JsonDict]:
        sentence = json_dict["sentence"]
        # tokens = self._tokenizer.split_words(sentence)
        words, pos_tags = zip(*list(self._pos_tagger.tag(sentence.split(" "))))
        sent_text = " ".join(words)

        result_dict: JsonDict = {"words": words, "verbs": []}

        instances: List[Instance] = []

        verb_indexes = []
        for i, word in enumerate(words):
            # here we should decide whether the i token is a predicate.
            # Take all candidate nouns (in the same way filtered for QANom annotations - using lexical resources)
            if self.classify_is_candidate_for_verbal_noun(words, i, pos_tags):
                result_dict["verbs"].append(word)

                instance = self._dataset_reader._make_instance_from_text(sent_text, i)
                instances.append(instance)
                verb_indexes.append(i)

        return instances, result_dict, words, verb_indexes


    def classify_is_candidate_for_verbal_noun(self, sentence_words: List[str], target_idx: int, pos_tags: List[str]) -> bool:
        # Take all candidate nouns (in the same way filtered for QANom annotations - using lexical resources)
        from qanom.candidate_extraction.prepare_qanom_prompts import COMMON_NOUNS_POS, is_candidate_noun
        target_word = sentence_words[target_idx]
        if pos_tags[target_idx] in COMMON_NOUNS_POS and is_candidate_noun(target_word):
            return True
        else:
            return False
