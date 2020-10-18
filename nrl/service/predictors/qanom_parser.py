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
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter, JustSpacesWordSplitter
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ListField, SpanField, MetadataField
from allennlp.service.predictors import Predictor

from nrl.data.util import QuestionSlots
from nrl.service.predictors.qasrl_parser import QaSrlParserPredictor
from nrl.service.predictors.unlabeled_qanom_parser import UnlabeledQaNomParserPredictor


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
@Predictor.register("qanom_parser")
class QaNomParserPredictor(UnlabeledQaNomParserPredictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader, prediction_threshold: float = None):
        super().__init__(model, dataset_reader, prediction_threshold=prediction_threshold or 0.01)
        # THIS ASSUMES THAT SENTENCES COME PRE-TOKENIZED!
        self.splitter = JustSpacesWordSplitter()

    def _sentence_to_qasrl_instances(self, json_dict: JsonDict) -> Tuple[
        List[Instance], JsonDict, List[str], List[int]]:
        sentence = json_dict["sentence"]
        tokens = self.splitter.split_words(sentence)
        words = [token.text for token in tokens]
        text = " ".join(words)
        predicate_indices = json_dict['predicate_indices']
        result_dict: JsonDict = {"words": words, "verbs": json_dict['verb_forms']}
        instances: List[Instance] = [self._dataset_reader._make_instance_from_text(text, pred_idx)
                                     for pred_idx in predicate_indices]
        if 'qasrl_id' in json_dict:
            result_dict['qasrl_id'] = json_dict['qasrl_id']
            for instance in instances:
                instance.add_field("qasrl_id", MetadataField(json_dict['qasrl_id']))
        return instances, result_dict, words, predicate_indices

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = 0) -> JsonDict:
        return QaSrlParserPredictor.predict_json(self, inputs, cuda_device)

    def make_question_text(self, slots: QuestionSlots.Type, verb):
        # considering changes at slots, especially in 'verb' slot
        slots = list(slots)
        verb_slot_idx = QuestionSlots.slots.index("verb_slot_inflection")
        verb_slot = slots[verb_slot_idx]
        split = verb_slot.split(" ")
        # HACK: For some reason verb inflection file uses CamelCase
        # while our vocabulary is PascalCase.. :-S
        split[-1] = split[-1][0].lower() + split[-1][1:]
        verb = verb.lower()
        if verb in self._verb_map:
            split[-1] = self._verb_map[verb][split[-1]]
        else:
            split[-1] = verb
        # re-instanciate the verb slot - put the actual verb (inflected as desired) instead of the inflection description
        slots[verb_slot_idx] = " ".join(split)
        sent_text = " ".join([slot for slot in QuestionSlots.get_surface_slots(slots)
                              if slot != "_"]) + "?"
        sent_text = sent_text[0].upper() + sent_text[1:]
        return sent_text