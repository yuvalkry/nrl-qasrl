from typing import List, Tuple

from overrides import overrides
import logging
import gzip
import torch
import numpy

from torch.nn.parameter import Parameter

from allennlp.models import Model
from allennlp.common.util import JsonDict
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ListField, SpanField
from allennlp.predictors import Predictor
from allennlp.common.file_utils import cached_path

from nrl.data.util import cleanse_sentence_text, QuestionSlots

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DO_VERBS = {"do", "does", "doing", "did", "done"}
BE_VERBS = {"be", "being", "been", "am", "'m", "is", "'s", "ai", "are", "'re", "was", "were"}
WILL_VERBS = {"will", "'ll", "wo"}
HAVE_VERBS = {"have", "having", "'ve", "has", "had", "'d"}
MODAL_VERBS = {"can", "ca", "could", "may", "might", "must", "shall", "should", "ought"}

AUX_VERBS = DO_VERBS | BE_VERBS | WILL_VERBS | HAVE_VERBS | MODAL_VERBS

def read_verb_file(verb_file):
    verb_map = {}
    with open(verb_file, 'r', encoding="utf-8") as f:
        for l in f.readlines():
            inflections = l.strip().split('\t')
            stem, presentsingular3rd, presentparticiple, past, pastparticiple = inflections
            for inf in inflections:
                verb_map[inf] = {"stem" : stem, "presentSingular3rd" : presentsingular3rd, "presentParticiple":presentparticiple, "past":past, "pastParticiple":pastparticiple}
    return verb_map

def read_pretrained_file(embeddings_filename, embedding_dim=100):
    embeddings = {}
    logger.info("Reading embeddings from file")
    with gzip.open(cached_path(embeddings_filename), 'rb') as embeddings_file:
        for line in embeddings_file:
            fields = line.decode('utf-8').strip().split(' ')
            if len(fields) - 1 != embedding_dim:
                # Sometimes there are funny unicode parsing problems that lead to different
                # fields lengths (e.g., a word with a unicode space character that splits
                # into more than one column).  We skip those lines.  Note that if you have
                # some kind of long header, this could result in all of your lines getting
                # skipped.  It's hard to check for that here; you just have to look in the
                # embedding_misses_file and at the model summary to make sure things look
                # like they are supposed to.
                logger.warning("Found line with wrong number of dimensions (expected %d, was %d): %s",
                               embedding_dim, len(fields) - 1, line)
                continue
            word = fields[0]
            vector = torch.from_numpy(numpy.asarray(fields[1:], dtype='float32'))
            embeddings[word] = vector
    return embeddings

@Predictor.register("qasrl_parser")
class QaSrlParserPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader, prediction_threshold= None) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyTokenizer(language='en_core_web_sm', pos_tags=True)
        self._model_vocab = model.vocab
        self._prediction_threshold = prediction_threshold or 0.5

        self._verb_map = read_verb_file("data/wiktionary/en_verb_inflections.txt")

        self._pretrained_vectors = read_pretrained_file("https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz")

    def _sentence_to_qasrl_instances(self, json_dict: JsonDict) -> Tuple[List[Instance], JsonDict, List[str], List[int]]:
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.tokenize(sentence)
        words = [token.text for token in tokens]
        text = " ".join(words)

        result_dict: JsonDict = {"words": words, "verbs": []}

        instances: List[Instance] = []

        verb_indexes = []
        for i, word in enumerate(tokens):
            if word.pos_ == "VERB" and not word.text.lower() in AUX_VERBS:
                verb = word.text
                result_dict["verbs"].append(verb)

                instance = self._dataset_reader._make_instance_from_text(text, i)
                instances.append(instance)
                verb_indexes.append(i)

        return instances, result_dict, words, verb_indexes


    def _expand_vocab(self, words):
        # Expand vocab
        cleansed_words = cleanse_sentence_text(words)
        added_words = []
        added_vectors = []
        for w in cleansed_words:
            w = w.lower()
            if self._model_vocab.get_token_index(w) == 1 and w in self._pretrained_vectors:
                added_words.append(w)
                added_vectors.append(self._pretrained_vectors[w])
        if added_words:
            first_ind = self._model_vocab.get_vocab_size("tokens")
            for w in added_words:
                self._model_vocab.add_token_to_namespace(w, "tokens")

            num_added_words = len(added_words)
            added_weights = torch.cat(added_vectors, dim=0)

            span_weights = self._model.span_detector.text_field_embedder.token_embedder_tokens.weight.data
            num_words, embsize = span_weights.size()
            new_weights = span_weights.new().resize_(num_words + num_added_words, embsize)
            new_weights[:num_words].copy_(span_weights)
            print("NUM_WORDS")
            print(num_words)
            new_weights[num_words:].copy_(torch.reshape(added_weights, (
                int(added_weights.shape[0] / new_weights[num_words:].shape[1]),
                int(added_weights.shape[0] / new_weights[num_words:].shape[0]))))
            self._model.span_detector.text_field_embedder.token_embedder_tokens.weight = Parameter(new_weights)

            ques_weights = self._model.question_predictor.text_field_embedder.token_embedder_tokens.weight.data
            num_words, embsize = ques_weights.size()
            new_weights = ques_weights.new().resize_(num_words + num_added_words, embsize)
            new_weights[:num_words].copy_(ques_weights)
            new_weights[num_words:].copy_(torch.reshape(added_weights, (
                int(added_weights.shape[0] / new_weights[num_words:].shape[1]),
                int(added_weights.shape[0] / new_weights[num_words:].shape[0]))))
            self._model.question_predictor.text_field_embedder.token_embedder_tokens.weight = Parameter(new_weights)


    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = 0) -> JsonDict:

        instances, results, words, verb_indexes = self._sentence_to_qasrl_instances(inputs)

        self._expand_vocab(words)

        verbs_for_instances = results["verbs"]

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

        verb_annotations = self.predict_questions(instances_with_spans, instance_spans, verbs_for_instances, verb_indexes, words)
        results["verbs"] = verb_annotations

        return results


    def predict_questions(self, instances_with_spans, instance_spans, verbs_for_instances, verb_indexes, words):
        verb_annotations = []
        if instances_with_spans:
            outputs = self._model.question_predictor.forward_on_instances(instances_with_spans)

            for output, spans, verb, index in zip(outputs, instance_spans, verbs_for_instances, verb_indexes):
                questions = {}
                for question, span in zip(output['questions'], spans):
                    question_text = self.make_question_text(question, verb)
                    span_text = " ".join([words[i] for i in range(span.start(), span.end()+1)])
                    span_rep = {"start": span.start(), "end": span.end(), "text":span_text}
                    questions.setdefault(question_text, []).append(span_rep)

                qa_pairs = []
                for question, spans in questions.items():
                    qa_pairs.append({"question":question, "spans":spans, "slot_logits": output["slot_logits"]}) # todo add slot logits
                                # todo but where do I get the q-slots for a specific span??

                verb_annotations.append({"verb": verb, "qa_pairs": qa_pairs, "index": index})
        return verb_annotations


    def make_question_text(self, slots: QuestionSlots.Type, verb):
        # considering changes at slots, especially in 'verb' slot
        slots = list(slots)
        verb_slot_idx = QuestionSlots.slots.index("verb_slot_inflection")
        verb_slot = slots[verb_slot_idx]
        split = verb_slot.split(" ")
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
