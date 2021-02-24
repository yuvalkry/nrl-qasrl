from typing import Dict, List, TextIO, Optional, Tuple

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F
import math

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy

from nrl.modules.question_generator.question_generator import QuestionGenerator
from nrl.metrics.question_prediction_metric import QuestionPredictionMetric

from nrl.metrics.threshold_metric import ThresholdMetric
from nrl.modules.span_rep_assembly import SpanRepAssembly
from nrl.common.span import Span


@Model.register("qanom_parser")
class QANomParser(Model):
    def __init__(self, vocab: Vocabulary,
                 architecture: str,
                 span_representation: str,
                 combined_metric: Dict[str, any],
                 question_generator: QuestionGenerator,
                 text_field_embedder: TextFieldEmbedder,
                 stacked_encoder_span: Seq2SeqEncoder,
                 stacked_encoder_ques: Seq2SeqEncoder,
                 predicate_feature_dim: int,
                 dim_hidden: int = 100,
                 embedding_dropout: float = 0.0,
                 thresholds: Tuple[float, ...] = (0.5,),    # high threshold lead to less yield (more precision).
                 iou_threshold: float = .3,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):

        super(QANomParser, self).__init__(vocab, regularizer)

        self.architecture = architecture
        self.span_representation = span_representation.lower()
        self.dim_hidden = dim_hidden
        self.combined_metric = combined_metric

        for target in list(combined_metric.keys()):
            if combined_metric[target].startswith('-'):
                self.combined_metric[target+'_sign'] = -1
            else:
                self.combined_metric[target+'_sign'] = 1
            if combined_metric[target][0] in "-+":
                self.combined_metric[target] = combined_metric[target][1:]

        self.text_field_embedder = text_field_embedder
        self.predicate_feature_embedding = Embedding(predicate_feature_dim, 2)

        self.embedding_dropout = Dropout(p=embedding_dropout)

        def match_heuristic(x, y) -> bool: return x.iou(y) >= iou_threshold
        # match_heuristic = lambda x, y: x.iou(y) >= iou_threshold
        self.threshold_metric = ThresholdMetric(thresholds=thresholds, match_heuristic=match_heuristic)

        self.stacked_encoder_span = stacked_encoder_span
        self.stacked_encoder_ques = stacked_encoder_ques

        self.bio_classifier = Linear(in_features=stacked_encoder_span.get_output_dim(),
                                     out_features=vocab.get_vocab_size('bio_labels'))

        self.bio_f1 = SpanBasedF1Measure(vocab, 'bio_labels')
        self.bio_acc = CategoricalAccuracy(tie_break=False)

        self.span_hidden = SpanRepAssembly(self.stacked_encoder_span.get_output_dim(), self.stacked_encoder_span.get_output_dim(), self.dim_hidden)
        self.pred = TimeDistributed(Linear(self.dim_hidden, 1))

        self.span_extractor = EndpointSpanExtractor(self.stacked_encoder_ques.get_output_dim(), combination="x,y")

        self.question_generator = question_generator
        self.slot_labels = question_generator.get_slot_labels()

        self.question_metric = QuestionPredictionMetric(vocab, question_generator.get_slot_labels())

    def forward(self,  # type: ignore
                # these all are fields in our Instances
                text: Dict[str, torch.LongTensor],
                predicate_indicator: torch.LongTensor,
                labeled_spans: torch.LongTensor = None,
                annotations: Dict = None,
                **kwargs):

        if self.architecture == 'parallel':
            return self.forward_parallel(text, predicate_indicator, labeled_spans, annotations, **kwargs)

    def forward_parallel(self,  # type: ignore
                # these all are fields in our Instances
                text: Dict[str, torch.LongTensor],
                predicate_indicator: torch.LongTensor,
                labeled_spans: torch.LongTensor = None,
                annotations: Dict = None,
                **kwargs):

        output_dict = {}

        embedded_text_input = self.embedding_dropout(self.text_field_embedder(text))

        embedded_predicate_indicator = self.predicate_feature_embedding(predicate_indicator.long())
        embedded_text_with_predicate_indicator = torch.cat([embedded_text_input, embedded_predicate_indicator], -1)
        batch_size, sequence_length, embedding_dim_with_predicate_feature = embedded_text_with_predicate_indicator.size()

        if self.stacked_encoder_span.get_input_dim() != embedding_dim_with_predicate_feature:
            raise ConfigurationError("The SRL model uses an indicator feature, which makes "
                                     "the embedding dimension one larger than the value "
                                     "specified. Therefore, the 'input_dim' of the stacked_encoder "
                                     "must be equal to total_embedding_dim + 1.")

        mask = get_text_field_mask(text)  # torch.Size([80, 25])

        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.make_output_human_readable.
        output_dict["mask"] = mask  # torch.Size([80, 25])

        encoded_text_span = self.stacked_encoder_span(embedded_text_with_predicate_indicator, mask)  # torch.Size([80, 25, 868])

        if self.span_representation == "spans":
            span_hidden, span_mask_span = self.span_hidden(encoded_text_span, encoded_text_span, mask, mask)
            # span_hidden: torch.Size([80, 325, 100])
            # span_mask_span: torch.Size([80, 325])
            logits_span = self.pred(F.relu(span_hidden)).squeeze(-1)  # torch.Size([80, 325])
            probs_span = torch.sigmoid(logits_span) * span_mask_span.float()  # torch.Size([80, 325])

            output_dict["logits"] = logits_span
            output_dict["probs"] = probs_span
            output_dict["span_mask_span"] = span_mask_span

            if labeled_spans is not None:
                span_label_mask = (labeled_spans[:, :, 0] >= 0).long()
                prediction_mask = self.get_prediction_map(labeled_spans, span_label_mask, sequence_length,
                                                          annotations=annotations)
                loss_span = F.binary_cross_entropy_with_logits(logits_span, prediction_mask, weight=span_mask_span.float(),
                                                          size_average=False)
                output_dict["loss_span"] = loss_span
                # if not self.training:
                spans = self.to_scored_spans(probs_span, span_mask_span)
                self.threshold_metric(spans, annotations)

        if self.span_representation == "bio" or self.span_representation == "iob":
            """
            Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
            constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
            ``"tags"`` key to the dictionary with the result.
            """

            classified = self.bio_classifier(encoded_text_span)
            iob_tags = kwargs['bio_label']

            self.bio_f1(classified, iob_tags, mask)

            """
            classified_acc_size = list(classified.size())
            classified_acc_size.insert(1, 1)
            classified_for_acc = torch.reshape(classified, classified_acc_size)
            classified_for_acc = classified_for_acc.max(-1)[1]
            """
            self.bio_acc(classified, iob_tags, mask)

            if iob_tags is not None:
                output_dict["loss_span"] = sequence_cross_entropy_with_logits(classified, iob_tags, mask)

            all_tags = []
            transition_matrix = self.get_viterbi_pairwise_potentials()
            sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

            if classified.dim() == 3:
                predictions_list = [classified[i].data.cpu() for i in range(classified.size(0))]
            else:
                predictions_list = [classified.cpu()]

            for predictions, length in zip(predictions_list, sequence_lengths):
                max_likelihood_sequence, _ = viterbi_decode(predictions[:length], transition_matrix)
                max_likelihood_sequence[-1] = max_likelihood_sequence[-1].item()
                tags = [self.vocab.get_token_from_index(x, namespace="bio_labels")
                        for x in max_likelihood_sequence]
                all_tags.append(tags)
            output_dict['tags'] = all_tags

        # start from quesgen

        span_mask_ques = (labeled_spans[:, :, 0] >= 0).long()  # torch.Size([80, 6])
        output_dict['span_mask_ques'] = span_mask_ques

        if labeled_spans is None:
            print("Labeled spans is None")

        encoded_text_ques = self.stacked_encoder_ques(embedded_text_with_predicate_indicator, mask)  # torch.Size([80, 25, 300])
        span_reps = self.span_extractor(encoded_text_ques, labeled_spans, sequence_mask=mask, span_indices_mask=span_mask_ques)  # torch.Size([80, 6, 600])

        span_slot_labels = []
        for n in self.slot_labels:
            slot_label = 'span_slot_%s' % n
            if slot_label in kwargs and kwargs[slot_label] is not None:
                span_slot_labels.append(kwargs[slot_label] * span_mask_ques)

        if len(span_slot_labels) == 0:
            span_slot_labels = None

        slot_logits = self.question_generator(span_reps, slot_labels=span_slot_labels)
        for i, n in enumerate(self.slot_labels):
            # Replace scores for padding and unk
            slot_logits[i][:, :, 0:2] -= 9999999
            output_dict["slot_logits_%s" % n] = slot_logits[i]  # torch.Size([80, 6, 10])

        loss = None
        if span_slot_labels is not None:
            for i, n in enumerate(self.slot_labels):
                slot_loss = sequence_cross_entropy_with_logits(slot_logits[i], span_slot_labels[i], span_mask_ques.float())
                if loss is None:
                    loss = slot_loss
                else:
                    loss += slot_loss
            self.question_metric(slot_logits, span_slot_labels, labeled_spans, mask=span_mask_ques, sequence_mask=mask)
            output_dict["loss_quesgen"] = loss  # scalar
            output_dict["slot_logits"] = slot_logits  # for evaluation  9 x torch.Size([80, 6, 10])

        # end from quesgen

        output_dict["loss"] = output_dict["loss_span"]+output_dict["loss_quesgen"]

        return output_dict

    def to_scored_spans(self, probs, score_mask):
        probs = probs.data.cpu()
        score_mask = score_mask.data.cpu()
        batch_size, num_spans = probs.size()
        spans = []
        for b in range(batch_size):
            batch_spans = []
            for start, end, i in self.start_end_range(num_spans):
                if score_mask[b, i] == 1 and probs[b, i] > 0:
                    batch_spans.append((Span(start, end), probs[b, i]))
            spans.append(batch_spans)
        return spans

    def start_end_range(self, num_spans):
        n = int(.5 * (math.sqrt(8 * num_spans + 1) -1))

        result = []
        i = 0
        for start in range(n):
            for end in range(start, n):
                result.append((start, end, i))
                i += 1

        return result

    def get_prediction_map(self, spans, span_mask, seq_length, annotations=None):
        batchsize, num_spans, _ = spans.size()
        num_labels = int((seq_length * (seq_length+1))/2)
        labels = spans.data.new().resize_(batchsize, num_labels).zero_().float()
        spans = spans.data
        arg_indexes = (2 * spans[:,:,0] * seq_length - spans[:,:,0].float().pow(2).long() + spans[:,:,0]) // 2 + (spans[:,:,1] - spans[:,:,0])
        arg_indexes = arg_indexes * span_mask.data

        for b in range(batchsize):
            for s in range(num_spans):
                if span_mask.data[b, s] > 0:
                    labels[b, arg_indexes[b, s]] = 1

        return torch.autograd.Variable(labels)

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor], remove_overlap=True) -> Dict[str, torch.Tensor]:
        output_dict = self.make_output_human_readable_span(output_dict, remove_overlap=remove_overlap)
        output_dict = self.make_output_human_readable_quesgen(output_dict, remove_overlap=remove_overlap)
        return output_dict

    def make_output_human_readable_span(self, output_dict: Dict[str, torch.Tensor], remove_overlap=True) -> Dict[str, torch.Tensor]:
        probs = output_dict['probs']
        mask = output_dict['span_mask']
        spans = self.to_scored_spans(probs, mask)
        output_dict['spans'] = spans
        return output_dict


        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].data.cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        transition_matrix = self.get_viterbi_paispan_detector.pyrwise_potentials()
        for predictions, length in zip(predictions_list, sequence_lengths):
            max_likelihood_sequence, _ = viterbi_decode(predictions[:length], transition_matrix)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in max_likelihood_sequence]
            all_tags.append(tags)
        output_dict['tags'] = all_tags
        return output_dict

    def make_output_human_readable_quesgen(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        span_mask = output_dict['span_mask'].data.cpu()
        batch_size, num_spans = span_mask.size()

        slot_preds = []
        for l in self.slot_labels:
            maxinds = output_dict['slot_logits_%s' % (l)].data.cpu().max(-1)[1]
            slot_preds.append(maxinds)

        questions = []
        for b in range(batch_size):
            batch_questions = []
            for i in range(num_spans):
                if span_mask[b, i] == 1:

                    slots = []
                    for l, n in enumerate(self.slot_labels):
                        slot_word = self.vocab.get_index_to_token_vocabulary("slot_%s" % n)[int(slot_preds[l][b, i])]
                        slots.append(slot_word)

                    slots = tuple(slots)

                    batch_questions.append(slots)

            questions.append(batch_questions)

        output_dict['questions'] = questions
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False):
        metric_dict_quesgen = self.question_metric.get_metric(reset=reset)

        if self.span_representation == "spans":
            metric_dict_span = self.threshold_metric.get_metric(reset=reset)
        else:  # IOB
            metric_dict_span = self.bio_f1.get_metric(reset=reset)
            metric_dict_span['bio_acc'] = self.bio_acc.get_metric(reset=reset)

        if self.training:
            metric_dict_quesgen = {x: y for x, y in metric_dict_quesgen.items() if
                                "word-accuracy" not in x or x == "word-accuracy-overall"}
            # This can be a lot of metrics, as there are 3 per class.
            # During training, we only really care about the overall
            # metrics, so we filter for them here.
            # TODO(Mark): This is fragile and should be replaced with some verbosity level in Trainer.
            metric_dict_span = {x: y for x, y in metric_dict_span.items() if "wh" not in x}

        span_metrics = list(metric_dict_span.keys())

        for metric in span_metrics:
            if metric in metric_dict_quesgen:
                metric_dict_span[metric+'span'] = metric_dict_span[metric]
                del metric_dict_span[metric]
                metric_dict_quesgen[metric + 'quesgen'] = metric_dict_quesgen[metric]
                del metric_dict_quesgen[metric]

        metric_dict = metric_dict_span.copy()
        metric_dict.update(metric_dict_quesgen.copy())

        metric_dict['combined'] = 0
        for target in ['span-'+self.span_representation, 'quesgen']:
            metric_dict['combined'] += self.combined_metric[target+'_sign']*metric_dict[self.combined_metric[target]]

        return metric_dict

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded
        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
        constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.

        Returns
        -------
        transition_matrix : torch.Tensor
            A (num_labels, num_labels) matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("bio_labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

    # @classmethod
    # def from_params(cls, vocab: Vocabulary, params: Params) -> 'SpanDetector':
    #     embedder_params = params.pop("text_field_embedder")
    #     text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
    #     stacked_encoder = Seq2SeqEncoder.from_params(params.pop("stacked_encoder"))
    #     predicate_feature_dim = params.pop("predicate_feature_dim")
    #     dim_hidden = params.pop("hidden_dim", 100)
    #
    #     initializer = InitializerApplicator.from_params(params.pop('initializer', []))
    #     regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
    #
    #     params.assert_empty(cls.__name__)
    #
    #     return cls(vocab=vocab,
    #                text_field_embedder=text_field_embedder,
    #                stacked_encoder=stacked_encoder,
    #                predicate_feature_dim=predicate_feature_dim,
    #                dim_hidden = dim_hidden,
    #                initializer=initializer,
    #                regularizer=regularizer)

def perceptron_loss(logits, prediction_mask, score_mask):
    batch_size, seq_length, _ = logits.size()

    max_bad, _ = (logits + (prediction_mask == 0).float().log() + score_mask.float().log()).view(batch_size, -1).max(1)
    min_good, _ = (logits - (prediction_mask == 1).float().log()  - score_mask.float().log()).view(batch_size, -1).min(1)

    bad_scores = (min_good.view(batch_size, 1, 1) - logits)
    bad_violations = bad_scores * (bad_scores < 0).float() * (prediction_mask == 0).float() * score_mask.float()
    #bad_norms = bad_violations.float().view(batch_size, -1).sum(1).view(batch_size, 1, 1).expand(batch_size, seq_length, seq_length)
    #bad_scores = bad_violations.masked_select(bad_violations < 0)
    #bad_scores = - bad_scores / bad_norms.masked_select(bad_violations < 0)
    #bad_violations = bad_scores

    good_scores = (max_bad.view(batch_size, 1, 1) - logits)
    good_violations = good_scores * (good_scores > 0).float() * prediction_mask.float() * score_mask.float()
    #good_norms = good_violations.float().view(batch_size, -1).sum(1).view(batch_size, 1, 1).expand(batch_size, seq_length, seq_length)
    #good_scores = good_violations.masked_select(good_violations > 0)
    #good_scores = good_scores / good_norms.masked_select(good_violations > 0)
    #good_violations = good_scores

    loss = good_violations.sum() - bad_violations.sum()

    return loss

def write_to_conll_eval_file(prediction_file: TextIO,
                             gold_file: TextIO,
                             verb_index: Optional[int],
                             sentence: List[str],
                             prediction: List[str],
                             gold_labels: List[str]):
    """
    Prints predicate argument predictions and gold labels for a single verbal
    predicate in a sentence to two provided file references.

    Parameters
    ----------
    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    verb_index : Optional[int], required.
        The index of the verbal predicate in the sentence which
        the gold labels are the arguments for, or None if the sentence
        contains no verbal predicate.
    sentence : List[str], required.
        The word tokens.
    prediction : List[str], required.
        The predicted BIO labels.
    gold_labels : List[str], required.
        The gold BIO labels.
    """
    verb_only_sentence = ["-"] * len(sentence)
    if verb_index:
        verb_only_sentence[verb_index] = sentence[verb_index]

    conll_format_predictions = convert_bio_tags_to_conll_format(prediction)
    conll_format_gold_labels = convert_bio_tags_to_conll_format(gold_labels)

    for word, predicted, gold in zip(verb_only_sentence,
                                     conll_format_predictions,
                                     conll_format_gold_labels):
        prediction_file.write(word.ljust(15))
        prediction_file.write(predicted.rjust(15) + "\n")
        gold_file.write(word.ljust(15))
        gold_file.write(gold.rjust(15) + "\n")
    prediction_file.write("\n")
    gold_file.write("\n")


def convert_bio_tags_to_conll_format(labels: List[str]):
    """
    Converts BIO formatted SRL tags to the format required for evaluation with the
    official CONLL 2005 perl script. Spans are represented by bracketed labels,
    with the labels of words inside spans being the same as those outside spans.
    Beginning spans always have a opening bracket and a closing asterisk (e.g. "(ARG-1*" )
    and closing spans always have a closing bracket (e.g. "*)" ). This applies even for
    length 1 spans, (e.g "(ARG-0*)").

    A full example of the conversion performed:

    [B-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, O]
    [ "(ARG-1*", "*", "*", "*", "*)", "*"]

    Parameters
    ----------
    labels : List[str], required.
        A list of BIO tags to convert to the CONLL span based format.

    Returns
    -------
    A list of labels in the CONLL span based format.
    """
    sentence_length = len(labels)
    conll_labels = []
    for i, label in enumerate(labels):
        if label == "O":
            conll_labels.append("*")
            continue
        new_label = "*"
        # Are we at the beginning of a new span, at the first word in the sentence,
        # or is the label different from the previous one? If so, we are seeing a new label.
        if label[0] == "B" or i == 0 or label[1:] != labels[i - 1][1:]:
            new_label = "(" + label[2:] + new_label
        # Are we at the end of the sentence, is the next word a new span, or is the next
        # word not in a span? If so, we need to close the label span.
        if i == sentence_length - 1 or labels[i + 1][0] == "B" or label[1:] != labels[i + 1][1:]:
            new_label = new_label + ")"
        conll_labels.append(new_label)
    return conll_labels

    def quesgen_get_metrics(self, reset: bool = False):
        metric_dict = self.question_metric.get_metric(reset=reset)
        if self.training:
            metric_dict = {x: y for x, y in metric_dict.items() if
                           "word-accuracy" not in x or x == "word-accuracy-overall"}

        return metric_dict

    def quesgen_make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        span_mask = output_dict['span_mask'].data.cpu()
        batch_size, num_spans = span_mask.size()

        slot_preds = []
        for l in self.slot_labels:
            maxinds = output_dict['slot_logits_%s' % (l)].data.cpu().max(-1)[1]
            slot_preds.append(maxinds)

        questions = []
        for b in range(batch_size):
            batch_questions = []
            for i in range(num_spans):
                if span_mask[b, i] == 1:

                    slots = []
                    for l, n in enumerate(self.slot_labels):
                        slot_word = self.vocab.get_index_to_token_vocabulary("slot_%s" % n)[int(slot_preds[l][b, i])]
                        slots.append(slot_word)

                    slots = tuple(slots)

                    batch_questions.append(slots)

            questions.append(batch_questions)

        output_dict['questions'] = questions
        return output_dict
