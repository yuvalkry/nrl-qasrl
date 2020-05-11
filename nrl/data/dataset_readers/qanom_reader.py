from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging
from overrides import overrides
from dataclasses import asdict

from allennlp.data import TokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from nrl.data.dataset_readers import qasrl_reader
from nrl.data.util import AnnotatedSpan, QuestionSlots
from qanom.annotations.common import read_annot_csv, iterate_qanom_responses
from qanom.annotations.decode_encode_answers import Question as QanomQuestion
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@DatasetReader.register("qanom")
class QANomReader(qasrl_reader.QaSrlReader):
    """
    Read a QANom annotation file (csv), as formatted by the qanom.annotations functions.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 training_data_precentage = 1.0,
                 has_provinence = False,
                 bio_labels = True,
                 slot_labels = None,
                 min_answers = 0,
                 min_valid_answers = 0,
                 question_sources = None):
        super(QANomReader, self).__init__(token_indexers=token_indexers,
                                          training_data_precentage=training_data_precentage,
                                          has_provinence=has_provinence,
                                          bio_labels=bio_labels,
                                          slot_labels=slot_labels,
                                          min_answers=min_answers,
                                          min_valid_answers=min_valid_answers,
                                          question_sources=question_sources)


    @overrides
    def _read(self, file_list: str):
        for file_path in file_list.split(","):
            if file_path.strip() == "":
                continue

            logger.info("Reading QANom instances from dataset file at: %s", file_path)
            data = []
            # for other formats - probably a qasrl file - redirect to QaSrlReader
            if file_path.endswith('.gz') or file_path.endswith(".jsonl"):
                super()._read(file_list)
            # here we are expecting CSV files
            elif file_path.endswith(".csv"):
                annot_df: pd.DataFrame = read_annot_csv(file_path)
                # iterate over all candidate nouns (i.e. QANom annotation prompts)
                for sent_id, sent, target_index, worker_id, response in iterate_qanom_responses(annot_df):
                    sentence_tokens: List[str] = sent.split(" ")
                    self._num_verbs += 1

                    # Pipeline qanom model: Instances are only verbal nouns with arguments
                    if not response.is_verbal or not response.roles:
                        self._no_ann += 1
                        continue

                    # translate qanom.Response.Role into nrl classes
                    replace_to_underscore = lambda s: "_" if not s else s # for qasrl question slots
                    annotatedSpans: List[AnnotatedSpan] = []
                    for qa in response.roles:
                        qSlots = QuestionSlots.get_from_qanom_question(qa.question)
                        qSlots = list(map(replace_to_underscore, qSlots))

                        from nrl.common.span import Span
                        # turn exclusive spans (as in qanom annotations - 1:2 for single word)
                        #   into inclusive spans (as the model expects - 1:1)
                        answerSpans: List[Span] = [Span(s[0], s[1]-1) for s in qa.arguments]

                        self._qa_pairs += 1
                        annotatedSpans.append(AnnotatedSpan(slots=qSlots, all_spans=answerSpans,
                                                            pred_index=target_index, provenance=worker_id))

                    # produce and yield Instance for target noun - with all QA information in it
                    self._instances += 1
                    yield self._make_instance_from_text(sentence_tokens, target_index, annotations=annotatedSpans,
                                                        sent_id=sent_id)

        # log all produced
        logger.info("Produced %d instances" % self._instances)
        logger.info("\t%d Verbs" % self._num_verbs)
        logger.info("\t%d QA pairs" % self._qa_pairs)
        logger.info("\t%d no annotation" % self._no_ann)

    @classmethod
    def qanomQuestionToVerbSlot(cls, qanomQuestion: QanomQuestion) -> str:
        """ Recover 'verb' slot from a qanom...Question object.
        'verb' information is within Q.aux, Q.is_passive and Q.is_negated
         """
        raise NotImplementedError
