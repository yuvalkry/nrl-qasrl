from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
from qanom.evaluation.roles import is_equivalent_question
from qanom.annotations.decode_encode_answers import Question as QanomQuestion



class QuestionSlots:
    Type = List[str]  # a symbol for annotating the type of the slot-dict in Instance
    default_slots = ["wh", "aux", "subj", "obj", "prep", "obj2", "is_passive", "is_negated"]
    slots = default_slots
    boolean_slots = ["is_passive", "is_negated"]

    @classmethod
    def get_from_qanom_question(cls, question: QanomQuestion) -> Type:
        qDict = asdict(question)
        # Take only relevant slots. Convert booleans to strings.
        slotDict = {k:str(v) for k,v in qDict.items() if k in QuestionSlots.slots}
        return QuestionSlots.from_dict(slotDict)

    @classmethod
    def get_from_qasrl_question_label(cls, question_label: Dict[str, str]) -> Type:
        slots_from_question_label = ["wh", "aux", "subj", "obj", "prep", "obj2"]
        slots = {slot_name: question_label[slot_name] for slot_name in slots_from_question_label}
        # extract is_passive and is_negated from aux and verb slots
        verb_slot = question_label["verb"]
        aux_slot = question_label["aux"]
        is_negated = "not" in verb_slot or "n't" in aux_slot
        is_passive = "be" in verb_slot and "pastParticiple" in verb_slot
        slots.update(is_negated=str(is_negated), is_passive=str(is_passive))
        return QuestionSlots.from_dict(slots)

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> Type:
        """ get a QuestionSlot.Type from dict. Note we are expecting all d.values to be strings. """
        from collections import defaultdict
        d = defaultdict(str, d)
        asList = [d[slot] for slot in QuestionSlots.slots]
        # replace "" with "_"
        asList = [e or "_" for e in asList]
        return asList

    @classmethod
    def as_dict(cls, slots: Type) -> Dict[str, str]:
        assert len(slots)==len(QuestionSlots.slots), "QuestionSlots.slots not updated!"
        return {QuestionSlots.slots[i]: slots[i] for i in range(len(slots))}

    @classmethod
    def as_qanom_question(cls, slots: Type) -> QanomQuestion:
        qDict = dict(text="", verb_prefix="", **QuestionSlots.as_dict(slots))
        # replace "_" slots back to "" (that's what the `qanom` package expects)
        for k,v in qDict.items():
            if v=="_":
                qDict[k] = ""
        # revert slots that stand for a boolean back to bool
        for b_slot in QuestionSlots.boolean_slots:
            qDict[b_slot] = str2bool(qDict[b_slot])
        return QanomQuestion(**qDict)


def str2bool(s: str) -> bool:
    """ Inverse of str(boolean) function. """
    assert s in ['True', 'False'], "Only `True` or `False` can cast to bool"
    return s == 'True'


class AnnotatedSpan:
    def __init__(self, text = None, span = None, slots=None, all_spans=None, pred_index = None, provenance = None):
        self.text = text
        self.span = span
        self.slots = slots
        self.all_spans = all_spans
        self.pred_index = pred_index
        self.provinence = provenance

def cleanse_sentence_text(sent_text):
    sent_text = ["?" if w == "/?" else w for w in sent_text]
    sent_text = ["." if w == "/." else w for w in sent_text]
    sent_text = ["-" if w == "/-" else w for w in sent_text]
    sent_text = ["(" if w == "-LRB-" else w for w in sent_text]
    sent_text = [")" if w == "-RRB-" else w for w in sent_text]
    sent_text = ["[" if w == "-LSB-" else w for w in sent_text]
    sent_text = ["]" if w == "-RSB-" else w for w in sent_text]
    return sent_text

