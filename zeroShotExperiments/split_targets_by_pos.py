"""
Given a file of SRL-like annotation in the format "sentence string ||| i:i-rel j-k-RoleName ...", e.g. ACE or
PropBank or Ontonotes data, split the file into two files according whether the POS of the target predicate (the 'rel')
is a noun POS, a verb POS, or other.
"""
from typing import List, Tuple, Dict
Span = Tuple[int, int]
import sys
sys.path.append("/home/nlp/kleinay/Synced/nrl-parser")

from qanom.candidate_extraction import prepare_qanom_prompts as cand

import stanfordnlp

MODELS_DIR = 'stanford-core-nlp-models'
stanfordnlp.download('en', MODELS_DIR) # Download the English models
core_nlp_config = {
        'processors': 'tokenize,pos',
        'tokenize_pretokenized': True,
        'pos_model_path': f'{MODELS_DIR}/en_ewt_models/en_ewt_tagger.pt',
        'pos_pretrain_path': f'{MODELS_DIR}/en_ewt_models/en_ewt.pretrain.pt',
        'pos_batch_size': 1000
         }
corenlp = stanfordnlp.Pipeline(**core_nlp_config)

def pos_tag(sentence: str) -> List[str]:
    doc = corenlp(sentence)
    pos_tags = [word.xpos for sent in doc.sentences for word in sent.words]
    return pos_tags


def split_str_twice(string: str, del1: str, del2: str) -> Tuple[str, str, str]:
    """ Return a 3-element tuple by applying two splits on a str using two different delimiters.
        only regard the first occurrence of each delimiter.
      E.g.  split_str_twice("an_email@gmail.com", "_", "@") -> ("an", "email", "gmail.com")
      """
    first, second_and_third = string.split(del1,1)
    second, third = second_and_third.split(del2,1)
    return first, second, third

def parse_arg_str(arg_str) -> Tuple[str, Span]:
    """ arg_str is in the format 'i:j-role', return (role, (i, j)). """
    start_idx, end_idx, role = split_str_twice(arg_str, ':', '-')
    # todo take care more carefully of hyphenations in nombank, denoted with subindexing e.g. '8_1:9-arg1'
    if "_" in start_idx:
        start_idx, subidx = start_idx.split("_")
    if "_" in end_idx:
        end_idx, subidx = end_idx.split("_")
    return (role, (int(start_idx), int(end_idx)))

def split_by_pos(file_name):
    verb_lines = []
    noun_lines = []
    other_lines = []
    with open(file_name) as fin:
        tok_mismatch_counter = 0
        for line in fin.readlines():
            sent, args_str = line.strip().split(" ||| ")
            # take out 'rel' index
            """
            #### This is prettier but doesn't work with '*' in the indexing (in nombank)
            args_info: List[str] = args_str.split()
            roles: Dict[str, Span] = dict([parse_arg_str(arg_info) for arg_info in args_info])
            predicate_span = roles['rel']
            predicate_idx = predicate_span[0]   # assuming single-word predicates
            """
            args: List[str] = args_str.split()
            # Get index of predicate. Predicate is always first argument.
            predicate_loc = args[0][:args[0].find(':')]
            sub_idx = predicate_loc.find('_')
            if sub_idx < 0:
                og_nom_loc = [int(predicate_loc)]
            else:
                og_nom_loc = (int(predicate_loc[:sub_idx]), int(predicate_loc[sub_idx + 1:]))
            predicate_idx = og_nom_loc[0]    # assuming single-word predicates
            # retrieve POS of predicate
            posTaggedSent = pos_tag(sent)
            if len(posTaggedSent) != len(sent.split()):
                print(f"ERR: tokenization mismatch -- len(pos)=={len(posTaggedSent)}, while toks by split() ({len(sent.split())}): {sent.split()}")
                tok_mismatch_counter += 1
                continue
            predicate_pos = posTaggedSent[predicate_idx]

            if predicate_pos in cand.COMMON_NOUNS_POS:
                noun_lines.append(line)
            elif predicate_pos.startswith("V"):
                verb_lines.append(line)
            else:
                other_lines.append(line)
    # print info
    print(f"splitting {file_name} by POS: found {len(verb_lines)} verbs, {len(noun_lines)} common nouns, and {len(other_lines)} others.")
    print(f"Number of tokenization mismathces: {tok_mismatch_counter}. \n")
    # write to files
    base_name, suffix = file_name.rsplit('.', 1)
    with open('.'.join([base_name, 'verbs', suffix]), "w") as verb_fout:
        verb_fout.writelines(verb_lines)
    with open('.'.join([base_name, 'nouns', suffix]), "w") as noun_fout:
        noun_fout.writelines(noun_lines)
    if other_lines:
        with open('.'.join([base_name, 'others', suffix]), "w") as other_fout:
            other_fout.writelines(other_lines)


if __name__=="__main__":
    input_file_names = sys.argv[1:]
    for input_file_name in input_file_names:
        split_by_pos(input_file_name)
