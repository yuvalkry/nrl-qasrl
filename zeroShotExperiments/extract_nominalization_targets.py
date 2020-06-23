"""
Given a file of SRL-like annotation in the format "sentence string ||| i:i-rel j-k-RoleName ...", e.g. ACE or
PropBank or Ontonotes data, extract from the file the instances for which the trigger is a nominalization.
We define here nominalization using the QANom candidate extraction heuristic, based on lexical resources.
"""
from typing import List, Tuple, Dict
Span = Tuple[int, int]


import spacy, sys, pandas as pd
sys.path.append("/home/nlp/kleinay/Synced/nrl-parser")
spacy_nlp = spacy.load("en_core_web_sm")
from qanom.candidate_extraction import prepare_qanom_prompts as extract
from qanom.utils import df_to_dict
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

def export_nominalization_sebset(file_name):
    """ ... based on qanom's candidate extraction heuristics """
    count = 0
    nom_lines = []
    with open(file_name) as fin:
        for line in fin.readlines():
            count += 1
            sent, args_str = line.strip().split(" ||| ")
            # take out 'rel' index
            str_list = line.strip().split()
            separator_index = str_list.index("|||")
            og_sentence = str_list[:separator_index]
            args = str_list[separator_index + 1:]
            # Get index of predicate. Predicate is always first argument.
            predicate_loc = args[0][:args[0].find(':')]
            sub_idx = predicate_loc.find('_')
            if sub_idx < 0:
                og_nom_loc = [int(predicate_loc)]
            else:
                og_nom_loc = (int(predicate_loc[:sub_idx]), int(predicate_loc[sub_idx + 1:]))
            predicate_idx = og_nom_loc[0]    # assuming single-word predicates
            noun = og_sentence[predicate_idx]
            pos = spacy_nlp(sent)[predicate_idx].pos_

            # is this noun a candidate for being a deverbal nominalization?
            if pos=="NOUN" and extract.is_candidate_noun(noun):
                nom_lines.append(line)
    # print info
    print(f"splitting {file_name} by nominalization candidates: found {len(nom_lines)} nominalizations out of {count} instances.")
    # write to files
    base_name, suffix = file_name.rsplit('.', 1)
    with open('.'.join([base_name, 'noms', suffix]), "w") as nom_fout:
        nom_fout.writelines(nom_lines)

nomTypeDf = pd.read_csv("zeroShotExperiments/nombank_as_spans/nom_types.tsv", sep='\t')
surface2nomType = df_to_dict(nomTypeDf, 'surface', 'nom_type')
nomTypesSet = set(surface2nomType.values())

def export_verbal_nomtype_sebset(file_name):
    """ for NomBank data - based on NomLex NOT-TYPE of the lexical entry of the predicate. """
    count = 0
    nom_lines = []
    with open(file_name) as fin:
        for line in fin.readlines():
            count += 1
            sent, args_str = line.strip().split(" ||| ")
            # take out 'rel' index
            str_list = line.strip().split()
            separator_index = str_list.index("|||")
            og_sentence = str_list[:separator_index]
            args = str_list[separator_index + 1:]
            # Get index of predicate. Predicate is always first argument.
            predicate_loc = args[0][:args[0].find(':')]
            sub_idx = predicate_loc.find('_')
            if sub_idx < 0:
                og_nom_loc = [int(predicate_loc)]
            else:
                og_nom_loc = (int(predicate_loc[:sub_idx]), int(predicate_loc[sub_idx + 1:]))
            predicate_idx = og_nom_loc[0]    # assuming single-word predicates
            noun = og_sentence[predicate_idx]

            # exclusion set - non-verbal nom-types
            exclude = {'NOM-REL', 'ADJ-NOM', 'CRISSCROSS'}
            # find noun's nom-type in nomlex dictionary
            nom_type = surface2nomType.get(noun, None)

            if nom_type and nom_type not in exclude:
                nom_lines.append(line)
    # print info
    print(f"splitting {file_name} by nom-type: found {len(nom_lines)} verbal nominalizations out of {count} instances.")
    # write to files
    base_name, suffix = file_name.rsplit('.', 1)
    with open('.'.join([base_name, 'verbal_nom_types', suffix]), "w") as nom_fout:
        nom_fout.writelines(nom_lines)


if __name__=="__main__":
    input_file_name = sys.argv[-1]
    export_verbal_nomtype_sebset(input_file_name)
