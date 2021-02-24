#!/bin/bash
CONFIG_FILE=./configs/train_qanom_quesgen_elmo.jsonnet
SAVE_DIRECTORY=./train_out/elmo-quesgen
/bin/rm -r $SAVE_DIRECTORY
python -m allennlp train $CONFIG_FILE --include-package nrl -s $SAVE_DIRECTORY
