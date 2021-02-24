#!/bin/bash
CONFIG_FILE=./configs/train_qanom_bert.jsonnet
SAVE_DIRECTORY=./train_out/qanom-all
/bin/rm -fr $SAVE_DIRECTORY
python -m allennlp train $CONFIG_FILE --include-package nrl -s $SAVE_DIRECTORY
