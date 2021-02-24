#!/bin/bash
CONFIG_FILE=./configs/train_qanom_span_bert.jsonnet
SAVE_DIRECTORY=./train_out/bert-span
/bin/rm -rf $SAVE_DIRECTORY
python -m allennlp train $CONFIG_FILE --include-package nrl -s $SAVE_DIRECTORY
