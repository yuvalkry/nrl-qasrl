#!/bin/bash
CONFIG_FILE=./configs/train_qanom_bert-what-bio-f1.jsonnet
SAVE_DIRECTORY=./train_out/qanom-what-bio-f1
/bin/rm -fr $SAVE_DIRECTORY
python -m allennlp train $CONFIG_FILE --include-package nrl -s $SAVE_DIRECTORY
