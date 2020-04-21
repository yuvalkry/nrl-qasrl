local QASRL_DATA_DIR = "data/qasrl-v2";
local QANOM_DATA_DIR = "data/qanom_annotations";
local bert_model = "bert-base-uncased";

{
  "vocabulary": {
    "pretrained_files": {"tokens": "data/glove/glove.6B.100d.txt.gz"},
    "only_include_pretrained_words": true
  },
  "dataset_reader": {
      "type": "qanom",
      "token_indexers": {
          "tokens": {
            "type": "single_id",
            "lowercase_tokens": true
          },
          "bert": {
                  "type": "pretrained_transformer",
                  "model_name": bert_model,
          },
      },
  },
  "train_data_path": QANOM_DATA_DIR + "/train_set/final/annot.train.csv",
  "validation_data_path": QANOM_DATA_DIR + "/gold_set/final/annot.final.dev.csv",
  "test_data_path": QANOM_DATA_DIR + "/gold_set/final/annot.final.test.csv",
  "model": {
    "type": "span_detector",
    "text_field_embedder": {
      "token_embedders": {
          "bert": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
          }
      }


    },
    "stacked_encoder": {
      "type": "alternating_lstm",
      "use_highway": true,
      "input_size": 1224,
      "hidden_size": 300,
      "num_layers": 8,
      "recurrent_dropout_probability": 0.1
    },
    "predicate_feature_dim":100,
    "thresholds": [0.2, 0.5, 0.8, 0.95],
    "iou_threshold": 0.3,
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "batch_size" : 80
  },
  "trainer": {
    "num_epochs": 200,
    "grad_norm": 1.0,
    "patience": 30,
    "validation_metric": "+fscore-at-0.95",
    "cuda_device": 1,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  }
}
