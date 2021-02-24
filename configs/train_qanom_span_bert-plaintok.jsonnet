// local QANOM_DATA_DIR = "data/qanom_annotations";
local QANOM_DATA_DIR = "/home/nlp/yuvalk/qanom/QANom-anlp-1.1.0/dataset/small" ;
local bert_model = "bert-base-uncased";

{
  "dataset_reader": {
      "type": "qanom",
      "token_indexers": {
          "bert": {
                  "type": "pretrained_transformer",
                  "model_name": bert_model,
          },
      },
  },

  "train_data_path": QANOM_DATA_DIR + "/train.csv",
  "validation_data_path": QANOM_DATA_DIR + "/dev.csv",
  "test_data_path": QANOM_DATA_DIR + "/test.csv",
// for debug (small data samples):
//  "train_data_path": "data/qanom_train_sample.csv",
//  "validation_data_path": QANOM_DATA_DIR + "/gold_set/final/annot.final.wikinews.dev.5.csv",
//  "test_data_path": QANOM_DATA_DIR + "/gold_set/final/annot.final.wikinews.test.1.csv",

// for real:
//  "train_data_path": QANOM_DATA_DIR + "/train_set/final/annot.train.csv",
//  "validation_data_path": QANOM_DATA_DIR + "/gold_set/final/annot.final.dev.csv",
//  "test_data_path": QANOM_DATA_DIR + "/gold_set/final/annot.final.test.csv",

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
//    "stacked_encoder": {
//      "type": "alternating_lstm",
//      "use_highway": true,
//      "input_size": 868,
//      "hidden_size": 400,
//      "num_layers": 8,
//      "recurrent_dropout_probability": 0.1
//    },
    "stacked_encoder": {
          "type": "multi_head_self_attention",
          "input_dim": 868,
          "values_dim": 600,
          "attention_dim": 600,
          "num_heads": 12,
          "attention_dropout_prob": 0.1
    },
    "regularizer":  {"regexes": [
        [
            ".*scalar_parameters.*",
            {
                "type": "l2",
                "alpha": 0.001
            }
        ]
    ] },
    "predicate_feature_dim":100,
    "thresholds": [0.05, 0.2, 0.5, 0.9],
    "iou_threshold": 0.3,
  },
  "data_loader": {
    batch_sampler: {
      "type": "bucket",
      // "sorting_keys": [["text", "num_tokens"]],
      "batch_size" : 80
    }
  },
  // "distributed": {
  // "cuda_devices": [1,2],
  // },
  "trainer": {
    "num_epochs": 500,
    "grad_clipping": 1.0,
    "patience": 500,
    "validation_metric": "+fscore-at-0.5",
    "cuda_device": 3,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  }
}
