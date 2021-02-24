// local QANOM_DATA_DIR = "data/qanom_annotations";
local QANOM_DATA_DIR = "/home/nlp/yuvalk/qanom/QANom-anlp-1.1.0/dataset" ;
local bert_model = "bert-base-uncased";

{
  "dataset_reader": {
      "type": "qanom",
      "tokenizer": {
        "type": "pretrained_transformer",
        "model_name": bert_model,
        "add_special_tokens": false,
        "tokenizer_kwargs": {
            "additional_special_tokens": ["@V_B", "@V_E", "@ARG_B", "@ARG_E"]
        }
      },
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
  "evaluate_on_test": true,

    // for debug (small data samples):
//  "train_data_path": "data/qanom_train_sample.csv",
//  "validation_data_path": QANOM_DATA_DIR + "/gold_set/final/annot.final.wikinews.dev.5.csv",
//  "test_data_path": QANOM_DATA_DIR + "/gold_set/final/annot.final.wikinews.test.1.csv",

    // for real:
//  "train_data_path": QANOM_DATA_DIR + "/train_set/final/annot.train.csv",
//  "validation_data_path": QANOM_DATA_DIR + "/gold_set/final/annot.final.dev.csv",
//  "test_data_path": QANOM_DATA_DIR + "/gold_set/final/annot.final.test.csv",

  "model": {
    "type": "question_predictor",
    "text_field_embedder": {
        "token_embedders": {
          "bert": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
          }
      }
    },
    "question_generator": {
        "type": "sequence",
        "dim_slot_hidden":100,
        "dim_rnn_hidden": 200,
        "input_dim": 300,
        "rnn_layers": 4,
        "share_rnn_cell": false
    },
    "stacked_encoder": {
      "type": "alternating_lstm",
      "use_highway": true,
      "input_size": 868,
      "hidden_size": 300,
      "num_layers": 4,
      "recurrent_dropout_probability": 0.1
    },
    "predicate_feature_dim":100
  },
  "data_loader": {
    batch_sampler: {
      "type": "bucket",
      // "sorting_keys": [["text", "num_tokens"]],
      "batch_size" : 30
    },
  },
  "trainer": {
    "num_epochs": 200,
    "grad_norm": 1.0,
    "patience": 50,
    "validation_metric": "+question-role-accuracy",
    "cuda_device": 3,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  }
}
