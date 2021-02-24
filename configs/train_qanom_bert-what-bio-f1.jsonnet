local QANOM_DATA_DIR = "/home/nlp/yuvalk/qanom/QANom-anlp-1.1.0/dataset" ;
local bert_model = "bert-base-uncased";

{
  "dataset_reader": {
     "type": "qanom",
     "wh_words": ["what"],
     "split_multiple_spans": false,
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

  "model": {
    "type": "qanom_parser",
    // how to combine span with question generation:
    // parallel/sequential
    "architecture": "parallel",
    "span_representation": "bio",
    "combined_metric": {
	    // "f1-measure-overall"
        "span-spans": "fscore-at-0.5",
        // "span-bio": "bio_acc",
        "span-bio": "f1-measure-overall",
        "quesgen": "question-role-accuracy"
    },
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
    "stacked_encoder_ques": {
      "type": "alternating_lstm",
      "use_highway": true,
      "input_size": 868,
      "hidden_size": 300,
      "num_layers": 4,
      "recurrent_dropout_probability": 0.1
    },
    "stacked_encoder_span": {
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
    "thresholds": [0.02, 0.05, 0.07, 0.2, 0.5, 0.9],
    "iou_threshold": 0.3
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      // "sorting_keys": [["text", "num_tokens"]],
      "batch_size" : 40
    }
  },
  // "distributed": {
  // "cuda_devices": [1,2],
  // },
  "trainer": {
    "num_epochs": 100,
    "grad_clipping": 1.0,
    "patience": 50,
    "validation_metric": "+combined",
    "cuda_device": 1,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  }
}
