{
  "dataset_reader": {
      "type": "qanom",
      "wh_words": ["what"],
      "split_multiple_spans": false,
      "tokenizer": {
        "type": "pretrained_transformer",
        "model_name": "bert-base-uncased",
        "add_special_tokens": false,
        "tokenizer_kwargs": {
            "additional_special_tokens": ["@V_B", "@V_E", "@ARG_B", "@ARG_E"]
        }
      },
      "token_indexers": {
          "bert": {
                  "type": "pretrained_transformer",
                  "model_name": "bert-base-uncased"
          }
      }
  },

  "train_data_path": "C:\\Users\\yuval\\Documents\\qanom\\dataset\\small\\train.csv",
  "validation_data_path": "C:\\Users\\yuval\\Documents\\qanom\\dataset\\small\\dev.csv",
  "test_data_path": "C:\\Users\\yuval\\Documents\\qanom\\dataset\\small\\test.csv",
  "evaluate_on_test": true,
  "model": {
    "type": "qanom_parser",
    "architecture": "parallel",
    "representation": "spans",
    "text_field_embedder": {
        "token_embedders": {
          "bert": {
             "type": "pretrained_transformer",
             "model_name": "bert-base-uncased"
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
    "thresholds": [0.05, 0.2, 0.5, 0.9],
    "iou_threshold": 0.3
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 10
    }
  },
  "trainer": {
    "num_epochs": 200,
    "grad_clipping": 1.0,
    "validation_metric": "+fscore-at-0.5",
    "cuda_device": -1,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  }
}
