{
    "dataset_reader": {
        "type":
        "openke-classification-dataset-negative-sampling",
        "dataset_name":
        "toy_tree_balanced_rank_based",
        "all_datadir":
        "data",
        "mode":
        "train",
        "number_negative_samples": 30
    },
    "validation_dataset_reader": {
        "type":
            "openke-rank-validation-dataset",
        "dataset_name":
            "toy_tree_balanced_rank_based",
         "all_datadir":
            "data",
  },
  "train_data_path": "dummpy_path",
  "validation_data_path": "dummy_path",
  "datasets_for_vocab_creation": [],
  "iterator": {
    "type": "basic",
    "batch_size": 30,
    "cache_instances": true,
  },
  "validation_iterator": {
    "type": "single-sample-rank-validation-iterator",
    "batch_size": 1,
  },
  "model": {
    "type": "BCE-bessel-approx-model",
    "num_entities": 40,
    "num_relations": 1,
    "embedding_dim": 2,
    "number_of_negative_samples": 0,
    "box_type": "DeltaBoxTensor",
    "debug": false,
    "regularization_weight": 0.00,
    "init_interval_center": 0.5,
    "init_interval_delta": 1,
    "softbox_temp": 7.0,
    "gumbel_beta": 0.01,
  },
  trainer: {
    "type": "callback",
    local common_debug = false,
    local common_freq = 2,
    "callbacks": [
      {
        "type": "debug-validate",
        "debug": common_debug,
        "log_freq": common_freq,
      },
      {
        "checkpointer": {
            "num_serialized_models_to_keep": 1
         },
         "type": "checkpoint"
       },
      {
        "type": "track_metrics",
        "patience": 100,
        "validation_metric": "+mrr",
      },
    ],
    "optimizer": {
      "type": "adam",
      "lr": 0.01,
    },
    "cuda_device": -1,
    "num_epochs": 100,
    "shuffle": true,
  },
}
