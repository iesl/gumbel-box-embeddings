{
    "dataset_reader": {
        "type": "openke-classification-dataset-negative-sampling",
        "all_datadir": "data",
        "dataset_name": "large_tree_rank_based",
        "mode": "train",
        "number_negative_samples": 40
    },
    "iterator": {
        "type": "basic",
        "batch_size": 2048,
        "cache_instances": true
    },
    "model": {
        "type": "BCE-bessel-approx-model",
        "box_type": "DeltaBoxTensor",
        "debug": false,
        "embedding_dim": 2,
        "gumbel_beta": 0.09196127139742319,
        "init_interval_center": 0.41830647062119236,
        "init_interval_delta": 1.2624614378801073,
        "neg_samples_in_dataset_reader": 40,
        "num_entities": 3000,
        "num_relations": 1,
        "number_of_negative_samples": 0,
        "regularization_weight": 0,
        "single_box": "true",
        "softbox_temp": 4.225604072207862
    },
    "train_data_path": "dummpy_path",
    "validation_data_path": "dummy_path",
    "trainer": {
        "type": "callback",
        "callbacks": [
            {
                "debug": false,
                "log_freq": 2,
                "type": "debug-validate"
            },
            {
                "checkpointer": {
                    "num_serialized_models_to_keep": 1
                },
                "type": "checkpoint"
            },
            {
                "patience": 130,
                "type": "track_metrics",
                "validation_metric": "+mrr"
            },
        ],
        "cuda_device": -1,
        "num_epochs": 500,
        "optimizer": {
            "type": "adam",
            "lr": 0.015739433585983263
        },
        "shuffle": true
    },
    "datasets_for_vocab_creation": [],
    "validation_dataset_reader": {
        "type": "openke-rank-validation-dataset",
        "all_datadir": "data",
        "dataset_name": "large_tree_rank_based",
        "validation_file": "classification_samples_valid2id.txt"
    },
    "validation_iterator": {
        "type": "single-sample-rank-validation-iterator",
        "batch_size": 1
    }
}
