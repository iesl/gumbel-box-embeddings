{
    "dataset_reader": {
        "type": "openke-dataset",
        "all_datadir": "data",
        "dataset_name": "HYPER_TR_0",
        "mode": "train"
    },
    "iterator": {
        "type": "basic",
        "batch_size": 8096,
        "cache_instances": true
    },
    "model": {
        "type": "BCE-bessel-approx-classification-model",
        "box_type": "DeltaBoxTensor",
        "debug": false,
        "embedding_dim": 5,
        "gumbel_beta": 0.020666635750361234,
        "init_interval_center": 0.6297446386176542,
        "init_interval_delta": 1.2803425602797711,
        "num_entities": 82114,
        "num_relations": 1,
        "number_of_negative_samples": 78,
        "regularization_weight": "2.048861618635022e-05",
        "single_box": "true",
        "softbox_temp": 7
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
                "patience": 200,
                "type": "track_metrics",
                "validation_metric": "+fscore"
            },
        ],
        "cuda_device": -1,
        "num_epochs": 5000,
        "optimizer": {
            "type": "adam",
            "lr": 0.0054919509155002
        },
        "shuffle": true
    },
    "datasets_for_vocab_creation": [],
    "validation_dataset_reader": {
        "type": "classification-validation-dataset",
        "all_datadir": "data",
        "dataset_name": "HYPER_TR_0",
        "validation_file": "classification_samples_valid2id.txt"
    },
    "validation_iterator": {
        "type": "basic",
        "batch_size": 5000,
        "cache_instances": true
    }
}

