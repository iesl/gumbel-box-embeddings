{
    "dataset_reader": {
        "type": "openke-dataset",
        "all_datadir": "data",
        "dataset_name": "HYPER_TR_0",
        "mode": "train"
    },
    "iterator": {
        "type": "basic",
        "batch_size": 1024,
        "cache_instances": true
    },
    "model": {
        "type": "BCE-bessel-approx-classification-model",
        "box_type": "DeltaBoxTensor",
        "debug": false,
        "embedding_dim": 5,
        "gumbel_beta": 0.06696274156519905,
        "init_interval_center": 0.3286913699760018,
        "init_interval_delta": 0.431347640187981,
        "num_entities": 82114,
        "num_relations": 1,
        "number_of_negative_samples": 59,
        "regularization_weight": 0,
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
            "lr": 0.0010076131785737292
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
