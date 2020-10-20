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
        "type": "BCE-sampled-classification-box-model",
        "box_type": "DeltaBoxTensor",
        "debug": false,
        "embedding_dim": 5,
        "init_interval_center": 0.5211166554488466,
        "init_interval_delta": 0.7437919327491256,
        "n_samples": 20,
        "num_entities": 82114,
        "num_relations": 1,
        "number_of_negative_samples": 18,
        "regularization_weight": 0,
        "sigma_init": 1,
        "single_box": "true",
        "softbox_temp": 1.0797603544176941
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
                "patience": 100,
                "type": "track_metrics",
                "validation_metric": "+fscore"
            },
        ],
        "cuda_device": -1,
        "num_epochs": 5000,
        "optimizer": {
            "type": "adam",
            "lr": 0.015533067312841504
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
