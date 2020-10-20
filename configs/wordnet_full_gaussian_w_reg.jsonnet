{
    "dataset_reader": {
        "type": "openke-dataset",
        "all_datadir": "data",
        "dataset_name": "HYPER_TR_0",
        "mode": "train"
    },
    "iterator": {
        "type": "basic",
        "batch_size": 16192,
        "cache_instances": true
    },
    "model": {
        "type": "BCE-sampled-classification-box-model",
        "box_type": "DeltaBoxTensor",
        "debug": false,
        "embedding_dim": 5,
        "init_interval_center": 0.21402520443207218,
        "init_interval_delta": 0.8931608780534448,
        "n_samples": 20,
        "num_entities": 82114,
        "num_relations": 1,
        "number_of_negative_samples": 67,
        "regularization_weight": "6.269192110350666e-05",
        "sigma_init": 2,
        "single_box": "true",
        "softbox_temp": 1.9775432129970145
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
            "lr": 0.14381115041068177
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
