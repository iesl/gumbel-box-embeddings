{
    "dataset_reader": {
        "type": "openke-classification-dataset-negative-sampling",
        "all_datadir": "data",
        "dataset_name": "mammal",
        "mode": "train",
        "number_negative_samples": 90
    },
    "iterator": {
        "type": "basic",
        "batch_size": 512,
        "cache_instances": true
    },
    "model": {
        "type": "BCE-box-model",
        "box_type": "DeltaBoxTensor",
        "debug": false,
        "embedding_dim": 2,
        "init_interval_center": 0.4118299719735693,
        "init_interval_delta": 0.4398838231359177,
        "neg_samples_in_dataset_reader": 90,
        "num_entities": 1182,
        "num_relations": 1,
        "number_of_negative_samples": 0,
        "regularization_weight": 0,
        "single_box": "true",
        "softbox_temp": 9.783415602504801
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
                "validation_metric": "+mrr"
            },
        ],
        "cuda_device": -1,
        "num_epochs": 500,
        "optimizer": {
            "type": "adam",
            "lr": 0.017163162757165732
        },
        "shuffle": true
    },
    "datasets_for_vocab_creation": [],
    "validation_dataset_reader": {
        "type": "openke-rank-validation-dataset",
        "all_datadir": "data",
        "dataset_name": "mammal",
        "validation_file": "classification_samples_valid2id.txt"
    },
    "validation_iterator": {
        "type": "single-sample-rank-validation-iterator",
        "batch_size": 1
    }
}
