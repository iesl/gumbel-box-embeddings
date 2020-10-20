{
    "dataset_reader": {
        "type": "openke-classification-dataset-negative-sampling",
        "all_datadir": "data",
        "dataset_name": "toy_tree_balanced_rank_based",
        "mode": "train",
        "number_negative_samples": 20
    },
    "iterator": {
        "type": "basic",
        "batch_size": 40,
        "cache_instances": true
    },
    "model": {
        "type": "BCE-sampled-box-model",
        "box_type": "DeltaBoxTensor",
        "debug": false,
        "embedding_dim": 1,
        "init_interval_center": 0.5726396310315645,
        "init_interval_delta": 0.7817885980117143,
        "n_samples": 25,
        "neg_samples_in_dataset_reader": 20,
        "num_entities": 40,
        "num_relations": 1,
        "number_of_negative_samples": 0,
        "regularization_weight": 0,
        "sigma_init": 1,
        "single_box": "true",
        "softbox_temp": 17.414078698770517
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
        ],
        "cuda_device": -1,
        "num_epochs": 500,
        "optimizer": {
            "type": "adam",
            "lr": 0.005559676276620351
        },
        "shuffle": true
    },
    "datasets_for_vocab_creation": [],
    "validation_dataset_reader": {
        "type": "openke-rank-validation-dataset",
        "all_datadir": "data",
        "dataset_name": "toy_tree_balanced_rank_based",
        "validation_file": "classification_samples_valid2id.txt"
    },
    "validation_iterator": {
        "type": "single-sample-rank-validation-iterator",
        "batch_size": 1
    }
}
