from typing import Dict, List, Iterator, Any, Union
from allennlp.common.params import Params, with_fallback
import os
from pathlib import Path
from allennlp.models import Model
from allennlp.data import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.common.util import import_submodules
from allennlp.nn import util as nn_util
import models.box.max_margin_models
import tqdm
import datasets
import torch
import logging
import re
import json
logger = logging.getLogger(__name__)

_DEFAULT_METRICS_FILE = 'metrics.json'


def load_modules(names: List[str]) -> None:
    for package_name in names:
        import_submodules(package_name)


def load_config(serialization_dir: str,
                config_path: str = None,
                overrides_dict: Dict = None) -> Params:

    if serialization_dir is None and (config_path is None):
        raise ValueError("Both cannot be None")

    if config_path is None:
        config_path = Path(serialization_dir) / "config.json"  # type: ignore
    config = Params.from_file(config_path)

    if overrides_dict is not None:
        config_dict = (config.as_dict())
        config = with_fallback(preferred=overrides_dict, fallback=config_dict)

        return Params(config)
    else:
        return config


def load_best_metrics(serialization_dir: str,
                      metrics_file: str = None) -> Dict:

    if metrics_file is None:
        metrics_file = _DEFAULT_METRICS_FILE
    with open(Path(serialization_dir) / metrics_file) as f:
        metrics = json.load(f)

    return metrics


_ts_pattern = re.compile(r"model_state_epoch_\d+.(\d+.\d+).th$")


def get_time_stamp(filename: str):
    m = _ts_pattern.match(filename)

    if not m:
        raise ValueError

    return m.group(1)


def sort_file_paths_according_to_time_stamp(paths: List[Path]) -> List[Path]:
    return sorted(paths, key=lambda p: get_time_stamp(p.name))


def get_intermidiate_models_filepaths(serialization_dir: str) -> List[Path]:
    serialization_dir = Path(serialization_dir)
    r = re.compile(
        # r"model_state_epoch_\d+.\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}.th$")
        r"model_state_epoch_\d+.\d+.\d+.th$")
    fs = [
        f for f in serialization_dir.glob('model_state_epoch_*.th')

        if r.match(f.name)
    ]

    return sort_file_paths_according_to_time_stamp(fs)


def load_model(serialization_dir: str,
               config: Params = None,
               overrides_dict: Dict = None,
               weights_file: str = None) -> Model:
    """Load the model from serialization dir

        Args:
            serialization_dir: `str`

            config: `str` Path to the config file. If not given, config in the serializatin dir is used
    """
    serialization_dir = serialization_dir

    if config is None:
        config = load_config(serialization_dir, overrides_dict=overrides_dict)
    model = Model.load(config, serialization_dir, weights_file=weights_file)

    return model


def load_intermidiate_models(serialization_dir: str,
                             config: Params = None,
                             overrides_dict: Dict = None) -> List[Model]:
    intermidiate_models = [
        load_model(serialization_dir, fpath.name, overrides_dict)

        for fpath in get_intermidiate_models_filepaths(serialization_dir)
    ]

    return intermidiate_models


def get_box_embedding_module(model: Model):
    return model.h


def load_dataset_reader(name: str = "validation_dataset_reader",
                        serialization_dir: str = None,
                        config: Params = None,
                        overrides_dict: Dict = None) -> DatasetReader:

    if config is None:
        if serialization_dir is None:
            raise ValueError
        config = load_config(serialization_dir, overrides_dict=overrides_dict)
    dataset_reader_params = config[name]

    return DatasetReader.from_params(dataset_reader_params)


def load_iterator(name: str = 'validation_iterator',
                  serialization_dir: str = None,
                  config: Params = None,
                  overrides_dict: Dict = None) -> DataIterator:

    if config is None:
        if serialization_dir is None:
            raise ValueError
        config = load_config(serialization_dir, overrides_dict=overrides_dict)
    val_iterr_params = config.pop(name, None)

    if val_iterr_params is None:
        raise ValueError("Config file should have {}".format(name))

    return DataIterator.from_params(val_iterr_params)


def create_onepass_generator(iterator: DataIterator,
                             dataset_reader: DatasetReader) -> Iterator:
    generator = iterator(
        dataset_reader.read("dummy_path"), num_epochs=1, shuffle=False)

    return generator


def load_outputs(file_name: str) -> List[Dict[str, Any]]:
    with open(file_name) as f:
        results = [json.loads(line) for line in f]

    return results


def predict_loop_or_load(
        model: Model,
        dataset_iterator: Iterator,
        device: str = 'cpu',
        output_file: Union[str, Path] = 'output.jsonl',
        force_repredict: bool = False) -> List[Dict[str, Any]]:
    """
    Checks if results are already present in the output file. If the file exists reads it and returns
    the contents. If it does not, runs the prediction loop and populates the file and returns results.
    """
    # check
    output_file: Path = Path(output_file)  # type: ignore

    if output_file.exists():
        if output_file.is_file():
            logger.info("{} file already exists...")

            if force_repredict:
                logger.info(
                    "force_repredict is True. Hence repredicting and overwritting"
                )
            else:
                logger.info("Reading results from the existing file ")

                return load_outputs(output_file)

    # Predict
    device = 'cpu'

    if device is 'cpu':
        device_instance = torch.device('cpu')
        device_int = -1
    else:
        device_instance = torch.device('cuda', 0)
        device_int = 0
    model = model.to(device=device_instance)
    model.eval()

    if hasattr(model, 'test'):
        model.test()
    results = []
    with open(output_file, 'w') as f:
        logger.info("Starting predictions ...")

        for i, input_batch in enumerate(tqdm.tqdm(dataset_iterator)):
            input_batch_on_device = nn_util.move_to_device(
                input_batch, device_int)
            result = model.forward(**input_batch_on_device)
            input_ = {
                'h': input_batch_on_device['tr_h'].item(),
                't': input_batch['hr_t'].item(),
                'r': input_batch['hr_r'].item()
            }
            result = {**(input_), **result}
            line = json.dumps(result) + '\n'
            results.append(result)
            print(i, ' : ', line)
            f.write(line)

    return results
