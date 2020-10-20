import sys
from allennlp.commands.predict import Predict, _get_predictor, _PredictManager
from allennlp.commands.subcommand import Subcommand
from allennlp.data.iterators import DataIterator
import argparse
from pathlib import Path
from allennlp.common import Params
from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
from allennlp.nn import util as nn_util
import json
import logging
logger = logging.getLogger(__name__)


class PredictForErrorAnalysis(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction
                      ) -> argparse.ArgumentParser:
        description = """Run predictions on the val dataset"""
        subparser = parser.add_parser(
            name,
            description=description,
            help="Use a trained model to make predictions on validation dataset"
        )
        subparser.add_argument(
            "serialization_dir",
            type=str,
            help="The directory which contains the config.json and the model archive"
        )
        subparser.add_argument(
            "--archive_file",
            type=str,
            help="the optional archived model to make predictions with. If specified overrides the one in serialization_dir"
        )
        subparser.add_argument(
            "--output-file", type=str, help="path to output file")
        subparser.add_argument(
            "--weights-file",
            type=str,
            help="a path that overrides which weights file to use")
        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device",
            type=int,
            default=-1,
            help="id of GPU to use (if any)")
        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help="a JSON structure used to override the experiment configuration",
        )
        subparser.add_argument(
            "--predictor",
            type=str,
            help="optionally specify a specific predictor to use")
        subparser.add_argument(
            "--print-to-console",
            action="store_true",
            help="Print to console")

        subparser.set_defaults(func=_predict_for_error_analysis)

        return subparser


def _read_params(args: argparse.Namespace) -> Params:
    return Params.from_file(
        str(Path(args.serialization_dir) / 'config.json'), args.overrides)


def _create_iterator(params: Params) -> DataIterator:
    val_iterr_params = params.pop("validation_iterator", None)

    if val_iterr_params is None:
        raise ValueError("Config file should have validation_iterator")

    return DataIterator.from_params(val_iterr_params)


def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(
        args.archive_file,
        weights_file=args.weights_file,
        cuda_device=args.cuda_device,
        overrides=args.overrides,
    )

    return Predictor.from_archive(
        archive, args.predictor, dataset_reader_to_load='validation')


class _PredictManagerForErrorAnalysis(_PredictManager):
    def __init__(self,
                 predictor: Predictor,
                 iterator: DataIterator,
                 output_file: str = None,
                 print_to_console: bool = True, device:int=-1):
        super().__init__(
            predictor,
            input_file="dummy_input_file",
            output_file=output_file,
            batch_size=0,
            print_to_console=print_to_console,
            has_dataset_reader=True)
        self.iterator=iterator
        self.device = device

    def call_forward(self, inputs):
        inputs = nn_util.move_to_device(inputs, self.device)
        return self._predictor._model.forward(**inputs)

    def _maybe_print_to_console_and_file(self, index, result, input_tensors):
        input_ = {'h': input_tensors['hr_e'][0][0].item(), 't': input_tensors['hr_t'].item(), 'r': input_tensors['hr_r'].item()}
        line = json.dumps({**(input_), **result}) + '\n'
        if self._print_to_console: 
            print(line)
        if self._output_file is not None:
            self._output_file.write(line)

    def run(self) -> None:
        has_reader = self._dataset_reader is not None
        index = 0
        generator = self.iterator(
            self._dataset_reader.read("dummy_file"), num_epochs=1, shuffle=False)

        if has_reader:
            logger.info("Starting predictions...")
            for batch in generator:
                result = self.call_forward(batch)
                self._maybe_print_to_console_and_file(index, result, batch)
                index = index + 1
        logger.info("Prediction finished ...")
        if self._output_file is not None:
            self._output_file.close()


def _predict_for_error_analysis(args: argparse.Namespace) -> None:
    # setup loading of archive

    if args.archive_file is None:
        args.archive_file = str(Path(args.serialization_dir) / 'model.tar.gz')

    params = _read_params(args)
    iterr = _create_iterator(params)
    predictor = _get_predictor(args)
    manager = _PredictManagerForErrorAnalysis(
        predictor,
        iterr,
        output_file=args.output_file,
        print_to_console=args.print_to_console, device=args.cuda_device)
    manager.run()
