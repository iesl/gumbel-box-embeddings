from typing import Any, Set, Optional, Callable, Dict
import logging
import os

from tensorboardX import SummaryWriter
import torch

from allennlp.common.from_params import FromParams
from allennlp.models.model import Model
from models.box.base import TensorBoardLoggable
logger = logging.getLogger(__name__)


class TensorboardWriter(FromParams):
    def __init__(self,
                 get_batch_num_total: Callable[[], int],
                 serialization_dir: str = None,
                 log_freq: int = 100,
                 debug: bool = False):
        self._get_batch_num_total = get_batch_num_total

        if serialization_dir is not None:
            self._train_log = SummaryWriter(
                os.path.join(serialization_dir, "log", "train"))
            self._validation_log = SummaryWriter(
                os.path.join(serialization_dir, "log", "validation"))
        else:
            self._train_log = self._validation_log = None
        self.debug = debug

        self.log_freq = log_freq

    @staticmethod
    def _item(value: Any):
        if hasattr(value, "item"):
            val = value.item()
        else:
            val = value

        return val

    def should_log_this_batch(self) -> bool:
        if self.debug:
            return self._get_batch_num_total() % self.log_freq == 0
        else:
            return False

    def add_train_scalar(self, name: str, value: float,
                         timestep: int = None) -> None:
        timestep = timestep or self._get_batch_num_total()
        # get the scalar

        if self._train_log is not None:
            self._train_log.add_scalar(name, self._item(value), timestep)

    def add_train_histogram(self, name: str, values: torch.Tensor) -> None:
        if self._train_log is not None:
            if isinstance(values, torch.Tensor):
                values_to_write = values.cpu().data.numpy().flatten()
                self._train_log.add_histogram(name, values_to_write,
                                              self._get_batch_num_total())

    def add_validation_scalar(self,
                              name: str,
                              value: float,
                              timestep: int = None) -> None:
        timestep = timestep or self._get_batch_num_total()

        if self._validation_log is not None:
            self._validation_log.add_scalar(name, self._item(value), timestep)

    def log_histograms(self, model: TensorBoardLoggable) -> None:

        for name, _tensor in model.get_histograms_to_log().items():
            self.add_train_histogram("model_histogram/" + name, _tensor)

    def log_scalars(self, model: TensorBoardLoggable) -> None:

        for name, _tensor in model.get_scalars_to_log().items():
            self.add_train_histogram("model_scalars/" + name, _tensor)

    def close(self) -> None:
        """
        Calls the ``close`` method of the ``SummaryWriter`` s which makes sure that pending
        scalars are flushed to disk and the tensorboard event files are closed properly.
        """

        if self._train_log is not None:
            self._train_log.close()

        if self._validation_log is not None:
            self._validation_log.close()
