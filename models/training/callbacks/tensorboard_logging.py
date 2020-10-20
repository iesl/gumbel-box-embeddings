# These should probably all live in separate files
from models.training.tensorboard_writer import TensorboardWriter
from allennlp.training.callbacks.events import Events
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.common.params import Params
import logging
from typing import Set, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer

logger = logging.getLogger(__name__)


@Callback.register('tensorboard_logging')
class TensorboardLogger(Callback):
    def __init__(self, tensorboard: TensorboardWriter):
        self.tensorboard = tensorboard

    @handle_event(Events.TRAINING_START)
    def training_start(self, trainer: "CallbackTrainer") -> None:
        # This is an ugly hack to get the tensorboard instance to know about the trainer, because
        # the callbacks are defined before the trainer.
        self.tensorboard._get_batch_num_total = lambda: trainer.batch_num_total

    @handle_event(Events.BATCH_END)
    def batch_end_logging(self, trainer: "CallbackTrainer"):
        if self.tensorboard.should_log_this_batch():
            self.tensorboard.log_histograms(trainer.model)
            self.tensorboard.log_scalars(trainer.model)

    @handle_event(Events.VALIDATE)
    def epoch_end_logging(self, trainer: "CallbackTrainer"):
        """Log regardless of debug mode or not"""
        logger.info("Logging for tensorboard")
        self.tensorboard.log_histograms(trainer.model)
        self.tensorboard.log_scalars(trainer.model)

    @classmethod
    def from_params(  # type: ignore
            cls, serialization_dir: str,
            params: Params) -> "TensorboardLogger":
        tensorboard = TensorboardWriter.from_params(
            params=params,
            serialization_dir=serialization_dir,
            get_batch_num_total=lambda: None)

        return cls(tensorboard)
