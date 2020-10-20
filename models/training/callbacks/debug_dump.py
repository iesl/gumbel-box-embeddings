from typing import Iterable, List, TYPE_CHECKING, Optional, Callable
from allennlp.training.callbacks.validate import Validate
from allennlp.training.callbacks import Callback, handle_event, Checkpoint
from allennlp.training.callbacks.events import Events
from allennlp.data.instance import Instance
from allennlp.data.iterators import DataIterator
import logging

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer
logger = logging.getLogger(__name__)


@Callback.register("debug-dump-model")
class DebugValidate(Callback):
    """Same as allennlp.training.callbacks.Validate except
    some special extral calculations when debug flag is set"""

    def __init__(self,
                 debug: bool,
                 validation_data: Iterable[Instance],
                 validation_iterator: DataIterator,
                 log_freq: int = 100) -> None:  # type: ignore
        super().__init__(validation_data, validation_iterator)
        self.debug = debug
        self.log_freq = log_freq
        # this should be initialized at the begining of training!!
        self._get_batch_num_total: Optional[Callable[[], int]] = None

    @handle_event(Events.TRAINING_START)
    def training_start(self, trainer: "CallbackTrainer") -> None:
        # This is an ugly hack to get the tensorboard instance to know about the trainer, because
        # the callbacks are defined before the trainer.
        self._get_batch_num_total = lambda: trainer.batch_num_total

    @handle_event(Events.BATCH_END, priority=100)
    def validate_batch_end(self, trainer: "CallbackTrainer") -> None:

        if self._get_batch_num_total is None:
            raise RuntimeError(
                "self._get_batch_num_total should have been set by now")

        if self.debug:
            if self._get_batch_num_total() % self.log_freq == 0:
                current_mode_is_training = trainer.model.training

                if current_mode_is_training:
                    trainer.model.eval()

                if self.debug:
                    self.validate(trainer)

                if current_mode_is_training:
                    trainer.model.train()

    @handle_event(Events.ERROR, priority=100)
    def validate_after_nan_loss(self, trainer: "CallbackTrainer") -> None:
        """ We need to try and run validation, even after nan loss"""

        if isinstance(trainer.exception, ValueError):
            if str(trainer.exception).strip() == "nan loss encountered":
                logger.info("Nan loss encountered. Trying to run validation ")
                trainer.model.eval()
                self.validate(trainer)
