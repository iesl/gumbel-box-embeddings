from typing import Iterable, List, TYPE_CHECKING, Optional, Callable
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events
import torch
import logging

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer
logger = logging.getLogger(__name__)


@Callback.register("gradient-debug")
class GradDebug(Callback):
    """ Sets `torch.autograd.set_detect_anomaly` to true
    at the start of training"""

    def __init__(self, stop_at: Optional[List[int]] = None):
        super().__init__()

        if stop_at is not None:
            self.stop_at = stop_at
        else:
            self.stop_at = []
        self._get_batch_num_total_this_epoch: Optional[[Callable[[], int]
                                                        ]] = None

    @handle_event(Events.TRAINING_START)
    def training_start(self, trainer: "CallbackTrainer") -> None:
        torch.autograd.set_detect_anomaly(True)
        self._get_batch_num_total_this_epoch = lambda: trainer.batches_this_epoch

    @handle_event(Events.BATCH_START, priority=200)
    def batch_start(self, trainer: "CallbackTrainer") -> None:
        batch_num = self._get_batch_num_total_this_epoch()

        if batch_num in self.stop_at:
            import pdb
            pdb.set_trace()  # noqa
