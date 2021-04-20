from typing import Iterable, List, TYPE_CHECKING, Optional, Callable
import torch
import math
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of
from allennlp.training import util as training_util
from allennlp.training.moving_average import MovingAverage
from allennlp.training.callbacks.validate import Validate
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events
from allennlp.data.instance import Instance
from allennlp.data.iterators import DataIterator
import logging

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer
logger = logging.getLogger(__name__)


@Callback.register("validate-interval")
class validate_interval(Validate):
    """ This class same as allennlp.training.callbacks.Validate except
    this class will have a option to put a validate frequency"""
    def __init__(self,
                 validation_data: Iterable[Instance],
                 validation_iterator: DataIterator,
                 log_freq: int = 100) -> None:  # type: ignore
        super().__init__(validation_data, validation_iterator)
        self.log_freq = log_freq

    @handle_event(Events.VALIDATE)
    def validate(self, trainer: 'CallbackTrainer'):
        # If the trainer has MovingAverage objects, use their weights for validation.
        if trainer.epoch_number % self.log_freq == 0:
            for moving_average in self.moving_averages:
                moving_average.assign_average_value()

            with torch.no_grad():
                # We have a validation set, so compute all the metrics on it.
                logger.info("Validating")

                trainer.model.eval()

                num_gpus = len(trainer._cuda_devices)  # pylint: disable=protected-access

                raw_val_generator = self.iterator(self.instances,
                                                  num_epochs=1,
                                                  shuffle=False)
                val_generator = lazy_groups_of(raw_val_generator, num_gpus)
                num_validation_batches = math.ceil(
                        self.iterator.get_num_batches(self.instances) / num_gpus)
                val_generator_tqdm = Tqdm.tqdm(val_generator,
                                               total=num_validation_batches)

                batches_this_epoch = 0
                val_loss = 0
                for batch_group in val_generator_tqdm:

                    loss = trainer.batch_loss(batch_group, for_training=False)
                    if loss is not None:
                        # You shouldn't necessarily have to compute a loss for validation, so we allow for
                        # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                        # currently only used as the divisor for the loss function, so we can safely only
                        # count those batches for which we actually have a loss.  If this variable ever
                        # gets used for something else, we might need to change things around a bit.
                        batches_this_epoch += 1
                        val_loss += loss.detach().cpu().numpy()

                    # Update the description with the latest metrics
                    val_metrics = training_util.get_metrics(trainer.model, val_loss, batches_this_epoch)
                    description = training_util.description_from_metrics(val_metrics)
                    val_generator_tqdm.set_description(description, refresh=False)

                trainer.val_metrics = training_util.get_metrics(trainer.model,
                                                                val_loss,
                                                                batches_this_epoch,
                                                                reset=True)

            # If the trainer has a moving average, restore
            for moving_average in self.moving_averages:
                moving_average.restore()


@Callback.register("debug-validate")
class DebugValidate(Validate):
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
