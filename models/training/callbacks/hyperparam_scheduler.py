from typing import Iterable, List, TYPE_CHECKING, Optional, Callable
from allennlp.training.callbacks.validate import Validate
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events
from allennlp.data.instance import Instance
from allennlp.data.iterators import DataIterator
from models.training.schedulers import Scheduler
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
import logging

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer
logger = logging.getLogger(__name__)


@Callback.register("parameter-scheduler")
class ParameterScheduler(Callback):
    def __init__(self,
                 field: str,
                 scheduler: Scheduler,
                 metric: Optional[str] = None) -> None:
        super().__init__()
        self.field = field
        self.scheduler = scheduler
        self.metric = metric

    def get_training_state(self) -> dict:
        """
        We need to persist the learning_rate_scheduler state as training state.
        """

        return {
            "scheduler": self.scheduler.state_dict(),
        }

    def restore_training_state(self, training_state: dict) -> None:
        state_dict = training_state.pop("scheduler", None)

        if state_dict is not None:
            self.scheduler.load_state_dict(state_dict)

    @handle_event(Events.TRAINING_START)
    def training_start(self, trainer: "CallbackTrainer") -> None:
        pass

    @handle_event(Events.EPOCH_END, priority=200)
    def update_params(self, trainer: "CallbackTrainer") -> None:
        """Validation has priority around 100. So this will be
        called after epoch end validation is done"""

        old_value = getattr(trainer.model, self.field)

        if self.metric is not None:
            metric = trainer.val_metrics[self.metric]
        else:
            metric = None
        next_value = self.scheduler.step(
            metric=metric, epoch=trainer.epoch_number)

        if next_value is None:
            next_value = old_value
        setattr(trainer.model, self.field, next_value)
        logger.info("Updated {} from {} to {}".format(self.field, old_value,
                                                      next_value))

    @classmethod
    def from_params(cls, params: Params) -> "ParameterScheduler":
        scheduler_params = params.pop("scheduler", None)

        if scheduler_params is None:
            raise ConfigurationError("scheduler needed")
        scheduler = Scheduler.from_params(scheduler_params)
        field = params.pop("field", None)

        if field is None:
            raise ConfigurationError("field needed")
        metric = params.pop("metric", None)

        return cls(field, scheduler, metric)
