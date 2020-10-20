from typing import Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch
import math
import logging
logger = logging.getLogger(__name__)


class Scheduler(Registrable):
    """
    A scheduler can be used to update any field in model in the trainer object,
    not just the learning rate.

    """

    def __init__(self) -> None:
        self.current_step = 0
        self._current_value: Optional[Any] = None

    @property
    def current_value(self) -> Any:
        return self._current_value

    @current_value.setter
    def current_value(self, v: Any) -> None:
        self._current_value = v

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the scheduler as a ``dict``.
        """

        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the schedulers state.

        Parameters
        ----------
        state_dict : ``Dict[str, Any]``
            Scheduler state. Should be an object returned from a call to ``state_dict``.
        """
        self.__dict__.update(state_dict)

    def step(self, metric: float = None, epoch: int = None) -> Any:
        """ Should update the current value and return it"""
        self.current_step += 1

        return


@Scheduler.register('exp-decay-scheduler')
class ExponentialDecayScheduler(Scheduler):
    """ Starts from upper"""

    def __init__(self, upper: float, lower: float, max_steps: int):
        super().__init__()
        self.upper = upper
        self.lower = lower
        self.max_steps = max_steps
        self.a = math.log(self.upper / self.lower) / max_steps

    def _calculate(self, step_value: float) -> float:
        v = self.upper * math.exp(-step_value * self.a)

        return v

    def step(self, metric: float = None, epoch: int = None) -> Optional[float]:
        if epoch is None:
            raise ValueError("Scheduler needs epoch")

        if epoch > self.max_steps:
            logger.info(
                "epoch ({}) exceeded max_steps ({}) for the scheduler. "
                "Not updating the parameter.".format(epoch, self.max_steps))

            return self._current_value
        self._current_value = self._calculate(epoch)
        self.current_step += 1

        return self._current_value


@Scheduler.register('exp-increase-scheduler')
class ExponentialIncreaseScheduler(ExponentialDecayScheduler):
    """Starts from lower"""

    def _calculate(self, step_value: float) -> float:
        v = self.lower * math.exp(step_value * self.a)

        return v
