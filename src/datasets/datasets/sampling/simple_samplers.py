from .negative_sampler import NegtiveSampler
from typing import (List, Any, Iterable, Callable, Hashable, MutableSet,
                    Optional, Set, Tuple)
import numpy as np
import itertools
from copy import deepcopy


@NegtiveSampler.register('simple-negative-sampler')
class UniformNegativeSampler(NegtiveSampler):
    sampler: Callable[[List[Any]], Any] = np.random.choice
    """Negative sampler which check against itself and a give hashtable"""

    def __init__(self,
                 entities: Optional[Iterable[Hashable]] = None,
                 all_positive: Optional[Set[Tuple]] = None,
                 max_attempts: int = 10000):
        self.entities_list = list(
            entities
        ) if entities is not None else None  # required by np.choice()
        self.max_attempts = max_attempts
        self.all_positive = all_positive if all_positive is not None else None

    @property
    def entities(self):
        return self.entities_list

    @entities.setter
    def entities(self, v: Iterable[Hashable]):
        self.entities_list = list(v)

    @property
    def positives(self):
        return self.all_positive

    @positives.setter
    def positives(self, v: Set[Tuple]):
        self.all_positive = v

    def generate_one_negative_sample(
            self,
            sample: Tuple,  # tuple
            replacement_index: int,
    ):
        """Generates one negative sample from self.entities and checks it against itself (sample)
        and also_check

        .. note:: It is the caller's responcibility to efficiently maintain and update also_check
        """

        if self.all_positive is None:
            raise RuntimeError

        for attempt_num in itertools.count():
            a_sample_entity = self.sampler(self.entities_list)

            if attempt_num > self.max_attempts:
                raise RuntimeError(
                    "Could not find a negative sample in {} attempsts".format(
                        self.max_attempts))

            if a_sample_entity == sample[replacement_index]:
                continue
            else:
                # check in all positives
                potential_negative_sample = list(deepcopy(sample))
                potential_negative_sample[replacement_index] = a_sample_entity

                if tuple(potential_negative_sample) in self.all_positive:
                    continue

                if potential_negative_sample[0] == potential_negative_sample[
                        1]:

                    if replacement_index == 1:
                        replacement_index = 0
                    else:
                        replacement_index = 1

                    continue

                return tuple(potential_negative_sample)
