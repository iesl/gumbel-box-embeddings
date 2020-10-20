from allennlp.data.iterators import BasicIterator, DataIterator
from typing import Tuple


@DataIterator.register('single-sample-rank-validation-iterator')
class SingleSampleRankValidationIterator(BasicIterator):
    """ The only job of this special iterator is to make sure
    that the batch size is 1. Because the validation logic depends on
    it"""

    def __init__(
            self,
            batch_size: int = 1,
            instances_per_epoch: int = None,
            max_instances_in_memory: int = None,
            cache_instances: bool = False,
            track_epoch: bool = False,
            maximum_samples_per_batch: Tuple[str, int] = None,
    ):
        batch_size = 1
        super().__init__(
            batch_size=batch_size, cache_instances=cache_instances)
