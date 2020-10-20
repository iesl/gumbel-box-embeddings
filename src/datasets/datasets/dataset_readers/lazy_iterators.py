from typing import Iterable, Iterator, Any, Callable
from .types import PathT, NegativeSamplerProtocol
from allennlp.data.instance import Instance


class LazyIteratorWithNegativeSampling(Iterable):
    """ Every call to __iter__ will dynamically generate
    the list of samples.

    .. note:: iter() is called once when a statement like
    the following is encountered:
        for element in iterable:
            ...
    This will return an iterator to the for by calling __iter__
    method.
    """

    def __init__(self, negative_sampler: Callable[[Any], Any],
                 positive_samples: Iterable[Any],
                 samples_to_instance: Callable[[Any, Any], Instance]):
        self.negative_sampler = negative_sampler
        self.positive_samples = positive_samples
        self.samples_to_instance = samples_to_instance

    def __iter__(self):
        for positive_sample in self.positive_samples:
            for negative_sample in self.negative_sampler(positive_sample):
                yield self.samples_to_instance(positive_sample,
                                               negative_sample)


class LazyIteratorWithSequentialNegativeSampling(
        LazyIteratorWithNegativeSampling):
    """Same as parent except that instead of generating (pos_sample, neg_sample),
    it will generate two samples one after the other, one pos and negatives after that.

    .. note:: The change in the signature of samples_to_instance Callable.
    """

    def __init__(self, negative_sampler: Callable[[Any], Any],
                 positive_samples: Iterable[Any],
                 samples_to_instance: Callable[[Any, int], Instance]):
        super().__init__(negative_sampler, positive_samples,
                         samples_to_instance)

    def __iter__(self):
        for positive_sample in self.positive_samples:
            yield self.samples_to_instance(positive_sample, 1)

            for negative_sample in self.negative_sampler(positive_sample):
                yield self.samples_to_instance(negative_sample, 0)


class LazyIteratorWithSingleNegativeSampling(LazyIteratorWithNegativeSampling):
    """ Every call to __iter__ will dynamically generate
    the list of samples.

    .. note:: iter() is called once when a statement like
    the following is encountered:
        for element in iterable:
            ...
    This will return an iterator to the for by calling __iter__
    method.
    """

    def __iter__(self):
        for positive_sample in self.positive_samples:
            negative_sample = self.negative_sampler(positive_sample)
            yield self.samples_to_instance(positive_sample, negative_sample)
