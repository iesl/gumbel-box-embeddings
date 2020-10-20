from allennlp.common.registrable import Registrable
from typing import Any, Iterable, Tuple, Union, List
from ..types import NegativeSamplerProtocol

from itertools import zip_longest


def zip_equal(*iterables):
    sentinel = object()

    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError('Iterables have different lengths')
        yield combo


class NegtiveSampler(Registrable, NegativeSamplerProtocol):
    def generate_one_negative_sample(self, sample: Tuple,
                                     replacement_index: int, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self,
                 positive_sample: Tuple,
                 N: int = 1,
                 replacement_index: Union[Iterable[int], int] = 0,
                 **kwargs) -> Iterable[Any]:

        if isinstance(replacement_index, int):
            replacement_index = [replacement_index] * N
        try:
            for i, ri in zip_equal(range(N), replacement_index):
                yield self.generate_one_negative_sample(
                    positive_sample, ri, **kwargs)
        except TypeError as te:
            raise TypeError(
                "replacement_index should be iterable or int") from te
        except ValueError as ve:
            raise ValueError(
                "lenght of replacement_index not equal to N") from ve

    def sample(self, positive_sample: Tuple, replacement_index: int,
               **kwargs) -> Any:

        return list(
            self.__call__(
                positive_sample,
                N=1,
                replacement_index=replacement_index,
            ))[0]
