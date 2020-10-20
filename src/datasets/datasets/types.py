from typing import Any, Iterable, Union, Tuple, List
from typing_extensions import Protocol


class NegativeSamplerProtocol(Protocol):
    def __call__(positive_sample: Tuple,
                 N: int = 1,
                 replacement_index: Union[List[int], int] = 0,
                 **kwargs) -> Iterable[Any]:
        pass
