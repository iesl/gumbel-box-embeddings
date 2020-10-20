from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import ArrayField, LabelField
from allennlp.common.params import Params
from typing import Iterable, Callable, Optional, Tuple, List
from .constants import Mode
from .types import PathT, NegativeSamplerProtocol
from ..file_readers.movielens import TrainIdReader, EntityIdReader, ValIdReader, SamplesIdReader
from ..sampling.simple_samplers import UniformNegativeSampler
from pathlib import Path
from .lazy_iterators import (LazyIteratorWithNegativeSampling,
                             LazyIteratorWithSequentialNegativeSampling)
import numpy as np
import logging
logger = logging.getLogger(__name__)
SampleT = Tuple[int, int, float]


class MovielensDataset(DatasetReader):
    """Base class for movielens style datasets. Do not create instance
    directly"""

    def __init__(self,
                 dataset_name: str = 'movielens',
                 all_datadir: PathT = Path('.data'),
                 mode: str = Mode.train,
                 lazy: bool = False):
        super().__init__(lazy)
        self.dataset_name = dataset_name
        self.all_datadir = Path(all_datadir)
        self.mode = mode

        if self.mode == Mode.train:
            self.file_reader = TrainIdReader(self.all_datadir / dataset_name)
        elif self.mode == Mode.validate:
            self.file_reader = ValIdReader(self.all_datadir / dataset_name)


@DatasetReader.register("movielens-dataset")
class MovielensDatasetReader(MovielensDataset):
    def __init__(self,
                 dataset_name: str = 'movielens',
                 all_datadir: PathT = Path('.data'),
                 mode: str = Mode.train):
        """ This reader does not support lazy because we want to be
        fast"""
        super().__init__(dataset_name, all_datadir, mode, lazy=False)
        logger.warn("THIS DATASET READER DOES NOT DO NEGATIVE SAMPLING")
        logger.warn(
            "IT IS THE DOWNSTREAM MODEL'S OR ITERATOR'S RESPONCIBILITY "
            "TO DO NEG SAMPLING, IF REQUIRED")

    def sample_to_instance(self, sample: SampleT):
        """ Always expect one positive sample"""
        pos_head = ArrayField(np.array(sample[0], dtype=np.int), dtype=np.int)
        pos_relation = ArrayField(np.array(0, dtype=np.int), dtype=np.int)
        pos_tail = ArrayField(np.array(sample[1], dtype=np.int), dtype=np.int)
        label = ArrayField(np.array(sample[2], dtype=np.float), dtype=np.float)
        fields = {
            'h': pos_head,
            't': pos_tail,
            'r': pos_relation,
            'label': label
        }

        return Instance(fields)

    def _read(self, filename=None) -> List[Instance]:
        logger.info("Reading data from file")
        samples = self.file_reader()
        instances = []
        logger.info("Creating instances from samples")

        for sample in samples:
            instances.append(self.sample_to_instance(sample))

        return instances

    def read(self, filename=None):
        return super().read(filename)

@DatasetReader.register("movielens-dataset-validation")
class MovielensDatasetReader_valid(MovielensDatasetReader):
    def __init__(self,
                 dataset_name: str = 'movielens',
                 all_datadir: PathT = Path('.data'),
                 mode: str = Mode.validate):
        """ This reader does not support lazy because we want to be
        fast"""
        super().__init__(dataset_name, all_datadir, mode)


if __name__ == "__main__":
    test_data_path = Path(
        'dummy_path'
    )
    OpenKEDatasetReader(all_datadir=test_data_path)
    instances = OpenKEDatasetReader.read()
