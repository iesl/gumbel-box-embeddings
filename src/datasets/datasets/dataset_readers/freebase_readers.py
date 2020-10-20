from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import ArrayField, LabelField
from allennlp.common.params import Params
from typing import Iterable, Callable, Optional, Tuple, List
from .constants import Mode
from .types import PathT, NegativeSamplerProtocol
from ..file_readers.openke import TrainIdReader, EntityIdReader, ValIdReader, SamplesIdReader
from ..sampling.simple_samplers import UniformNegativeSampler
from pathlib import Path
from .lazy_iterators import (LazyIteratorWithNegativeSampling,
                             LazyIteratorWithSequentialNegativeSampling)
import numpy as np
import logging
logger = logging.getLogger(__name__)
SampleT = Tuple[int, int, int]
SampleQ = Tuple[int, int, int, int]


class OpenKEDataset(DatasetReader):
    """Base class for openke style datasets. Do not create instance
    directly"""

    def __init__(self,
                 dataset_name: str = 'FB15K237',
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


@DatasetReader.register("openke-dataset-negative-sampling")
class OpenKEDatasetReaderWithNegativeSampling(OpenKEDataset):
    """OpenKE style KB completion dataset reader
    """
    lazy_iter = LazyIteratorWithNegativeSampling

    def __init__(self,
                 dataset_name: str = 'FB15K237',
                 all_datadir: PathT = Path('.data'),
                 mode: str = Mode.train,
                 positive_files: Optional[List[str]] = None,
                 number_negative_samples: int = 1):
        super().__init__(dataset_name, all_datadir, mode, lazy=True)
        self.positive_files = positive_files

        if self.positive_files is None:
            self.positive_files = [
                'train2id.txt', 'valid2id.txt', 'test2id.txt'
            ]

        self.number_negative_samples = number_negative_samples

        self.negative_sampler = UniformNegativeSampler()

    def generate_replacement_index(self):
        return np.random.choice([0, 1])  # (eh,et,r)

    def replacement_index_generator(self):
        for i in range(self.number_negative_samples):
            yield self.generate_replacement_index()

    def _read(self, filename=None) -> Iterable[Instance]:
        """Read positive samples

        Arguments:

            dirname: Could be parent directory containing all dataset
                or the directory containing a particular dataset
        """

        return self.file_reader()

    def samples_to_instance(self, positive_sample: SampleT,
                            negative_sample: SampleT) -> Instance:
        pos_head = ArrayField(
            np.array(positive_sample[0], dtype=np.int), dtype=np.int)
        pos_relation = ArrayField(
            np.array(positive_sample[2], dtype=np.int), dtype=np.int)
        pos_tail = ArrayField(
            np.array(positive_sample[1], dtype=np.int), dtype=np.int)
        neg_head = ArrayField(
            np.array(negative_sample[0], dtype=np.int), dtype=np.int)
        neg_relation = ArrayField(
            np.array(negative_sample[2], dtype=np.int), dtype=np.int)
        neg_tail = ArrayField(
            np.array(negative_sample[1], dtype=np.int), dtype=np.int)
        label = LabelField(
            1, skip_indexing=True)  # first one is always the pos sample
        fields = {
            "p_h": pos_head,
            "p_r": pos_relation,
            "p_t": pos_tail,
            "n_h": neg_head,
            "n_r": neg_relation,
            "n_t": neg_tail,
            "label": label
        }

        return Instance(fields)

    def single_negative_sampler_generator(self):
        return lambda pos: self.negative_sampler.sample(pos,
                                                        self.generate_replacement_index())

    def negative_sampler_generator(self):
        return lambda pos: self.negative_sampler(
            pos, N=self.number_negative_samples,
            replacement_index=self.replacement_index_generator())

    def read(self, filename=None) -> Iterable[Instance]:
        """Lazyly return instances by negatively sampling"""
        all_pos_lists = [
            SamplesIdReader().read(self.all_datadir / self.dataset_name / f)

            for f in self.positive_files
        ]
        positive_samples = set(
            [item for sublist in all_pos_lists for item in sublist])
        all_entities = list(
            EntityIdReader(self.all_datadir / self.dataset_name)())
        self.negative_sampler.entities = all_entities
        self.negative_sampler.positives = positive_samples

        return self.lazy_iter(self.negative_sampler_generator(),
                              positive_samples, self.samples_to_instance)


@DatasetReader.register("openke-classification-dataset-negative-sampling")
class OpenKEClassificationDatasetReaderWithNegativeSampling(
        OpenKEDatasetReaderWithNegativeSampling):
    lazy_iter = LazyIteratorWithSequentialNegativeSampling

    def samples_to_instance(self, sample: SampleT, label: int) -> Instance:
        head = ArrayField(np.array(sample[0], dtype=np.int), dtype=np.int)
        relation = ArrayField(np.array(sample[2], dtype=np.int), dtype=np.int)
        tail = ArrayField(np.array(sample[1], dtype=np.int), dtype=np.int)
        label_f = LabelField(label, skip_indexing=True)
        fields = {'h': head, 't': tail, 'r': relation, 'label': label_f}

        return Instance(fields)


@DatasetReader.register("openke-dataset")
class OpenKEDatasetReader(OpenKEDataset):
    def __init__(self,
                 dataset_name: str = 'FB15K237',
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
        pos_relation = ArrayField(
            np.array(sample[2], dtype=np.int), dtype=np.int)
        pos_tail = ArrayField(np.array(sample[1], dtype=np.int), dtype=np.int)
        label = LabelField(
            1, skip_indexing=True)  # first one is always the pos sample
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


@DatasetReader.register("openke-dataset-max-margin")
class OpenKEDatasetReaderMaxMargin(OpenKEDatasetReader):
    def sample_to_instance(
            self,
            sample: SampleT,
    ) -> Instance:
        positive_sample = sample[0:3]
        negative_sample = sample[3:6]
        pos_head = ArrayField(
            np.array(positive_sample[0], dtype=np.int), dtype=np.int)
        pos_relation = ArrayField(
            np.array(positive_sample[2], dtype=np.int), dtype=np.int)
        pos_tail = ArrayField(
            np.array(positive_sample[1], dtype=np.int), dtype=np.int)
        neg_head = ArrayField(
            np.array(negative_sample[0], dtype=np.int), dtype=np.int)
        neg_relation = ArrayField(
            np.array(negative_sample[2], dtype=np.int), dtype=np.int)
        neg_tail = ArrayField(
            np.array(negative_sample[1], dtype=np.int), dtype=np.int)
        label = LabelField(
            1, skip_indexing=True)  # first one is always the pos sample
        fields = {
            "p_h": pos_head,
            "p_r": pos_relation,
            "p_t": pos_tail,
            "n_h": neg_head,
            "n_r": neg_relation,
            "n_t": neg_tail,
            "label": label
        }

        return Instance(fields)


@DatasetReader.register("openke-dataset-relation-transform")
class OpenKEDatasetReaderRelTransform(OpenKEDatasetReader):
    def sample_to_instance(
            self,
            sample: SampleQ,
    ) -> Instance:
        """ Always expect one positive sample"""
        pos_head = ArrayField(np.array(sample[0], dtype=np.int), dtype=np.int)
        pos_relation_head = ArrayField(
            np.array(sample[2], dtype=np.int), dtype=np.int)
        pos_relation_tail = ArrayField(
            np.array(sample[3], dtype=np.int), dtype=np.int)
        pos_tail = ArrayField(np.array(sample[1], dtype=np.int), dtype=np.int)
        label = LabelField(
            1, skip_indexing=True)  # first one is always the pos sample

        fields = {
            'h': pos_head,
            't': pos_tail,
            'r_h': pos_relation_head,
            'r_t': pos_relation_tail,
            'label': label
        }

        return Instance(fields)


if __name__ == "__main__":
    test_data_path = Path(
        'dummy_path'
    )
    OpenKEDatasetReader(all_datadir=test_data_path)
    instances = OpenKEDatasetReader.read()
