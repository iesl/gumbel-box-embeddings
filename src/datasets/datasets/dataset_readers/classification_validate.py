from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from pathlib import Path
from .types import PathT
from ..file_readers.openke import SamplesIdReader, JustNumberReader, ClassificationSamplesIdReader

from typing import Iterable, Optional, List, Tuple
import numpy as np
from allennlp.data.fields import ArrayField, LabelField
import pickle
import itertools
import logging
logger = logging.getLogger(__name__)


@DatasetReader.register('classification-validation-dataset')
class ClassificationValidationDatasetReader(DatasetReader):
    """ Expects file of the form:
        number of samples(int)
        head_id(int) tail_id(int) relation_id(int) label(0 or 1)
        ...
        ...
    """

    def __init__(self,
                 dataset_name: Optional[str] = None,
                 all_datadir: PathT = Path('.data'),
                 validation_file: str = 'classification_dev.txt'):

        if dataset_name is None:
            raise ValueError("provide dataset_name")
        self.dataset_name = dataset_name
        self.all_datadir = Path(all_datadir)
        self.validation_file = validation_file
        self.file_reader = SamplesIdReader()

    def sample_to_instance(self,
                           sample: Tuple[int, int, int, int]) -> Instance:
        head = ArrayField(np.array(sample[0], dtype=np.int), dtype=np.int)
        tail = ArrayField(np.array(sample[1], dtype=np.int), dtype=np.int)
        relation = ArrayField(np.array(sample[2], dtype=np.int), dtype=np.int)
        label = ArrayField(np.array(sample[3], dtype=np.int), dtype=np.int)

        return Instance({'h': head, 't': tail, 'r': relation, 'label': label})

    def _read(self, filename=None) -> Iterable[Tuple]:
        fname = None

        if filename is not None:
            if filename.strip() != 'dummy_path':
                fname = filename

        if fname is None:
            return self.file_reader.read(
                self.all_datadir / self.dataset_name / self.validation_file)
        else:
            return self.file_reader.read(fname)

    def read(self, filename=None) -> Iterable[Instance]:
        instances = [self.sample_to_instance(i) for i in self._read(filename)]

        return instances


@DatasetReader.register('classification-with-negs-validation-dataset')
class ClassificationValidationDatasetReader(
        ClassificationValidationDatasetReader):
    def __init__(self,
                 dataset_name: Optional[str] = None,
                 all_datadir: PathT = Path('.data'),
                 validation_file: str = 'classification_valid2id.txt'):
        super().__init__(dataset_name, all_datadir, validation_file)

    def sample_to_instance(
            self, sample: Tuple[int, int, int, int, int, int]) -> Instance:
        pos_head = ArrayField(np.array(sample[0], dtype=np.int), dtype=np.int)
        pos_tail = ArrayField(np.array(sample[1], dtype=np.int), dtype=np.int)
        pos_relation = ArrayField(
            np.array(sample[2], dtype=np.int), dtype=np.int)
        neg_head = ArrayField(np.array(sample[3], dtype=np.int), dtype=np.int)
        neg_tail = ArrayField(np.array(sample[4], dtype=np.int), dtype=np.int)
        neg_relation = ArrayField(
            np.array(sample[5], dtype=np.int), dtype=np.int)
        label = LabelField(0, skip_indexing=True)

        return Instance({
            'p_h': pos_head,
            'p_t': pos_tail,
            'p_r': pos_relation,
            'n_h': neg_head,
            'n_t': neg_tail,
            'n_r': neg_relation,
            'label': label
        })


@DatasetReader.register('classification-validation-dataset-relation-transform')
class ClassificationValidationDatasetReaderRelTransform(
        ClassificationValidationDatasetReader):
    def __init__(self,
                 dataset_name: Optional[str] = None,
                 all_datadir: PathT = Path('.data'),
                 validation_file: str = 'classification_valid2id.txt'):
        super().__init__(dataset_name, all_datadir, validation_file)

    def sample_to_instance(self,
                           sample: Tuple[int, int, int, int, int]) -> Instance:
        head = ArrayField(np.array(sample[0], dtype=np.int), dtype=np.int)
        tail = ArrayField(np.array(sample[1], dtype=np.int), dtype=np.int)
        relation_head = ArrayField(np.array(sample[2], dtype=np.int), dtype=np.int)
        relation_tail = ArrayField(np.array(sample[3], dtype=np.int), dtype=np.int)
        label = ArrayField(np.array(sample[4], dtype=np.int), dtype=np.int)

        return Instance(
            {'h': head,
             't': tail,
             'r_h': relation_head,
             'r_t': relation_tail,
             'label': label})
