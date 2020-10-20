from .file_reader import FileReader
from pathlib import Path
from typing import Union, Iterable, Any, List, Tuple, Optional
import pickle
import itertools
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class JustNumberReader(FileReader):
    """Reads just the number in the first line of the file.

    Sometimes just this is enough. For instance when we just
    want to read the ids of entities but not the exact entities"""

    def read(self, filename: Path) -> Iterable[int]:
        with open(filename) as f:
            try:
                for line in f:
                    v = int(line)

                    break
            except Exception as e:
                raise IOError(
                    "File should have an integer entry in first line but is {}"
                    .format(line)) from e

        return range(v)


@FileReader.register('entity-id-reader')
class EntityIdReader(JustNumberReader):
    filename = 'entity2id.txt'


class SamplesIdReader(FileReader):
    """ Reads samples from Openke files assuming the following structure

    numsamples (int)
    head_entity_id tail_id relation_id
    ...            ...         ...
    """
    filename = 'override_this'

    def read(self, filename: Path) -> Iterable[Tuple[int, int, int]]:

        with open(filename) as f:
            # read number of samples
            try:
                for line in f:
                    num_samples = int(line)

                    break
            except Exception as e:
                raise IOError(
                    "Format of first line in file {} not as expected".format(
                        filename)) from e
            # read the actual samples
            samples = []

            for i, line in enumerate(f):
                samples.append(tuple(int(idx) for idx in line.split()))

            if len(samples) != num_samples:
                raise IOError(
                    "Number of samples in the file {} "
                    "does not match the number given in the first line".format(
                        filename))

            return samples


class ClassificationSamplesIdReader(SamplesIdReader):
    """ Reads samples from Openke files assuming the following structure

    numsamples (int)
    pos_head_entity_id pos_tail_id pos_relation_id neg_head_entity_id neg_tail_entity_id neg_relation_id
    ...            ...         ...
    """

    def read(self,
             filename: Path) -> Iterable[Tuple[int, int, int, int, int, int]]:

        return super().read(filename)


@FileReader.register('train-id-reader')
class TrainIdReader(SamplesIdReader):
    filename = 'train2id.txt'


@FileReader.register('val-id-reader')
class ValIdReader(SamplesIdReader):
    filename = 'valid2id.txt'


@FileReader.register('test-id-reader')
class TestIdReader(SamplesIdReader):
    filename = 'test2id.txt'


RankValidationSampleT = Tuple[
    int, int, List[int],
    int]  # eh (or et), r, negatives of et (or eh) with first entry positive, 0 or 1 (0 for eh 1 for et)
RankValidationSampleHeadAndTailT = Tuple[RankValidationSampleT,
                                         RankValidationSampleT]


def get_relation_from_val_sample(
        val_sample: RankValidationSampleHeadAndTailT) -> int:
    assert val_sample[0][1] == val_sample[1][1]

    return val_sample[0][1]


@FileReader.register('rank-val-id-reader')
class RankValIdReader(FileReader):
    files = [Path('valid2id.txt'), Path('train2id.txt'), Path('test2id.txt')]
    valfile = Path('valid2id.txt')
    """Reads the validation file or cache
    to produce samples for rank validation"""

    def cache_file(self, files: List[Path]):
        cache_file = '_'.join(f.name for f in files)
        cache_file += '_cache.pkl'

        return self.dataset_dir / cache_file

    def __call__(self,
                 check_files: Optional[List[Path]] = None,
                 valfile: Optional[Path] = None,
                 entity2idfile: Optional[Path] = None,
                 val_format: bool = True,
                 **kwargs) -> Iterable[RankValidationSampleHeadAndTailT]:

        if check_files is None:
            check_files = self.files

        if valfile is None:
            valfile = self.valfile

            # check if there is cache
        cache_file = self.cache_file(check_files + [valfile])

        if cache_file.exists() and cache_file.is_file():
            with open(cache_file, 'rb') as cf:
                logger.info("Loading rank validation data from cache at {}".
                            format(cache_file))
                samples = pickle.load(cf)
        else:

            entity2idreader = EntityIdReader(self.dataset_dir)

            if entity2idfile is not None:
                entity2idreader.filename = entity2idfile.name
            all_entities = (entity2idreader())

            samples = []
            # read all files
            all_positive = set(
                itertools.chain(*(self.read(self.dataset_dir / f)
                                  for f in check_files)))
            tail_idx = 1
            head_idx = 0
            rel_idx = 2

            def negatives(head_or_tail, relation, which):
                if which == tail_idx:
                    s = [head_or_tail, None, relation]
                else:
                    s = [None, head_or_tail, relation]

                also_check = tuple([head_or_tail, head_or_tail, relation])
                #also_check = []

                for entity in all_entities:
                    possible = deepcopy(s)
                    possible[which] = entity
                    possible = tuple(possible)

                    if possible in all_positive:
                        continue
                    elif possible == also_check:
                        continue
                    else:
                        yield entity

            for val_sample in self.read(self.dataset_dir / valfile):
                tail_replacement_entities = list(
                    itertools.chain(
                        negatives(val_sample[head_idx], val_sample[rel_idx],
                                  tail_idx), [val_sample[tail_idx]]))
                head_replacement_entities = list(
                    itertools.chain(
                        negatives(val_sample[tail_idx], val_sample[rel_idx],
                                  head_idx),
                        [val_sample[head_idx]]))  # create by replacing head
                head_replaced_sample = [
                    val_sample[tail_idx], val_sample[rel_idx],
                    head_replacement_entities, head_idx
                ]
                tail_replaced_sample = [
                    val_sample[head_idx], val_sample[rel_idx],
                    tail_replacement_entities, tail_idx
                ]
                samples.append((tuple(head_replaced_sample),
                                tuple(tail_replaced_sample)))
            # save to cache
            with open(cache_file, 'wb') as cf:
                logger.info(
                    "Writing rank validation cache at {}".format(cache_file))
                pickle.dump(samples, cf)

        return samples

    def read(self, filename: Path) -> Iterable[Tuple[int, int, int]]:

        with open(filename) as f:
            # read number of samples
            try:
                for line in f:
                    num_samples = int(line)

                    break
            except Exception as e:
                raise IOError(
                    "Format of first line in file {} not as expected".format(
                        filename)) from e
            # read the actual samples
            samples = []

            for i, line in enumerate(f):
                samples.append(tuple(int(idx) for idx in line.split()))

            if len(samples) != num_samples:
                raise IOError(
                    "Number of samples in the file {} "
                    "does not match the number given in the first line".format(
                        filename))

        return samples


@FileReader.register('rank-test-id-reader')
class RankTestIdReader(RankValIdReader):
    files = [Path('valid2id.txt'), Path('train2id.txt'), Path('test2id.txt')]
    valfile = Path('test2id.txt')
