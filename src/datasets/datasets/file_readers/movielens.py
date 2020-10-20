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


@FileReader.register('mv-entity-id-reader')
class EntityIdReader(JustNumberReader):
    filename = 'entity2id.txt'


class SamplesIdReader(FileReader):
    """ Reads samples from Openke files assuming the following structure

    numsamples (int)
    head_entity_id tail_id relation_id
    ...            ...         ...
    """
    filename = 'override_this'

    def read(self, filename: Path) -> Iterable[Tuple[int, int, float]]:

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
                l = line.split()
                samples.append(tuple([int(l[0]), int(l[1]), float(l[2])]))

            if len(samples) != num_samples:
                raise IOError(
                    "Number of samples in the file {} "
                    "does not match the number given in the first line".format(
                        filename))

            return samples


@FileReader.register('mv-train-id-reader')
class TrainIdReader(SamplesIdReader):
    filename = 'train2id.txt'


@FileReader.register('mv-val-id-reader')
class ValIdReader(SamplesIdReader):
    filename = 'valid2id.txt'


@FileReader.register('mv-test-id-reader')
class TestIdReader(SamplesIdReader):
    filename = 'test2id.txt'
