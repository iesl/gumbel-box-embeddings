from typing import List, Tuple, Iterable


def samples_write(samples: Iterable[Iterable], filename: str):
    num_samples = len(samples)
    with open(filename, 'w') as f:
        f.write(str(num_samples))

        for line in samples:
            f.write('\n')
            f.write(' '.join([str(value) for value in line]))
