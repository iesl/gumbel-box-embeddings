from pathlib import Path
from typing import Any


def fill_relation(input_file: Path,
                  output_file: Path,
                  fill_using: Any = 0,
                  skip_firstline=True,
                  input_delimiter=None,
                  output_delimiter=None):
    with open(input_file) as inp:
        if skip_firstline:
            inp.__next__()

        if input_delimiter is None:

            def inp_split(l):
                return l.split()
        else:

            def inp_split(l):
                return l.split(input_delimiter)

        if output_delimiter is None:
            output_delimiter = ' '
        with open(output_file, 'w') as out:
            for line in inp:
                output_line = inp_split(line)
                output_line.append(str(fill_using))
                output_line.append('\n')
                out.write(output_delimiter.join(output_line))
