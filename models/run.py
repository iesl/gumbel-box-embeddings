#!/usr/bin/env python

import logging
import os
import sys

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(
    0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL)

from allennlp.commands import main  # noqa
from models.commands import PredictForErrorAnalysis  # noqa
# add project specific subcommands

subcommands = {"error-analysis": PredictForErrorAnalysis()}


def run():
    main(prog="allennlp", subcommand_overrides=subcommands)


if __name__ == "__main__":
    run()
