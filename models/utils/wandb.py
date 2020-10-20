from typing import Dict, List
import wandb
import logging
from pathlib import Path
import shutil
import tempfile
import re
logger = logging.getLogger(__name__)


class RunData(object):
    def __init__(self,
                 run_id: str,
                 group: str = None,
                 project: str = None,
                 download_dir: str = None):
        """
            Arguments:
                run_id: wandb runid of the run

                group: wandb group of the project for the run

                project: wandb project

                download_dir: Path to dir where runs files should be downloaded.
                    If not specified, a temp dir will be created
        """
        self.api = wandb.Api()
        components = []

        if group is not None:
            components.append(group)

        if project is not None:
            components.append(project)
        components.append('runs')
        components.append(run_id)
        self.run_path = '/'.join(components)
        self.run = self.api.run(self.run_path)
        self.using_temp = False

        if download_dir is None:
            self.download_dir = Path(
                tempfile.mkdtemp(suffix=None, prefix=None, dir=None))
            logger.info("Created temp dir {}".format(self.download_dir))
            self.using_temp = True
        else:
            self.download_dir = Path(download_dir) / run_id
        logger.info("Setting up download dir as {}".format(self.download_dir))

    def download_files(self, replace=False,
                       skip_patterns: List[str] = None) -> None:

        for file in self.run.files():
            skip = False

            if skip_patterns is not None:
                for pattern in skip_patterns:
                    m = re.match(pattern, file.name)

                    if m:
                        skip = True

            if skip:
                logger.debug("Skipping {}".format(file.name))

                continue

            try:
                logger.info("Downloading {}".format(file.name))
                file.download(root=self.download_dir, replace=replace)
            except (wandb.CommError, ValueError) as e:
                logger.debug(
                    "{} already present. Not redownloading. Set replace=True to redownload"
                    .format(file.name))

    def upload_file(self, file_on_local: str):
        wandb.save(file_on_local)

    def __del__(self) -> None:
        if self.using_temp:
            logger.info("Removing {}".format(self.download_dir))
            shutil.rmtree(self.download_dir)
