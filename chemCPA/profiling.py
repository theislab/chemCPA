import os
import shutil
import signal
import subprocess
from pathlib import Path

import sacred


class Profiler:
    outpath: Path
    _process: subprocess.Popen

    def __init__(self, seed: str, save_dir: str):
        """
        Creates a new profiler without start it yet.
        @param seed: random string used for generating unique filepath.
        @param save_dir: directory to save the file to.
        """
        assert Path(save_dir).is_dir(), f"{save_dir} is not a directory!"
        self.outpath = Path(save_dir) / f"profile_{seed}.speedscope"
        assert shutil.which("py-spy"), "py-spy not found, please install it first."

    def start(self):
        """Start recording the current Python process"""
        # starts py-spy in a new subprocess
        self._process = subprocess.Popen(
            [
                shutil.which("py-spy"),
                "record",
                "--pid",
                str(os.getpid()),  # tells py-spy to profile the current Python process
                "--rate",
                "3",  # three samples per second should be fine-grained enough and the outfile won't get too large
                "--format",
                "speedscope",  # look at profiles via https://speedscope.app
                "--output",
                str(
                    self.outpath
                ),  # file to save results at (once profiling has finished)
            ]
        )

    def stop(self, experiment: sacred.Experiment):
        """
        Stop recording and save the results to a file and to MongoDB
        @param experiment: The seml / sacred experiment.
        """
        # First, send same signal as CTRL+C would. Py-spy should quit and save the results.
        self._process.send_signal(signal.SIGINT)
        try:
            # if the profiler didn't exit after 10s, kill it
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # sends SIGKILL. py-spy will quit, but will not save a profile.
            self._process.kill()
            print("killed py-spy due to timeout.")
            # collect the zombie process
            self._process.wait(timeout=2)

        # upload the profiling results to mongoDB as a binary
        if self.outpath.is_file():
            experiment.add_artifact(
                str(self.outpath),
                name="py_spy_profile",
                content_type="application/octet-stream",
            )
