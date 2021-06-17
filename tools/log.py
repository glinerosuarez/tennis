import os
import time
import json
import torch
import atexit
import joblib
import warnings
import numpy as np
import pandas as pd
import os.path as osp
from config import settings
from agents import ActorCritic
from typing import Dict, Optional
from tools import mpi, serialization
from matplotlib import pyplot as plt


def setup_logger_kwargs(
        exp_name: str, seed: int = None, data_dir: str = None, datestamp: bool = False
) -> Dict[str, str]:
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs. If no seed is given and datestamp is
    false, output_dir = data_dir/exp_name. If a seed is given and datestamp is false,
    output_dir = data_dir/exp_name/exp_name_s[seed]. If datestamp is true, amend to
    output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]
    Args:
        exp_name: Name for experiment.
        seed: Seed for random number generators used by experiment.
        data_dir: Path to folder where results should be saved.
        datestamp: Whether to include a date and timestamp in the name of the save directory.
    Returns:
        logger_kwargs, a dict containing output_dir and exp_name.
    """

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])

    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)

    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath), exp_name=exp_name)
    return logger_kwargs


def colorize(string: str, color: str, bold: bool = False, highlight: bool = False) -> str:
    """
    Colorize a string.
    """
    color2num = dict(
        gray=30,
        red=31,
        green=32,
        yellow=33,
        blue=34,
        magenta=35,
        cyan=36,
        white=37,
        crimson=38
    )

    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class Logger:
    """
    A general-purpose logger. Makes it easy to save diagnostics, hyperparameter configurations, the state of a training
    run, and the trained model.
    """

    def __init__(self, output_dir: str = None, output_fname: str = 'progress.txt', exp_name: str = None):
        """
        Initialize a Logger.
        Args:
            output_dir:     A directory for saving results to. If ``None``, defaults to a temp directory of the form
                            ``/tmp/experiments/somerandomnumber``.
            output_fname:   Name for the tab-separated-value file containing metrics logged throughout a training run.
                            Defaults to ``progress.txt``.
            exp_name:       Experiment name. If you run multiple training runs and give them all the same ``exp_name``,
                            the plotter will know to group them. (Use case: if you run the same hyperparameter
                            configuration with multiple random seeds, you should give them all the same ``exp_name``.)
        """
        if mpi.proc_id() == 0:
            self.output_dir = output_dir or "/tmp/experiments/%i" % int(time.time())
            if osp.exists(self.output_dir):
                print("Warning: Log dir %s already exists! Storing info there anyway." % self.output_dir)
            else:
                os.makedirs(self.output_dir)
            self.output_file_path = osp.join(self.output_dir, output_fname)
            self.output_file = open(self.output_file_path, 'w')
            atexit.register(self.output_file.close)
            print(colorize("Logging data to %s" % self.output_file.name, 'green', bold=True))
        else:
            self.output_dir = None
            self.output_file = None
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg: str, color: str = 'green') -> None:
        """ Print a colorized message to stdout."""
        if mpi.proc_id() == 0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key: str, val: float) -> None:
        """
        Log a value of some diagnostic. Call this only once for each diagnostic quantity, each iteration. After
        using ``log_tabular`` to store values for each diagnostic, make sure to call ``dump_tabular`` to write them out
        to file and stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            if key not in self.log_headers:
                raise AssertionError(f"Trying to introduce a new key {key} that you didn't include in the first iteration")
        assert key not in (
            self.log_current_row,
            "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
        )
        self.log_current_row[key] = val

    def save_config(self, config) -> None:
        """
        Log an experiment configuration. Call this once at the top of your experiment, passing in all important config
        vars as a dict. This will serialize the config to JSON, while handling anything which can't be serialized in a
        graceful way (writing as informative a string as possible).
        Example use:
        .. code-block:: python
            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = serialization.convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        if mpi.proc_id() == 0:
            output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
            print(colorize('Saving config:\n', color='cyan', bold=True))
            print(output)
            with open(osp.join(self.output_dir, "config.json"), 'w') as out:
                out.write(output)

    def save_state(self, state_dict: Dict, itr: Optional[int] = None):
        """
        Saves the state of an experiment. To be clear: this is about saving *state*, not logging diagnostics. All
        diagnostic logging is separate from this function. This function will save whatever is in
        ``state_dict``---usually just a copy of the environment---and the most recent parameters for the model you
        previously set up saving for with ``setup_pytorch_saver``. Call with any frequency you prefer. If you only want to
        maintain a single state and overwrite it at each call with the most recent version, leave ``itr=None``.
        If you want to keep all of the states you save, provide unique (increasing) values for 'itr'.
        Args:
            state_dict: Dictionary containing essential elements to describe the current state of training.
            itr: Current iteration of training.
        """
        if mpi.proc_id() == 0:
            fname = 'vars.pkl' if itr is None else 'vars%d.pkl' % itr
            try:
                joblib.dump(state_dict, osp.join(self.output_dir, fname))
            except Exception as e:
                self.log('Warning: could not pickle state_dict.', color='red')
                self.log(repr(state_dict), color='red')
                self.log(repr(type(state_dict)), color='red')
                self.log(e, color='red')
            if hasattr(self, 'pytorch_saver_elements'):
                self._pytorch_simple_save(itr)

    def setup_pytorch_saver(self, agent: ActorCritic) -> None:
        """
        Set up easy model saving for a single PyTorch model. Because PyTorch saving and loading is especially painless,
        this is very minimal; we just need references to whatever we would like to pickle. This is integrated into the
        logger because the logger knows where the user would like to save information about this training run.
        Args:
            agent: PyTorch model or serializable object containing PyTorch models.
        """
        self.pytorch_saver_elements = {
            "policy_state_dict": agent.pi.state_dict(),
            "value_state_dict": agent.v.state_dict()
        }

    def _pytorch_simple_save(self, itr: int = None) -> None:
        """ Saves the PyTorch model (or models). """
        if mpi.proc_id() == 0:
            assert hasattr(self, 'pytorch_saver_elements'), \
                "First have to setup saving with self.setup_pytorch_saver"
            fpath = 'pyt_save'
            fpath = osp.join(self.output_dir, fpath)
            fname = 'model' + ('%d' % itr if itr is not None else '') + '.pt'
            fname = osp.join(fpath, fname)
            os.makedirs(fpath, exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.save(self.pytorch_saver_elements, fname)

    def dump_tabular(self):
        """ Write all of the diagnostics from the current iteration. Writes both to stdout, and to the output file. """
        if mpi.proc_id() == 0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15, max(key_lens))
            keystr = '%' + '%d' % max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            print("-" * n_slashes)
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                valstr = "%8.3g" % val if hasattr(val, "__float__") else val
                print(fmt % (key, valstr))
                vals.append(val)
            print("-" * n_slashes, flush=True)
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write("\t".join(self.log_headers) + "\n")
                self.output_file.write("\t".join(map(str, vals)) + "\n")
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs. Typical use case: there is some quantity which
    is calculated many times throughout an epoch, and at the end of the epoch, you would like to report the
    average / std / min / max value of that quantity. With an EpochLogger, each time the quantity is calculated, you
    would use
    .. code-block:: python
        epoch_logger.store(NameOfQuantity=quantity_value)
    to load it into the EpochLogger's state. Then at the end of the epoch, you would use
    .. code-block:: python
        epoch_logger.log_tabular(NameOfQuantity, **options)
    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state. Provide an arbitrary number of keyword arguments with
        numerical values.
        """
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key: str, val: float = None, with_min_and_max: bool = False, average_only: bool = False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.
        Args:
            key: The name of the diagnostic. If you are logging a diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.
            val: A value for the diagnostic. If you have previously saved values for this key via ``store``, do *not*
            provide a ``val`` here.
            with_min_and_max: If true, log min and max values of the diagnostic over the epoch.
            average_only: If true, do not log the standard deviation of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
            stats = mpi.mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else 'Average' + key, stats[0])
            if not average_only:
                super().log_tabular('Std' + key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max' + key, stats[3])
                super().log_tabular('Min' + key, stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """ Lets an algorithm ask the logger for mean/std/min/max of a diagnostic."""
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
        return mpi.mpi_statistics_scalar(vals)

    def plot_rewards(self):
        """Plot average rewards of the last ``settings.PPO.epochs_mean_rewards`` epochs"""
        print(f"output file: ", self.output_file_path)
        progress_df = pd.read_csv(self.output_file_path, sep="\t")
        print(progress_df)
        scores = progress_df[f'AvgRewardsLast{settings.PPO.epochs_mean_rewards}Ep']

        # create plot
        fig: plt.Figure() = plt.figure()
        ax: plt.axes.SubplotBase = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')

        # save to file
        plt.savefig(osp.join(self.output_dir, "scores.png"))
